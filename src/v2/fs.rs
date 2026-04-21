//! V2 [`Filesystem`] — top-level init / mount / write / read / unmount.
//!
//! Step 6 scope: single-file model (the "root directory inode" is a
//! stand-in for the single user file — hierarchical directories land
//! in step 11). Direct pointers only (step 7 adds indirect). Fixed-
//! size chunking (step 8 swaps in FastCDC). No dedup (step 9), no
//! dirty bitmap (step 10).
//!
//! # Flow
//!
//! **Init** [`Filesystem::init`]:
//! 1. Build [`CeilingSummary`] from the pristine cover.
//! 2. Build [`Allocator`] (full cover free).
//! 3. Compute the anchor placement and reserve its weights in the
//!    allocator so later allocs never hand them back.
//! 4. Allocate + write an empty [`Inode`] (the "root") to a chunk.
//! 5. Build a [`SuperRoot`] pointing at the root inode with
//!    generation = 1.
//! 6. Allocate + write the super-root to its own chunk.
//! 7. [`anchor::init_anchor`] installs both anchor slots.
//!
//! **Mount** [`Filesystem::mount`]:
//! 1. Read the anchor. Use the active (higher-gen valid) slot.
//! 2. Rebuild the allocator; reserve anchor + the currently-referenced
//!    chunks (super-root, root inode, each direct-pointer data chunk)
//!    so subsequent allocs don't clobber live state.
//! 3. Cross-check super-root generation against anchor generation.
//!
//! **Write** [`Filesystem::write`]:
//! 1. Chunk the data into ≤ [`NUM_DIRECT`] fixed-size pieces.
//! 2. Allocate + write each chunk.
//! 3. Build + allocate + write the new root inode and super-root.
//! 4. [`anchor::commit_anchor`] flips to the new super-root.
//! 5. In-memory state updated; **old chunks are leaked** for this
//!    milestone (garbage collection lands with the free-path dedup /
//!    dirty-bitmap work in later steps).
//!
//! **Read** [`Filesystem::read`]: walk the root inode's direct
//! pointers, concatenate chunks, truncate to `inode.length`.

use thiserror::Error;

use crate::stego::calibration::placement::MetadataPlacement;
use crate::stego::tensor_map::TensorMap;
use crate::v2::alloc::{AllocError, Allocator};
use crate::v2::anchor::{self, AnchorError};
use crate::v2::ceiling::CeilingSummary;
use crate::v2::chunk::{ChunkError, byte_capacity, read_chunk, write_chunk};
use crate::v2::inode::{INODE_BYTES, Inode, InodeError, NUM_DIRECT};
use crate::v2::pointer::{Pointer, PointerError};
use crate::v2::super_root::{SUPER_ROOT_BYTES, SuperRoot, SuperRootError};

const DEFAULT_CHUNK_SIZE_BYTES: u32 = 4096;

/// Top-level V2 filesystem. Owns the cover bytes for the lifetime of
/// the mount; [`Self::unmount`] returns them for persistence.
///
/// Caches the anchor placement so init / mount / write don't rescan
/// every eligible weight on each operation. Without the cache, a 270
/// MB cover pays ~30 s per write (the scan is O(eligible_weights)
/// and dominates the commit); with the cache it's a one-shot cost at
/// init / mount.
#[derive(Debug)]
pub struct Filesystem {
    cover: Vec<u8>,
    map: TensorMap,
    alloc: Allocator,
    anchor_placement: MetadataPlacement,
    generation: u64,
    super_root: SuperRoot,
    super_root_ptr: Pointer,
    root_inode: Inode,
    root_inode_ptr: Pointer,
    data_chunk_size_bytes: u32,
}

#[derive(Debug, Error)]
pub enum FsError {
    #[error("anchor: {0}")]
    Anchor(#[from] AnchorError),

    #[error("alloc: {0}")]
    Alloc(#[from] AllocError),

    #[error("chunk I/O: {0}")]
    Chunk(#[from] ChunkError),

    #[error("inode codec: {0}")]
    Inode(#[from] InodeError),

    #[error("super-root codec: {0}")]
    SuperRoot(#[from] SuperRootError),

    #[error("pointer codec: {0}")]
    Pointer(#[from] PointerError),

    #[error("out of space: could not allocate {requested_bits} bits")]
    OutOfSpace { requested_bits: u32 },

    #[error("invalid data chunk size {chunk_size}; must be greater than zero")]
    InvalidChunkSize { chunk_size: u32 },

    #[error(
        "file of {bytes} bytes needs more than {max_chunks} direct chunks at chunk size {chunk_size}; indirect pointers land in a later step"
    )]
    FileTooLarge {
        bytes: usize,
        max_chunks: usize,
        chunk_size: u32,
    },

    #[error("super-root generation {super_root} disagrees with anchor generation {anchor}")]
    GenerationMismatch { anchor: u64, super_root: u64 },

    #[error("pointer references slot {slot} but map has only {slot_count} slots")]
    PointerSlotOutOfRange { slot: u16, slot_count: usize },

    #[error("pointer targets slot {slot}, which has no stealable bits")]
    PointerTargetsNonStealableSlot { slot: u16 },

    #[error("pointer range [{start_weight}, {end_weight}) lies outside slot {slot} with {weight_count} weights")]
    PointerOutOfBounds {
        slot: u16,
        start_weight: u32,
        end_weight: u64,
        weight_count: u64,
    },
}

impl Filesystem {
    /// Initialise a fresh V2 filesystem on a cover that has no prior
    /// V2 state. Uses the default data chunk size
    /// ([`DEFAULT_CHUNK_SIZE_BYTES`] = 4 KB).
    pub fn init(cover: Vec<u8>, map: TensorMap) -> Result<Self, FsError> {
        Self::init_with_chunk_size(cover, map, DEFAULT_CHUNK_SIZE_BYTES)
    }

    /// As [`Self::init`] but with a configurable data chunk size —
    /// useful for tests on smaller covers.
    pub fn init_with_chunk_size(
        mut cover: Vec<u8>,
        map: TensorMap,
        data_chunk_size_bytes: u32,
    ) -> Result<Self, FsError> {
        validate_chunk_size(data_chunk_size_bytes)?;
        let ceiling = CeilingSummary::build(&cover, &map);
        let mut allocator = Allocator::new_for_map(&map, ceiling)?;

        // Compute anchor placement once and keep it; every subsequent
        // read_anchor / commit_anchor reuses this cached copy.
        let anchor_placement = anchor::find_anchor_placement(&cover, &map);
        allocator.reserve_weights(unique_weights(&anchor_placement))?;

        let root_inode = Inode::EMPTY;
        let root_inode_ptr =
            alloc_and_write(&mut cover, &map, &mut allocator, &root_inode.encode())?;

        let super_root = SuperRoot {
            root_dir_inode: root_inode_ptr,
            generation: 1,
            ..SuperRoot::EMPTY
        };
        let super_root_ptr =
            alloc_and_write(&mut cover, &map, &mut allocator, &super_root.encode())?;

        anchor::init_anchor(&mut cover, &map, super_root_ptr)?;

        Ok(Self {
            cover,
            map,
            alloc: allocator,
            anchor_placement,
            generation: 1,
            super_root,
            super_root_ptr,
            root_inode,
            root_inode_ptr,
            data_chunk_size_bytes,
        })
    }

    /// Mount an existing V2 filesystem from the cover.
    pub fn mount(cover: Vec<u8>, map: TensorMap) -> Result<Self, FsError> {
        Self::mount_with_chunk_size(cover, map, DEFAULT_CHUNK_SIZE_BYTES)
    }

    /// As [`Self::mount`] but with a configurable data chunk size —
    /// must match the size used at init-time for subsequent writes
    /// to stay within the direct-pointer budget (step 6 limitation).
    pub fn mount_with_chunk_size(
        cover: Vec<u8>,
        map: TensorMap,
        data_chunk_size_bytes: u32,
    ) -> Result<Self, FsError> {
        validate_chunk_size(data_chunk_size_bytes)?;
        // Compute the anchor placement once. The cover's ceiling
        // magnitudes are the same whether we're inside read_anchor or
        // new_for_map, so this single scan covers both.
        let anchor_placement = anchor::find_anchor_placement(&cover, &map);
        let anchor_outcome =
            anchor::read_anchor_with_placement(&cover, &map, &anchor_placement)?;

        let ceiling = CeilingSummary::build(&cover, &map);
        let mut allocator = Allocator::new_for_map(&map, ceiling)?;

        allocator.reserve_weights(unique_weights(&anchor_placement))?;

        let super_root_ptr = anchor_outcome.active.super_root;
        let mut super_root_bytes = [0u8; SUPER_ROOT_BYTES];
        read_chunk(&cover, &map, super_root_ptr, 0, &mut super_root_bytes)?;
        let super_root = SuperRoot::decode(&super_root_bytes)?;

        if super_root.generation != anchor_outcome.active.generation {
            return Err(FsError::GenerationMismatch {
                anchor: anchor_outcome.active.generation,
                super_root: super_root.generation,
            });
        }

        reserve_pointer(&mut allocator, &map, super_root_ptr)?;

        let root_inode_ptr = super_root.root_dir_inode;
        let mut inode_bytes = [0u8; INODE_BYTES];
        read_chunk(&cover, &map, root_inode_ptr, 0, &mut inode_bytes)?;
        let root_inode = Inode::decode(&inode_bytes)?;

        reserve_pointer(&mut allocator, &map, root_inode_ptr)?;
        // Walk the full inode tree — direct + indirect blocks + the
        // data chunks each indirect tier reaches — and reserve
        // every chunk so subsequent allocs avoid live state.
        visit_inode_chunks(&root_inode, &cover, &map, |ptr| {
            reserve_pointer(&mut allocator, &map, ptr)
        })?;

        Ok(Self {
            cover,
            map,
            alloc: allocator,
            anchor_placement,
            generation: anchor_outcome.active.generation,
            super_root,
            super_root_ptr,
            root_inode,
            root_inode_ptr,
            data_chunk_size_bytes,
        })
    }

    /// Replace the single-file contents with `data`. Allocates fresh
    /// chunks, builds a new inode + super-root (with indirect blocks
    /// as needed), and commits via an anchor-slot swap. Old chunks
    /// leak for this milestone.
    pub fn write(&mut self, data: &[u8]) -> Result<(), FsError> {
        let chunk_size = self.data_chunk_size_bytes as usize;
        let ppb = ptrs_per_indirect_block(chunk_size);
        let chunk_count = if data.is_empty() {
            0
        } else {
            data.len().div_ceil(chunk_size.max(1))
        };

        let ppb2 = ppb.saturating_mul(ppb);
        let ppb3 = ppb2.saturating_mul(ppb);
        let max_chunks = NUM_DIRECT
            .saturating_add(ppb)
            .saturating_add(ppb2)
            .saturating_add(ppb3);
        if chunk_count > max_chunks {
            return Err(FsError::FileTooLarge {
                bytes: data.len(),
                max_chunks,
                chunk_size: self.data_chunk_size_bytes,
            });
        }

        // Allocate + write every data chunk in order.
        let mut data_chunk_ptrs = Vec::with_capacity(chunk_count);
        let mut offset = 0;
        for _ in 0..chunk_count {
            let this_bytes = (data.len() - offset).min(chunk_size);
            let bit_count = (this_bytes * 8) as u32;
            let ptr = self
                .alloc
                .alloc(&self.map, bit_count)
                .ok_or(FsError::OutOfSpace {
                    requested_bits: bit_count,
                })?;
            write_chunk(
                &mut self.cover,
                &self.map,
                ptr,
                0,
                &data[offset..offset + this_bytes],
            )?;
            data_chunk_ptrs.push(ptr);
            offset += this_bytes;
        }

        // Thread data chunk pointers into direct + single/double/triple
        // indirect blocks per the standard ext2-style layout.
        let mut direct = [Pointer::NULL; NUM_DIRECT];
        let direct_used = chunk_count.min(NUM_DIRECT);
        direct[..direct_used].copy_from_slice(&data_chunk_ptrs[..direct_used]);

        let mut cursor = direct_used;
        let single_indirect = if cursor < chunk_count {
            let end = (cursor + ppb).min(chunk_count);
            let ptr = write_pointer_list(
                &mut self.cover,
                &self.map,
                &mut self.alloc,
                &data_chunk_ptrs[cursor..end],
            )?;
            cursor = end;
            ptr
        } else {
            Pointer::NULL
        };

        let double_indirect = if cursor < chunk_count {
            let end = (cursor + ppb * ppb).min(chunk_count);
            let ptr = build_double_indirect(
                &mut self.cover,
                &self.map,
                &mut self.alloc,
                &data_chunk_ptrs[cursor..end],
                ppb,
            )?;
            cursor = end;
            ptr
        } else {
            Pointer::NULL
        };

        let triple_indirect = if cursor < chunk_count {
            let end = chunk_count;
            let ptr = build_triple_indirect(
                &mut self.cover,
                &self.map,
                &mut self.alloc,
                &data_chunk_ptrs[cursor..end],
                ppb,
            )?;
            cursor = end;
            ptr
        } else {
            Pointer::NULL
        };

        debug_assert_eq!(cursor, chunk_count);

        let new_inode = Inode {
            length: data.len() as u64,
            direct,
            single_indirect,
            double_indirect,
            triple_indirect,
        };
        let new_inode_ptr =
            alloc_and_write(&mut self.cover, &self.map, &mut self.alloc, &new_inode.encode())?;

        let new_generation = self.generation + 1;
        let new_super_root = SuperRoot {
            root_dir_inode: new_inode_ptr,
            generation: new_generation,
            ..SuperRoot::EMPTY
        };
        let new_super_root_ptr = alloc_and_write(
            &mut self.cover,
            &self.map,
            &mut self.alloc,
            &new_super_root.encode(),
        )?;

        let committed_gen = anchor::commit_anchor_with_placement(
            &mut self.cover,
            &self.map,
            &self.anchor_placement,
            new_super_root_ptr,
            self.generation,
        )?;
        debug_assert_eq!(committed_gen, new_generation);

        self.generation = new_generation;
        self.super_root = new_super_root;
        self.super_root_ptr = new_super_root_ptr;
        self.root_inode = new_inode;
        self.root_inode_ptr = new_inode_ptr;

        Ok(())
    }

    /// Read the single file's full byte contents. Walks the inode's
    /// direct + single / double / triple indirect chains to collect
    /// every data chunk in order, concatenates their bytes, and
    /// truncates to `inode.length`.
    pub fn read(&self) -> Result<Vec<u8>, FsError> {
        let length = self.root_inode.length as usize;
        let data_ptrs = collect_data_pointers(&self.root_inode, &self.cover, &self.map)?;

        let mut out = Vec::with_capacity(length);
        for ptr in data_ptrs {
            if out.len() >= length {
                break;
            }
            let chunk_bytes = byte_capacity(ptr) as usize;
            let this_read = chunk_bytes.min(length - out.len());
            let mut buf = vec![0u8; this_read];
            read_chunk(&self.cover, &self.map, ptr, 0, &mut buf)?;
            out.extend_from_slice(&buf);
        }
        out.truncate(length);
        Ok(out)
    }

    /// Consume the filesystem and return the cover bytes for
    /// persistence.
    pub fn unmount(self) -> Vec<u8> {
        self.cover
    }

    /// Current committed anchor generation. Bumps by 1 per successful
    /// [`Self::write`].
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Length of the single file in bytes.
    pub fn file_length(&self) -> u64 {
        self.root_inode.length
    }
}

/// Collect the unique `(slot, weight_index)` pairs that appear in a
/// placement's bit positions. Deduplicates across the multiple bits
/// of a single weight (e.g. F16 contributes 4 bits per weight).
fn unique_weights(placement: &MetadataPlacement) -> Vec<(u16, u32)> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for p in &placement.positions {
        let key = (p.slot_index as u16, p.weight_index as u32);
        if seen.insert(key) {
            out.push(key);
        }
    }
    out
}

fn validate_chunk_size(data_chunk_size_bytes: u32) -> Result<(), FsError> {
    if data_chunk_size_bytes == 0 {
        return Err(FsError::InvalidChunkSize {
            chunk_size: data_chunk_size_bytes,
        });
    }
    Ok(())
}

/// Reserve every weight covered by `ptr` in the allocator.
fn reserve_pointer(
    allocator: &mut Allocator,
    map: &TensorMap,
    ptr: Pointer,
) -> Result<(), FsError> {
    if ptr.is_null() {
        return Ok(());
    }
    let slot = map
        .slots
        .get(ptr.slot as usize)
        .ok_or(FsError::PointerSlotOutOfRange {
            slot: ptr.slot,
            slot_count: map.slots.len(),
        })?;
    let bpw = slot.stealable_bits_per_weight as u32;
    if bpw == 0 {
        return Err(FsError::PointerTargetsNonStealableSlot { slot: ptr.slot });
    }
    let weights = ptr.length_in_bits.div_ceil(bpw);
    let end_weight = u64::from(ptr.start_weight) + u64::from(weights);
    if end_weight > slot.weight_count {
        return Err(FsError::PointerOutOfBounds {
            slot: ptr.slot,
            start_weight: ptr.start_weight,
            end_weight,
            weight_count: slot.weight_count,
        });
    }
    let iter = (0..weights).map(|i| (ptr.slot, ptr.start_weight + i));
    allocator.reserve_weights(iter)?;
    Ok(())
}

/// Shared helper: allocate a chunk of exactly `data.len() * 8` bits
/// and write `data` into it.
fn alloc_and_write(
    cover: &mut [u8],
    map: &TensorMap,
    allocator: &mut Allocator,
    data: &[u8],
) -> Result<Pointer, FsError> {
    let bit_count = (data.len() * 8) as u32;
    let ptr = allocator
        .alloc(map, bit_count)
        .ok_or(FsError::OutOfSpace {
            requested_bits: bit_count,
        })?;
    write_chunk(cover, map, ptr, 0, data)?;
    Ok(ptr)
}

// ------------------------------------------------------------------
// Indirect-block helpers
// ------------------------------------------------------------------

/// Max pointers a full indirect block can hold at the given data
/// chunk size — the threshold for overflowing to the next indirect
/// tier. A "full" indirect block is sized exactly like a data chunk.
fn ptrs_per_indirect_block(data_chunk_size_bytes: usize) -> usize {
    data_chunk_size_bytes / Pointer::SIZE
}

/// Allocate + write a pointer list as a chunk. Returns the chunk's
/// pointer. The chunk is sized to hold exactly `pointers.len()`
/// pointers — partial indirect blocks are not padded out to the
/// full `ptrs_per_indirect_block` size.
fn write_pointer_list(
    cover: &mut [u8],
    map: &TensorMap,
    allocator: &mut Allocator,
    pointers: &[Pointer],
) -> Result<Pointer, FsError> {
    let mut bytes = Vec::with_capacity(pointers.len() * Pointer::SIZE);
    for p in pointers {
        bytes.extend_from_slice(&p.encode());
    }
    alloc_and_write(cover, map, allocator, &bytes)
}

/// Read the pointer list out of an indirect block. The block's
/// pointer count is derived from its byte capacity — `byte_capacity /
/// 16` — so variable-size partial blocks decode correctly.
fn read_pointer_list(
    cover: &[u8],
    map: &TensorMap,
    block_ptr: Pointer,
) -> Result<Vec<Pointer>, FsError> {
    if block_ptr.is_null() {
        return Ok(Vec::new());
    }
    let byte_count = byte_capacity(block_ptr) as usize;
    debug_assert!(
        byte_count.is_multiple_of(Pointer::SIZE),
        "pointer block capacity {byte_count} not a multiple of Pointer::SIZE",
    );
    let pointer_count = byte_count / Pointer::SIZE;
    let mut buf = vec![0u8; byte_count];
    read_chunk(cover, map, block_ptr, 0, &mut buf)?;
    let mut out = Vec::with_capacity(pointer_count);
    for i in 0..pointer_count {
        let start = i * Pointer::SIZE;
        out.push(Pointer::decode(&buf[start..start + Pointer::SIZE])?);
    }
    Ok(out)
}

/// Build a double-indirect block: group `data_ptrs` into chunks of
/// `ppb` data pointers each, write a single-indirect block per
/// group, then write the list of single-indirect-block pointers as
/// the double-indirect block. Returns the double-indirect block's
/// pointer.
fn build_double_indirect(
    cover: &mut [u8],
    map: &TensorMap,
    allocator: &mut Allocator,
    data_ptrs: &[Pointer],
    ppb: usize,
) -> Result<Pointer, FsError> {
    let mut single_indirect_ptrs = Vec::new();
    for group in data_ptrs.chunks(ppb) {
        let sip = write_pointer_list(cover, map, allocator, group)?;
        single_indirect_ptrs.push(sip);
    }
    write_pointer_list(cover, map, allocator, &single_indirect_ptrs)
}

/// Build a triple-indirect block: group `data_ptrs` into chunks of
/// `ppb² data pointers each, each of which becomes a
/// double-indirect block via [`build_double_indirect`]. Returns the
/// triple-indirect block's pointer.
fn build_triple_indirect(
    cover: &mut [u8],
    map: &TensorMap,
    allocator: &mut Allocator,
    data_ptrs: &[Pointer],
    ppb: usize,
) -> Result<Pointer, FsError> {
    let group_size = ppb * ppb;
    let mut double_indirect_ptrs = Vec::new();
    for group in data_ptrs.chunks(group_size) {
        let dip = build_double_indirect(cover, map, allocator, group, ppb)?;
        double_indirect_ptrs.push(dip);
    }
    write_pointer_list(cover, map, allocator, &double_indirect_ptrs)
}

/// Walk the inode's direct + single / double / triple indirect tree
/// and collect every non-null **data** chunk pointer in order. The
/// indirect block pointers themselves are traversed but not
/// included — callers asking "where's my file's bytes?" want leaves
/// only.
fn collect_data_pointers(
    inode: &Inode,
    cover: &[u8],
    map: &TensorMap,
) -> Result<Vec<Pointer>, FsError> {
    let mut out = Vec::new();

    for p in &inode.direct {
        if p.is_null() {
            break;
        }
        out.push(*p);
    }

    if !inode.single_indirect.is_null() {
        for p in read_pointer_list(cover, map, inode.single_indirect)? {
            if p.is_null() {
                break;
            }
            out.push(p);
        }
    }

    if !inode.double_indirect.is_null() {
        for sip in read_pointer_list(cover, map, inode.double_indirect)? {
            if sip.is_null() {
                break;
            }
            for p in read_pointer_list(cover, map, sip)? {
                if p.is_null() {
                    break;
                }
                out.push(p);
            }
        }
    }

    if !inode.triple_indirect.is_null() {
        for dip in read_pointer_list(cover, map, inode.triple_indirect)? {
            if dip.is_null() {
                break;
            }
            for sip in read_pointer_list(cover, map, dip)? {
                if sip.is_null() {
                    break;
                }
                for p in read_pointer_list(cover, map, sip)? {
                    if p.is_null() {
                        break;
                    }
                    out.push(p);
                }
            }
        }
    }

    Ok(out)
}

/// Visit every non-null pointer the inode transitively references —
/// data chunks AND the indirect block chunks themselves. Used by
/// mount to reserve all live chunks in the allocator.
fn visit_inode_chunks(
    inode: &Inode,
    cover: &[u8],
    map: &TensorMap,
    mut visit: impl FnMut(Pointer) -> Result<(), FsError>,
) -> Result<(), FsError> {
    for p in &inode.direct {
        if !p.is_null() {
            visit(*p)?;
        }
    }

    if !inode.single_indirect.is_null() {
        visit(inode.single_indirect)?;
        for p in read_pointer_list(cover, map, inode.single_indirect)? {
            if !p.is_null() {
                visit(p)?;
            }
        }
    }

    if !inode.double_indirect.is_null() {
        visit(inode.double_indirect)?;
        for sip in read_pointer_list(cover, map, inode.double_indirect)? {
            if sip.is_null() {
                continue;
            }
            visit(sip)?;
            for p in read_pointer_list(cover, map, sip)? {
                if !p.is_null() {
                    visit(p)?;
                }
            }
        }
    }

    if !inode.triple_indirect.is_null() {
        visit(inode.triple_indirect)?;
        for dip in read_pointer_list(cover, map, inode.triple_indirect)? {
            if dip.is_null() {
                continue;
            }
            visit(dip)?;
            for sip in read_pointer_list(cover, map, dip)? {
                if sip.is_null() {
                    continue;
                }
                visit(sip)?;
                for p in read_pointer_list(cover, map, sip)? {
                    if !p.is_null() {
                        visit(p)?;
                    }
                }
            }
        }
    }

    Ok(())
}
