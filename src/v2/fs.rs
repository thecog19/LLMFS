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
use crate::v2::pointer::Pointer;
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

    #[error("out of space: could not allocate {requested_bits} bits")]
    OutOfSpace { requested_bits: u32 },

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
        for ptr in &root_inode.direct {
            if !ptr.is_null() {
                reserve_pointer(&mut allocator, &map, *ptr)?;
            }
        }

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
    /// chunks, builds a new inode + super-root, and commits via an
    /// anchor-slot swap. Old chunks leak for this milestone.
    pub fn write(&mut self, data: &[u8]) -> Result<(), FsError> {
        let chunk_size = self.data_chunk_size_bytes as usize;
        let chunk_count = if data.is_empty() {
            0
        } else {
            data.len().div_ceil(chunk_size.max(1))
        };
        if chunk_count > NUM_DIRECT {
            return Err(FsError::FileTooLarge {
                bytes: data.len(),
                max_chunks: NUM_DIRECT,
                chunk_size: self.data_chunk_size_bytes,
            });
        }

        let mut chunk_ptrs = [Pointer::NULL; NUM_DIRECT];
        let mut offset = 0;
        for chunk_ptr in chunk_ptrs.iter_mut().take(chunk_count) {
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
            *chunk_ptr = ptr;
            offset += this_bytes;
        }

        let new_inode = Inode {
            length: data.len() as u64,
            direct: chunk_ptrs,
            single_indirect: Pointer::NULL,
            double_indirect: Pointer::NULL,
            triple_indirect: Pointer::NULL,
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

    /// Read the single file's full byte contents.
    pub fn read(&self) -> Result<Vec<u8>, FsError> {
        let length = self.root_inode.length as usize;
        let mut out = Vec::with_capacity(length);
        for ptr in &self.root_inode.direct {
            if ptr.is_null() || out.len() >= length {
                break;
            }
            let chunk_bytes = byte_capacity(*ptr) as usize;
            let this_read = chunk_bytes.min(length - out.len());
            let mut buf = vec![0u8; this_read];
            read_chunk(&self.cover, &self.map, *ptr, 0, &mut buf)?;
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

/// Reserve every weight covered by `ptr` in the allocator.
fn reserve_pointer(
    allocator: &mut Allocator,
    map: &TensorMap,
    ptr: Pointer,
) -> Result<(), FsError> {
    if ptr.is_null() {
        return Ok(());
    }
    let slot = &map.slots[ptr.slot as usize];
    let bpw = slot.stealable_bits_per_weight as u32;
    let weights = ptr.length_in_bits.div_ceil(bpw);
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
