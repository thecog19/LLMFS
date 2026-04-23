//! V2 [`Filesystem`] — hierarchical path-addressed files and
//! directories over a GGUF cover.
//!
//! The super-root's `root_dir_inode` points at the inode whose data
//! chunks hold the serialized root [`Directory`]. Every directory —
//! root or nested — uses the same shape: an [`Inode`] with
//! direct / single / double / triple indirect pointers to variable-
//! length CDC-chunked content. Files are identical in shape; only
//! their content is user bytes instead of entry tables.
//!
//! # Path-based API
//!
//! All mutations target an absolute path:
//!
//! - [`Filesystem::mkdir`] — create an empty directory
//! - [`Filesystem::rmdir`] — remove an empty directory
//! - [`Filesystem::create_file`] — create or overwrite a file
//! - [`Filesystem::unlink`] — remove a file
//! - [`Filesystem::read_file`] — load a file's bytes
//! - [`Filesystem::readdir`] — list a directory's entries
//! - [`Filesystem::exists`] — check path presence
//!
//! Each mutation is a single CoW commit: the leaf's new content is
//! allocated, its parent directory is rewritten with the updated
//! entry, and the directory chain is rewritten back up to the root.
//! The super-root + anchor slot swap at the end makes the whole
//! rewrite atomic.
//!
//! # Allocator priority
//!
//! File content chunks consult the dedup index first (identical
//! bytes → same pointer; no re-allocation), fall through to
//! `alloc_preferring_dirty` (already-perturbed free runs), and
//! finally to the pristine ceiling-magnitude-ranked free list.
//! Directory content, super-roots, inodes, and the dirty bitmap do
//! not dedup — they're overwritten on every commit and the hash
//! lookup would be wasted work.
//!
//! # Reclamation
//!
//! After each commit the old tree's chunks are returned to the
//! free list (marked dirty) by diffing the old vs. new reachable
//! sets. [`collect_tree_chunks`] walks the directory tree
//! recursively and unions in the inode trees of every file and
//! directory it reaches.

use thiserror::Error;

use crate::stego::calibration::placement::MetadataPlacement;
use crate::stego::tensor_map::TensorMap;
use crate::v2::alloc::{AllocError, Allocator};
use crate::v2::anchor::{self, AnchorError};
use crate::v2::cdc::{FastCdcError, FastCdcParams, FastCdcStream, chunk_ranges};
use crate::v2::ceiling::CeilingSummary;
use crate::v2::chunk::{ChunkError, byte_capacity, read_chunk, write_chunk};
use crate::v2::cover::CoverStorage;
use crate::v2::dedup::{DedupIndex, hash_chunk};
use crate::v2::directory::{DirEntry, Directory, DirectoryError, EntryKind, MAX_NAME_LEN};
use crate::v2::dirty::{DirtyBitmap, DirtyBitmapError};
use crate::v2::inode::{INODE_BYTES, Inode, InodeError, NUM_DIRECT};
use crate::v2::pointer::{Pointer, PointerError};
use crate::v2::salience::{SalienceError, SalienceTable};
use crate::v2::super_root::{SUPER_ROOT_BYTES, SuperRoot, SuperRootError};

/// Top-level V2 filesystem. Owns the cover storage for the lifetime
/// of the mount; [`Self::unmount`] flushes any pending writes (for
/// file-backed `MmapMut` covers) and returns the storage for the
/// caller to drop or re-mount.
#[derive(Debug)]
pub struct Filesystem {
    cover: Box<dyn CoverStorage>,
    map: TensorMap,
    alloc: Allocator,
    anchor_placement: MetadataPlacement,
    generation: u64,
    super_root: SuperRoot,
    super_root_ptr: Pointer,
    /// The root directory's inode (describes where the serialized
    /// [`Directory`] lives). Decoded in-memory for cheap lookups.
    root_inode: Inode,
    root_inode_ptr: Pointer,
    /// Deserialized root directory. Kept in memory for O(log n)
    /// path walks without re-reading inode + content chunks each
    /// time.
    root_directory: Directory,
    cdc_params: FastCdcParams,
    /// Content-hash index of every live **file** data chunk. Rebuilt
    /// from scratch at mount and after every commit.
    dedup_index: DedupIndex,
    /// Dirty-weight bitmap. Marked as chunks get written; allocator
    /// uses it to prefer previously-perturbed free runs.
    dirty_bitmap: DirtyBitmap,
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

    #[error("salience codec: {0}")]
    Salience(#[from] SalienceError),

    #[error("pointer codec: {0}")]
    Pointer(#[from] PointerError),

    #[error("invalid CDC params: {0}")]
    Cdc(#[from] FastCdcError),

    #[error("dirty bitmap: {0}")]
    Dirty(#[from] DirtyBitmapError),

    #[error("directory codec: {0}")]
    Directory(#[from] DirectoryError),

    #[error("out of space: could not allocate {requested_bits} bits")]
    OutOfSpace { requested_bits: u32 },

    #[error(
        "file of {bytes} bytes produced {chunk_count} chunks, exceeding the inode's {max_chunks}-chunk capacity (direct + single/double/triple indirect)"
    )]
    FileTooLarge {
        bytes: usize,
        chunk_count: usize,
        max_chunks: usize,
    },

    #[error("super-root generation {super_root} disagrees with anchor generation {anchor}")]
    GenerationMismatch { anchor: u64, super_root: u64 },

    #[error("pointer references slot {slot} but map has only {slot_count} slots")]
    PointerSlotOutOfRange { slot: u16, slot_count: usize },

    #[error("pointer targets slot {slot}, which has no stealable bits")]
    PointerTargetsNonStealableSlot { slot: u16 },

    #[error(
        "pointer range [{start_weight}, {end_weight}) lies outside slot {slot} with {weight_count} weights"
    )]
    PointerOutOfBounds {
        slot: u16,
        start_weight: u32,
        end_weight: u64,
        weight_count: u64,
    },

    #[error("invalid path '{0}': must be absolute and contain only valid components")]
    InvalidPath(String),

    #[error("path '{0}' does not exist")]
    PathNotFound(String),

    #[error("'{0}' is not a directory")]
    NotADirectory(String),

    #[error("'{0}' is a directory")]
    IsADirectory(String),

    #[error("'{0}' already exists")]
    AlreadyExists(String),

    #[error("directory '{0}' is not empty")]
    DirectoryNotEmpty(String),

    #[error("path cannot be '/': root has no leaf name to act on")]
    PathCannotBeRoot,
}

impl Filesystem {
    // --------------------------------------------------------------
    // Init / Mount / Unmount
    // --------------------------------------------------------------

    /// Initialise a fresh V2 filesystem on a cover that has no prior
    /// V2 state. Uses the default CDC params (1 KB / 4 KB / 16 KB).
    /// `cover` may be a `Vec<u8>` (in-memory; great for tests) or a
    /// [`memmap2::MmapMut`] (file-backed; bounded RAM).
    pub fn init<C: CoverStorage + 'static>(cover: C, map: TensorMap) -> Result<Self, FsError> {
        Self::init_with_cdc_params(cover, map, FastCdcParams::default())
    }

    /// As [`Self::init`] but with configurable CDC parameters.
    pub fn init_with_cdc_params<C: CoverStorage + 'static>(
        cover: C,
        map: TensorMap,
        cdc_params: FastCdcParams,
    ) -> Result<Self, FsError> {
        let mut cover: Box<dyn CoverStorage> = Box::new(cover);
        cdc_params.validate()?;
        let ceiling = CeilingSummary::build(cover.bytes(), &map);
        let mut allocator = Allocator::new_for_map(&map, ceiling)?;

        let anchor_placement = anchor::find_anchor_placement(cover.bytes(), &map);
        allocator.reserve_weights(unique_weights(&anchor_placement))?;

        let mut dirty_bitmap = DirtyBitmap::new(&map);
        let mut dedup_index = DedupIndex::new();

        // Fresh root directory → empty serialized entry list.
        let root_directory = Directory::new();
        let root_inode_ptr = write_directory_content(
            cover.bytes_mut(),
            &map,
            &mut allocator,
            &mut dirty_bitmap,
            &mut dedup_index,
            &cdc_params,
            &root_directory,
        )?;
        let root_inode = read_inode(cover.bytes(), &map, root_inode_ptr)?;

        // Persist the dirty bitmap via the streaming path. At init
        // it's all zeros; every CDC chunk hashes identically and
        // the shared `dedup_index` collapses them to a single
        // physical allocation (plus a handful of indirect blocks),
        // keeping init-time perturbation bounded regardless of
        // cover size. Streaming avoids materialising the dense
        // `total_bytes()` Vec — for a 280 GB cover that's a 17.5 GB
        // allocation we can't afford.
        let bitmap_inode_ptr = persist_bitmap_streaming(
            cover.bytes_mut(),
            &map,
            &mut allocator,
            &mut dirty_bitmap,
            &mut dedup_index,
            &cdc_params,
        )?;

        let super_root = SuperRoot {
            root_dir_inode: root_inode_ptr,
            dirty_bitmap_inode: bitmap_inode_ptr,
            generation: 1,
            ..SuperRoot::EMPTY
        };
        let super_root_ptr = alloc_and_write(
            cover.bytes_mut(),
            &map,
            &mut allocator,
            &super_root.encode(),
        )?;
        mark_pointer_dirty(&mut dirty_bitmap, &map, super_root_ptr);

        anchor::init_anchor(cover.bytes_mut(), &map, super_root_ptr)?;

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
            root_directory,
            cdc_params,
            dedup_index,
            dirty_bitmap,
        })
    }

    /// Mount an existing V2 filesystem from the cover.
    pub fn mount<C: CoverStorage + 'static>(cover: C, map: TensorMap) -> Result<Self, FsError> {
        Self::mount_with_cdc_params(cover, map, FastCdcParams::default())
    }

    /// As [`Self::mount`] but with configurable CDC parameters. Must
    /// match the params used at init-time for subsequent writes to
    /// produce compatible chunking.
    pub fn mount_with_cdc_params<C: CoverStorage + 'static>(
        cover: C,
        map: TensorMap,
        cdc_params: FastCdcParams,
    ) -> Result<Self, FsError> {
        let cover: Box<dyn CoverStorage> = Box::new(cover);
        cdc_params.validate()?;
        let anchor_placement = anchor::find_anchor_placement(cover.bytes(), &map);
        let anchor_outcome =
            anchor::read_anchor_with_placement(cover.bytes(), &map, &anchor_placement)?;

        let super_root_ptr = anchor_outcome.active.super_root;
        // The super-root chunk may be v1 (100B) or v2 (116B). Size
        // the read to the pointer's stored length so either lands.
        let super_root_chunk_len =
            ((super_root_ptr.length_in_bits as usize) / 8).min(SUPER_ROOT_BYTES);
        let mut super_root_bytes = [0u8; SUPER_ROOT_BYTES];
        read_chunk(
            cover.bytes(),
            &map,
            super_root_ptr,
            0,
            &mut super_root_bytes[..super_root_chunk_len],
        )?;
        let super_root = SuperRoot::decode(&super_root_bytes[..super_root_chunk_len])?;

        if super_root.generation != anchor_outcome.active.generation {
            return Err(FsError::GenerationMismatch {
                anchor: anchor_outcome.active.generation,
                super_root: super_root.generation,
            });
        }

        // Load the calibration salience (if any) *before* building
        // the allocator, so every initial free run's `max_salience`
        // reflects the real per-weight data. Load failures surface
        // through `FsError::Salience`.
        let salience_table = if super_root.salience_inode.is_null() {
            SalienceTable::empty()
        } else {
            let bytes = load_byte_stream(cover.bytes(), &map, super_root.salience_inode)?;
            SalienceTable::decode(&bytes).map_err(FsError::Salience)?
        };

        let ceiling = CeilingSummary::build(cover.bytes(), &map);
        let mut allocator =
            Allocator::new_for_map_with_salience(&map, ceiling, salience_table)?;
        allocator.reserve_weights(unique_weights(&anchor_placement))?;

        reserve_pointer(&mut allocator, &map, super_root_ptr)?;

        let root_inode_ptr = super_root.root_dir_inode;

        // Walk the directory tree, validating + reserving every
        // pointer as we encounter it. Corrupted pointers (e.g. an
        // out-of-range slot) surface through reserve_pointer's
        // checks rather than opaque chunk-read failures downstream.
        let mut reserved: std::collections::HashSet<(u16, u32)> = std::collections::HashSet::new();
        let mut tree_chunks: std::collections::HashSet<(u16, u32, u32)> =
            std::collections::HashSet::new();
        walk_tree_pointers(cover.bytes(), &map, root_inode_ptr, &mut |ptr| {
            tree_chunks.insert((ptr.slot, ptr.start_weight, ptr.length_in_bits));
            if reserved.insert((ptr.slot, ptr.start_weight)) {
                reserve_pointer(&mut allocator, &map, ptr)?;
            }
            Ok(())
        })?;

        let root_inode = read_inode(cover.bytes(), &map, root_inode_ptr)?;
        let root_directory = read_directory(cover.bytes(), &map, root_inode_ptr)?;

        // Load the persisted dirty bitmap into a sparse-page
        // structure via the streaming helper — for a 280 GB cover
        // we can't materialise the 17.5 GB dense byte form.
        let mut dirty_bitmap = DirtyBitmap::new(&map);
        if !super_root.dirty_bitmap_inode.is_null() {
            load_bitmap_streaming(
                cover.bytes(),
                &map,
                super_root.dirty_bitmap_inode,
                &mut dirty_bitmap,
            )?;
        }
        if !super_root.dirty_bitmap_inode.is_null() {
            let bitmap_chunks =
                inode_tree_chunks(cover.bytes(), &map, super_root.dirty_bitmap_inode)?;
            for (slot, start, len_bits) in bitmap_chunks {
                if reserved.insert((slot, start)) {
                    let p = Pointer {
                        slot,
                        start_weight: start,
                        length_in_bits: len_bits,
                        flags: 0,
                        reserved: 0,
                    };
                    reserve_pointer(&mut allocator, &map, p)?;
                }
            }
        }

        // Salience inode tree (B1): same reservation pass as the
        // bitmap. Without this, subsequent commits would reclaim
        // live salience chunks on the first commit after mount.
        if !super_root.salience_inode.is_null() {
            let salience_chunks =
                inode_tree_chunks(cover.bytes(), &map, super_root.salience_inode)?;
            for (slot, start, len_bits) in salience_chunks {
                if reserved.insert((slot, start)) {
                    let p = Pointer {
                        slot,
                        start_weight: start,
                        length_in_bits: len_bits,
                        flags: 0,
                        reserved: 0,
                    };
                    reserve_pointer(&mut allocator, &map, p)?;
                }
            }
        }

        // In-memory bitmap must reflect every live V2 metadata
        // chunk; the persisted snapshot may lag by one commit.
        mark_pointer_dirty(&mut dirty_bitmap, &map, super_root_ptr);
        for (slot, start, len_bits) in &tree_chunks {
            let p = Pointer {
                slot: *slot,
                start_weight: *start,
                length_in_bits: *len_bits,
                flags: 0,
                reserved: 0,
            };
            mark_pointer_dirty(&mut dirty_bitmap, &map, p);
        }

        // Rebuild the dedup index across every content chunk in
        // the tree plus the bitmap's inode tree plus the salience
        // inode tree — writes consult the same index for files,
        // directories, the bitmap, and (post-calibration) salience.
        let dedup_index = build_dedup_index(
            cover.bytes(),
            &map,
            root_inode_ptr,
            super_root.dirty_bitmap_inode,
            super_root.salience_inode,
        )?;

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
            root_directory,
            cdc_params,
            dedup_index,
            dirty_bitmap,
        })
    }

    /// Consume the filesystem; flush any pending writes (msync for
    /// file-backed storage) and return the cover for the caller to
    /// drop or remount. Tests typically discard the returned value;
    /// the CLI's mmap-backed cover relies on the flush before drop
    /// to make changes durable.
    pub fn unmount(mut self) -> std::io::Result<Box<dyn CoverStorage>> {
        self.cover.flush()?;
        Ok(self.cover)
    }

    // --------------------------------------------------------------
    // Mutation: mkdir / rmdir / create_file / unlink
    // --------------------------------------------------------------

    /// Create an empty directory at `path`. Errors if anything
    /// already exists there, or if any intermediate component is
    /// missing or isn't a directory.
    pub fn mkdir(&mut self, path: &str) -> Result<(), FsError> {
        let components = parse_path(path)?;
        if components.is_empty() {
            return Err(FsError::PathCannotBeRoot);
        }
        let (parent_dir, leaf_name) = self.read_parent_directory(&components)?;
        if parent_dir.find(leaf_name).is_some() {
            return Err(FsError::AlreadyExists(leaf_name.to_owned()));
        }

        // Semantic failures must be a no-op, so prove the parent path
        // and leaf availability before allocating the child directory.
        let empty = Directory::new();
        let child_inode = write_directory_content(
            self.cover.bytes_mut(),
            &self.map,
            &mut self.alloc,
            &mut self.dirty_bitmap,
            &mut self.dedup_index,
            &self.cdc_params,
            &empty,
        )?;

        let new_root_ptr = self.mutate_parent_directory(&components, |parent, name| {
            if parent.find(name).is_some() {
                return Err(FsError::AlreadyExists(name.to_owned()));
            }
            parent.insert(DirEntry {
                kind: EntryKind::Directory,
                name: name.to_owned(),
                inode: child_inode,
            })?;
            Ok(())
        })?;

        self.commit_with_new_root(new_root_ptr)
    }

    /// Remove an empty directory at `path`. Errors if the directory
    /// doesn't exist, isn't empty, is actually a file, or if any
    /// intermediate component is missing.
    pub fn rmdir(&mut self, path: &str) -> Result<(), FsError> {
        let components = parse_path(path)?;
        if components.is_empty() {
            return Err(FsError::PathCannotBeRoot);
        }

        // Check kind first (file → NotADirectory) before trying to
        // parse the target's content as a serialized directory.
        let (target_inode, kind) = self.lookup_leaf(&components)?;
        if kind != EntryKind::Directory {
            return Err(FsError::NotADirectory(path.to_owned()));
        }
        let target_dir = read_directory(self.cover.bytes(), &self.map, target_inode)?;
        if !target_dir.is_empty() {
            return Err(FsError::DirectoryNotEmpty(path.to_owned()));
        }

        let new_root_ptr = self.mutate_parent_directory(&components, |parent, name| {
            let entry = parent
                .find(name)
                .ok_or_else(|| FsError::PathNotFound(name.to_owned()))?;
            if entry.kind != EntryKind::Directory {
                return Err(FsError::NotADirectory(name.to_owned()));
            }
            parent.remove(name);
            Ok(())
        })?;

        self.commit_with_new_root(new_root_ptr)
    }

    /// Create or overwrite a file at `path` with `data`. Errors if
    /// any intermediate component is missing or isn't a directory,
    /// or if `path` itself is an existing directory.
    pub fn create_file(&mut self, path: &str, data: &[u8]) -> Result<(), FsError> {
        let components = parse_path(path)?;
        if components.is_empty() {
            return Err(FsError::PathCannotBeRoot);
        }
        let (parent_dir, leaf_name) = self.read_parent_directory(&components)?;
        if let Some(existing) = parent_dir.find(leaf_name)
            && existing.kind == EntryKind::Directory
        {
            return Err(FsError::IsADirectory(leaf_name.to_owned()));
        }

        // Path errors must not perturb allocator state or cover bytes,
        // so write the file content only after the path preflight above.
        let file_inode_ptr = self.write_file_content(data)?;

        let new_root_ptr = self.mutate_parent_directory(&components, |parent, name| {
            if let Some(existing) = parent.find(name) {
                if existing.kind == EntryKind::Directory {
                    return Err(FsError::IsADirectory(name.to_owned()));
                }
                parent.replace(name, EntryKind::File, file_inode_ptr);
            } else {
                parent.insert(DirEntry {
                    kind: EntryKind::File,
                    name: name.to_owned(),
                    inode: file_inode_ptr,
                })?;
            }
            Ok(())
        })?;

        self.commit_with_new_root(new_root_ptr)
    }

    /// Remove the file at `path`. Errors if absent or if `path` is
    /// a directory.
    pub fn unlink(&mut self, path: &str) -> Result<(), FsError> {
        let components = parse_path(path)?;
        if components.is_empty() {
            return Err(FsError::PathCannotBeRoot);
        }

        let new_root_ptr = self.mutate_parent_directory(&components, |parent, name| {
            let entry = parent
                .find(name)
                .ok_or_else(|| FsError::PathNotFound(name.to_owned()))?;
            if entry.kind == EntryKind::Directory {
                return Err(FsError::IsADirectory(name.to_owned()));
            }
            parent.remove(name);
            Ok(())
        })?;

        self.commit_with_new_root(new_root_ptr)
    }

    // --------------------------------------------------------------
    // Reads
    // --------------------------------------------------------------

    /// Load the bytes of the file at `path`. Errors if the path is
    /// missing, is a directory, or any intermediate component is
    /// missing / non-directory.
    pub fn read_file(&self, path: &str) -> Result<Vec<u8>, FsError> {
        let components = parse_path(path)?;
        if components.is_empty() {
            return Err(FsError::IsADirectory(path.to_owned()));
        }
        let (target_inode, kind) = self.lookup_leaf(&components)?;
        if kind == EntryKind::Directory {
            return Err(FsError::IsADirectory(path.to_owned()));
        }
        load_byte_stream(self.cover.bytes(), &self.map, target_inode)
    }

    /// List the entries in the directory at `path`. Entries are
    /// returned in sorted order. Errors if the path is missing or
    /// isn't a directory.
    pub fn readdir(&self, path: &str) -> Result<Vec<DirEntry>, FsError> {
        let components = parse_path(path)?;
        if components.is_empty() {
            return Ok(self.root_directory.entries().to_vec());
        }
        let (target_inode, kind) = self.lookup_leaf(&components)?;
        if kind != EntryKind::Directory {
            return Err(FsError::NotADirectory(path.to_owned()));
        }
        let dir = read_directory(self.cover.bytes(), &self.map, target_inode)?;
        Ok(dir.entries().to_vec())
    }

    /// Whether `path` resolves to an existing file or directory.
    /// Invalid paths return `false` (the path cannot exist). Returns
    /// `true` for `/` (the root always exists).
    pub fn exists(&self, path: &str) -> bool {
        let Ok(components) = parse_path(path) else {
            return false;
        };
        if components.is_empty() {
            return true;
        }
        self.lookup_leaf(&components).is_ok()
    }

    /// Return the inode at `path` (file or directory). Primarily
    /// for tests that need to inspect the direct/indirect layout.
    pub fn inode_at(&self, path: &str) -> Result<Inode, FsError> {
        let components = parse_path(path)?;
        if components.is_empty() {
            return Ok(self.root_inode);
        }
        let (target_inode, _) = self.lookup_leaf(&components)?;
        read_inode(self.cover.bytes(), &self.map, target_inode)
    }

    // --------------------------------------------------------------
    // Diagnostic accessors
    // --------------------------------------------------------------

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn cdc_params(&self) -> &FastCdcParams {
        &self.cdc_params
    }

    /// The root *directory's* inode — not a per-file inode. Tests
    /// that want the data file's inode should use [`Self::inode_at`].
    pub fn root_inode(&self) -> &Inode {
        &self.root_inode
    }

    pub fn root_directory(&self) -> &Directory {
        &self.root_directory
    }

    pub fn dedup_index(&self) -> &DedupIndex {
        &self.dedup_index
    }

    pub fn dirty_bitmap(&self) -> &DirtyBitmap {
        &self.dirty_bitmap
    }

    pub fn allocator_free_weights(&self) -> u64 {
        self.alloc.total_free_weights()
    }

    // --------------------------------------------------------------
    // Internals — path walking + tree mutation
    // --------------------------------------------------------------

    /// Find the inode pointer of the leaf named by `components`.
    /// Walks from root, following `EntryKind::Directory` components.
    /// Returns `(leaf_inode_ptr, leaf_kind)`.
    fn lookup_leaf(&self, components: &[&str]) -> Result<(Pointer, EntryKind), FsError> {
        if components.is_empty() {
            return Ok((self.root_inode_ptr, EntryKind::Directory));
        }
        let (leaf, parents) = components.split_last().unwrap();
        let mut cur: Option<Directory> = None;
        for pn in parents {
            let dir = match cur.as_ref() {
                Some(d) => d,
                None => &self.root_directory,
            };
            let entry = dir
                .find(pn)
                .ok_or_else(|| FsError::PathNotFound(join_path(components)))?;
            if entry.kind != EntryKind::Directory {
                return Err(FsError::NotADirectory((*pn).to_owned()));
            }
            cur = Some(read_directory(self.cover.bytes(), &self.map, entry.inode)?);
        }
        let dir = cur.as_ref().unwrap_or(&self.root_directory);
        let entry = dir
            .find(leaf)
            .ok_or_else(|| FsError::PathNotFound(join_path(components)))?;
        Ok((entry.inode, entry.kind))
    }

    /// Load the directory that should contain `components`' leaf and
    /// return it alongside the leaf name. Used by mutators to prove
    /// path semantics before they allocate any new chunks.
    fn read_parent_directory<'a>(
        &self,
        components: &'a [&'a str],
    ) -> Result<(Directory, &'a str), FsError> {
        let (leaf, parent_names) = components.split_last().unwrap();
        let mut cur: Option<Directory> = None;
        for pn in parent_names {
            let dir = cur.as_ref().unwrap_or(&self.root_directory);
            let entry = dir
                .find(pn)
                .ok_or_else(|| FsError::PathNotFound(join_path(components)))?;
            if entry.kind != EntryKind::Directory {
                return Err(FsError::NotADirectory((*pn).to_owned()));
            }
            cur = Some(read_directory(self.cover.bytes(), &self.map, entry.inode)?);
        }
        let parent = cur.unwrap_or_else(|| self.root_directory.clone());
        Ok((parent, *leaf))
    }

    /// Walk to the parent directory of `components`, apply `mutate`
    /// to that parent (passing the leaf's name), then rewrite the
    /// directory chain from the leaf's parent back to the root.
    /// Returns the pointer to the new root directory's inode (ready
    /// for [`Self::commit_with_new_root`]).
    fn mutate_parent_directory(
        &mut self,
        components: &[&str],
        mutate: impl FnOnce(&mut Directory, &str) -> Result<(), FsError>,
    ) -> Result<Pointer, FsError> {
        let (leaf_name, parent_names) = components.split_last().unwrap();

        // Load the parent chain:
        //   dirs[0] = root
        //   dirs[i] = root[parent_names[0]]...[parent_names[i-1]]
        let mut dirs: Vec<Directory> = Vec::with_capacity(parent_names.len() + 1);
        dirs.push(self.root_directory.clone());
        for pn in parent_names {
            let parent = dirs.last().unwrap();
            let entry = parent
                .find(pn)
                .ok_or_else(|| FsError::PathNotFound(join_path(components)))?;
            if entry.kind != EntryKind::Directory {
                return Err(FsError::NotADirectory((*pn).to_owned()));
            }
            let child = read_directory(self.cover.bytes(), &self.map, entry.inode)?;
            dirs.push(child);
        }

        // Apply the mutation to the parent-of-leaf directory.
        let mut cur = dirs.pop().unwrap();
        mutate(&mut cur, leaf_name)?;

        // Write it and walk up, rewriting each ancestor.
        let mut new_child_ptr = write_directory_content(
            self.cover.bytes_mut(),
            &self.map,
            &mut self.alloc,
            &mut self.dirty_bitmap,
            &mut self.dedup_index,
            &self.cdc_params,
            &cur,
        )?;
        for i in (0..parent_names.len()).rev() {
            let name = parent_names[i];
            let mut parent = dirs.pop().unwrap();
            parent.replace(name, EntryKind::Directory, new_child_ptr);
            new_child_ptr = write_directory_content(
                self.cover.bytes_mut(),
                &self.map,
                &mut self.alloc,
                &mut self.dirty_bitmap,
                &mut self.dedup_index,
                &self.cdc_params,
                &parent,
            )?;
        }
        debug_assert!(dirs.is_empty());
        Ok(new_child_ptr)
    }

    /// Finalise a CoW mutation with a new root directory, carrying
    /// the existing salience inode forward.
    fn commit_with_new_root(&mut self, new_root_dir_inode: Pointer) -> Result<(), FsError> {
        let salience = self.super_root.salience_inode;
        self.commit_snapshot(new_root_dir_inode, salience)
    }

    /// Persist `table` as a new salience inode and commit a super-root
    /// that points at it. The directory tree is unchanged; only the
    /// salience slot moves. Returns the new salience inode pointer.
    ///
    /// The allocator doesn't reload from the new table until the next
    /// mount — B4 intentionally scopes re-indexing to the mount path
    /// to keep commits cheap. Tests that want to observe the new
    /// placement bias cycle through unmount + mount.
    pub fn commit_salience(&mut self, table: &SalienceTable) -> Result<Pointer, FsError> {
        let data = table.encode();
        let new_salience_inode = persist_as_byte_stream(
            self.cover.bytes_mut(),
            &self.map,
            &mut self.alloc,
            &mut self.dirty_bitmap,
            &mut self.dedup_index,
            &self.cdc_params,
            &data,
        )?;
        let root = self.root_inode_ptr;
        self.commit_snapshot(root, new_salience_inode)?;
        Ok(new_salience_inode)
    }

    /// Load the salience table from the salience inode. Returns
    /// `None` when the cover hasn't been calibrated yet.
    pub fn load_salience(&self) -> Result<Option<SalienceTable>, FsError> {
        if self.super_root.salience_inode.is_null() {
            return Ok(None);
        }
        let bytes = load_byte_stream(
            self.cover.bytes(),
            &self.map,
            self.super_root.salience_inode,
        )?;
        let table = SalienceTable::decode(&bytes).map_err(FsError::Salience)?;
        Ok(Some(table))
    }

    /// Shared commit pipeline: persist bitmap, write super-root,
    /// swap anchor slots, reclaim unreachable chunks, rebuild the
    /// dedup index. Callers choose which of the two new inode
    /// pointers to change.
    fn commit_snapshot(
        &mut self,
        new_root_dir_inode: Pointer,
        new_salience_inode: Pointer,
    ) -> Result<(), FsError> {
        // Persist the in-memory dirty bitmap via the streaming path
        // (no transient dense-byte allocation; see
        // `persist_bitmap_streaming`).
        let new_bitmap_inode_ptr = persist_bitmap_streaming(
            self.cover.bytes_mut(),
            &self.map,
            &mut self.alloc,
            &mut self.dirty_bitmap,
            &mut self.dedup_index,
            &self.cdc_params,
        )?;

        let new_generation = self.generation + 1;
        let new_super_root = SuperRoot {
            root_dir_inode: new_root_dir_inode,
            dirty_bitmap_inode: new_bitmap_inode_ptr,
            salience_inode: new_salience_inode,
            generation: new_generation,
            ..SuperRoot::EMPTY
        };
        let new_super_root_ptr = alloc_and_write(
            self.cover.bytes_mut(),
            &self.map,
            &mut self.alloc,
            &new_super_root.encode(),
        )?;
        mark_pointer_dirty(&mut self.dirty_bitmap, &self.map, new_super_root_ptr);

        let committed_gen = anchor::commit_anchor_with_placement(
            self.cover.bytes_mut(),
            &self.map,
            &self.anchor_placement,
            new_super_root_ptr,
            self.generation,
        )?;
        debug_assert_eq!(committed_gen, new_generation);

        // Reclaim every chunk referenced by the old snapshot that
        // the new snapshot doesn't reach.
        reclaim_abandoned_chunks(
            &mut self.alloc,
            &self.map,
            self.cover.bytes(),
            TreeScope {
                root_dir_inode_ptr: self.root_inode_ptr,
                bitmap_inode_ptr: self.super_root.dirty_bitmap_inode,
                salience_inode_ptr: self.super_root.salience_inode,
                super_root_ptr: self.super_root_ptr,
            },
            TreeScope {
                root_dir_inode_ptr: new_root_dir_inode,
                bitmap_inode_ptr: new_bitmap_inode_ptr,
                salience_inode_ptr: new_super_root.salience_inode,
                super_root_ptr: new_super_root_ptr,
            },
        )?;

        // Update in-memory state.
        self.generation = new_generation;
        self.super_root = new_super_root;
        self.super_root_ptr = new_super_root_ptr;
        self.root_inode_ptr = new_root_dir_inode;
        self.root_inode = read_inode(self.cover.bytes(), &self.map, new_root_dir_inode)?;
        self.root_directory = read_directory(self.cover.bytes(), &self.map, new_root_dir_inode)?;

        // Rebuild the dedup index across the new tree + new bitmap
        // tree + salience tree. Next commit's writes (file,
        // directory, bitmap, or salience) all consult this shared
        // index.
        self.dedup_index = build_dedup_index(
            self.cover.bytes(),
            &self.map,
            new_root_dir_inode,
            new_bitmap_inode_ptr,
            new_super_root.salience_inode,
        )?;
        Ok(())
    }

    /// Chunk + write a user file's bytes through the shared dedup
    /// path and return the file's inode pointer. Thin wrapper over
    /// [`persist_as_byte_stream`] — file, directory, and bitmap
    /// writes all go through the same function so identical chunks
    /// across any of them collapse to one physical allocation.
    fn write_file_content(&mut self, data: &[u8]) -> Result<Pointer, FsError> {
        persist_as_byte_stream(
            self.cover.bytes_mut(),
            &self.map,
            &mut self.alloc,
            &mut self.dirty_bitmap,
            &mut self.dedup_index,
            &self.cdc_params,
            data,
        )
    }
}

// ==================================================================
// Path parsing
// ==================================================================

/// Validate and split an absolute path into its non-empty components.
/// Rejects relative paths, `.` / `..`, empty components with bytes
/// (NUL), and components longer than 255 bytes.
fn parse_path(path: &str) -> Result<Vec<&str>, FsError> {
    if !path.starts_with('/') {
        return Err(FsError::InvalidPath(path.to_owned()));
    }
    let mut out = Vec::new();
    for component in path.split('/') {
        if component.is_empty() {
            continue;
        }
        if component == "." || component == ".." {
            return Err(FsError::InvalidPath(path.to_owned()));
        }
        if component.len() > MAX_NAME_LEN {
            return Err(FsError::InvalidPath(path.to_owned()));
        }
        if component.as_bytes().contains(&0) {
            return Err(FsError::InvalidPath(path.to_owned()));
        }
        out.push(component);
    }
    Ok(out)
}

fn join_path(components: &[&str]) -> String {
    let mut s = String::with_capacity(components.iter().map(|c| c.len() + 1).sum());
    for c in components {
        s.push('/');
        s.push_str(c);
    }
    if s.is_empty() { "/".to_owned() } else { s }
}

// ==================================================================
// Cover-side directory + file helpers
// ==================================================================

/// Read + decode an inode at `ptr`.
fn read_inode(cover: &[u8], map: &TensorMap, ptr: Pointer) -> Result<Inode, FsError> {
    let mut buf = [0u8; INODE_BYTES];
    read_chunk(cover, map, ptr, 0, &mut buf)?;
    Ok(Inode::decode(&buf)?)
}

/// Read + decode the directory whose inode is at `ptr`.
fn read_directory(cover: &[u8], map: &TensorMap, ptr: Pointer) -> Result<Directory, FsError> {
    let bytes = load_byte_stream(cover, map, ptr)?;
    Ok(Directory::deserialize(&bytes)?)
}

/// Allocate + write a directory's serialized bytes through an inode
/// and return the inode's pointer.
fn write_directory_content(
    cover: &mut [u8],
    map: &TensorMap,
    allocator: &mut Allocator,
    dirty: &mut DirtyBitmap,
    dedup: &mut DedupIndex,
    cdc_params: &FastCdcParams,
    dir: &Directory,
) -> Result<Pointer, FsError> {
    let bytes = dir.serialize();
    persist_as_byte_stream(cover, map, allocator, dirty, dedup, cdc_params, &bytes)
}

/// Collect the `(slot, weight_index)` pairs across a placement's bit
/// positions. Deduplicates across the multiple bits of a single
/// weight.
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
    let ptr = allocator.alloc(map, bit_count).ok_or(FsError::OutOfSpace {
        requested_bits: bit_count,
    })?;
    write_chunk(cover, map, ptr, 0, data)?;
    Ok(ptr)
}

fn mark_pointer_dirty(dirty: &mut DirtyBitmap, map: &TensorMap, ptr: Pointer) {
    if ptr.is_null() {
        return;
    }
    let Some(slot) = map.slots.get(ptr.slot as usize) else {
        return;
    };
    let bpw = slot.stealable_bits_per_weight as u32;
    if bpw == 0 {
        return;
    }
    let weights = ptr.length_in_bits.div_ceil(bpw);
    dirty.mark_range(ptr.slot, ptr.start_weight, weights);
}

// ==================================================================
// Indirect-block helpers
// ==================================================================

/// Max pointers a "full" indirect block can hold at the average
/// chunk size. Used as the threshold for spilling to the next tier.
fn ptrs_per_indirect_block(data_chunk_size_bytes: usize) -> usize {
    data_chunk_size_bytes / Pointer::SIZE
}

fn max_chunks_for(cdc_params: &FastCdcParams) -> usize {
    let ppb = ptrs_per_indirect_block(cdc_params.avg_size);
    let ppb2 = ppb.saturating_mul(ppb);
    let ppb3 = ppb2.saturating_mul(ppb);
    NUM_DIRECT
        .saturating_add(ppb)
        .saturating_add(ppb2)
        .saturating_add(ppb3)
}

/// Allocate-or-reuse a content chunk. Consults the dedup index
/// first; on a miss, allocates a fresh chunk, writes, marks dirty,
/// and inserts the `(hash, pointer)` pair.
///
/// All byte-stream data and every indirect block routes through
/// here. Sharing one dedup index across files, directories, and the
/// dirty bitmap is what keeps `llmdb v2-init` from perturbing ~34M
/// Q8_0 weights on the first commit: the fresh bitmap is 17 MB of
/// zeros, every CDC chunk of which hashes identically, so the whole
/// inode tree collapses to a handful of physically-written chunks.
fn dedup_or_write(
    cover: &mut [u8],
    map: &TensorMap,
    allocator: &mut Allocator,
    dirty: &mut DirtyBitmap,
    dedup: &mut DedupIndex,
    chunk: &[u8],
) -> Result<Pointer, FsError> {
    let hash = hash_chunk(chunk);
    if let Some(p) = dedup.lookup(&hash) {
        return Ok(p);
    }
    let bit_count = (chunk.len() * 8) as u32;
    let p = allocator
        .alloc_preferring_dirty(map, bit_count, dirty)
        .ok_or(FsError::OutOfSpace {
            requested_bits: bit_count,
        })?;
    write_chunk(cover, map, p, 0, chunk)?;
    mark_pointer_dirty(dirty, map, p);
    dedup.insert(hash, p);
    Ok(p)
}

/// Outcome reported by [`dedup_or_write_no_mark`] so callers know
/// whether they need to mark the returned pointer dirty afterwards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriteOutcome {
    /// Chunk's hash matched the dedup index — no allocation, no
    /// write. Returned pointer was already in the index.
    Deduplicated,
    /// Chunk was newly written to the cover at the returned pointer.
    /// Caller must mark the pointer's weight range dirty.
    Wrote,
}

/// Like [`dedup_or_write`] but takes `dirty` as an immutable
/// reference (used for `alloc_preferring_dirty` decisions only) and
/// **does not** mark the resulting pointer. Returned `WriteOutcome`
/// tells the caller whether they need to mark.
///
/// Used exclusively by the bitmap streaming path: persisting the
/// bitmap reads bytes from the bitmap itself, so we can't hold a
/// `&mut DirtyBitmap` across the read+write cycle. Caller marks
/// every `Wrote` pointer after the streaming pass completes.
fn dedup_or_write_no_mark(
    cover: &mut [u8],
    map: &TensorMap,
    allocator: &mut Allocator,
    dirty: &DirtyBitmap,
    dedup: &mut DedupIndex,
    chunk: &[u8],
) -> Result<(Pointer, WriteOutcome), FsError> {
    let hash = hash_chunk(chunk);
    if let Some(p) = dedup.lookup(&hash) {
        return Ok((p, WriteOutcome::Deduplicated));
    }
    let bit_count = (chunk.len() * 8) as u32;
    let p = allocator
        .alloc_preferring_dirty(map, bit_count, dirty)
        .ok_or(FsError::OutOfSpace {
            requested_bits: bit_count,
        })?;
    write_chunk(cover, map, p, 0, chunk)?;
    dedup.insert(hash, p);
    Ok((p, WriteOutcome::Wrote))
}

/// Persist `bitmap` as a V2 byte stream, returning the inode
/// pointer for [`super_root.dirty_bitmap_inode`]. The streaming
/// path never allocates the dense `bitmap.total_bytes()` buffer —
/// for a 280 GB cover that's a 17.5 GB Vec we'd otherwise need.
///
/// Walks the bitmap one page at a time (snapshotting via
/// [`DirtyBitmap::page_at`] into a 4 KB stack-resident buffer),
/// feeds bytes into [`FastCdcStream`], and routes each emitted
/// chunk through [`dedup_or_write_no_mark`]. The dedup index
/// collapses the long runs of zeros (typical bitmap content) to
/// a single backing chunk; only non-zero regions allocate fresh
/// storage.
///
/// Marking happens in two phases:
///   1. During streaming, we can only hold an immutable borrow on
///      `dirty` (we're reading from it); chunk pointers from
///      `Wrote` outcomes get queued.
///   2. After streaming, we mutably borrow `dirty` and mark every
///      queued pointer plus the inode chunks themselves. The
///      persisted bitmap therefore lags the in-memory bitmap by
///      one commit's worth of marks — the same one-commit lag the
///      previous Vec-based path had.
fn persist_bitmap_streaming(
    cover: &mut [u8],
    map: &TensorMap,
    allocator: &mut Allocator,
    dirty: &mut DirtyBitmap,
    dedup: &mut DedupIndex,
    cdc_params: &FastCdcParams,
) -> Result<Pointer, FsError> {
    cdc_params.validate()?;
    let max_chunks = max_chunks_for(cdc_params);

    let mut chunker = FastCdcStream::new(*cdc_params);
    let mut all_ptrs: Vec<Pointer> = Vec::new();
    let mut written_ptrs: Vec<Pointer> = Vec::new();
    let mut total_emitted: u64 = 0;

    let total_bytes = dirty.total_bytes();
    let mut byte_offset: u64 = 0;
    let mut page_buf = [0u8; crate::v2::dirty::PAGE_BYTES];
    while byte_offset < total_bytes {
        let page_idx = (byte_offset / crate::v2::dirty::PAGE_BYTES as u64) as u32;
        let in_page = (byte_offset % crate::v2::dirty::PAGE_BYTES as u64) as usize;
        let bytes_left = total_bytes - byte_offset;
        let bytes_this_step =
            ((crate::v2::dirty::PAGE_BYTES - in_page) as u64).min(bytes_left) as usize;

        // Snapshot one page (or zeros if missing). The borrow on
        // `dirty` is scoped to this match — released before the
        // inner feed loop calls `dedup_or_write_no_mark`.
        match dirty.page_at(page_idx) {
            Some(p) => {
                page_buf[..bytes_this_step].copy_from_slice(&p[in_page..in_page + bytes_this_step]);
            }
            None => page_buf[..bytes_this_step].fill(0),
        }

        for &byte in &page_buf[..bytes_this_step] {
            if let Some(chunk) = chunker.feed(byte) {
                if all_ptrs.len() >= max_chunks {
                    return Err(FsError::FileTooLarge {
                        bytes: total_bytes as usize,
                        chunk_count: all_ptrs.len() + 1,
                        max_chunks,
                    });
                }
                total_emitted += chunk.len() as u64;
                let (p, outcome) =
                    dedup_or_write_no_mark(cover, map, allocator, dirty, dedup, &chunk)?;
                all_ptrs.push(p);
                if matches!(outcome, WriteOutcome::Wrote) {
                    written_ptrs.push(p);
                }
            }
        }

        byte_offset += bytes_this_step as u64;
    }
    if let Some(chunk) = chunker.flush() {
        if all_ptrs.len() >= max_chunks {
            return Err(FsError::FileTooLarge {
                bytes: total_bytes as usize,
                chunk_count: all_ptrs.len() + 1,
                max_chunks,
            });
        }
        total_emitted += chunk.len() as u64;
        let (p, outcome) = dedup_or_write_no_mark(cover, map, allocator, dirty, dedup, &chunk)?;
        all_ptrs.push(p);
        if matches!(outcome, WriteOutcome::Wrote) {
            written_ptrs.push(p);
        }
    }

    debug_assert_eq!(total_emitted, total_bytes);

    // Phase 2: mark every newly-written chunk.
    for p in &written_ptrs {
        mark_pointer_dirty(dirty, map, *p);
    }

    // Phase 3: build the inode (also writes indirect blocks via the
    // standard `dedup_or_write`, which marks dirty inline).
    let inode = build_inode(
        cover,
        map,
        allocator,
        dirty,
        dedup,
        cdc_params,
        total_emitted,
        &all_ptrs,
    )?;
    let inode_ptr = alloc_and_write(cover, map, allocator, &inode.encode())?;
    mark_pointer_dirty(dirty, map, inode_ptr);
    Ok(inode_ptr)
}

/// Stream-load a persisted dirty bitmap from the V2 inode at
/// `inode_ptr` into `bitmap`. Walks the inode's data chunks and
/// feeds bytes into [`DirtyBitmap::write_bytes_at`] one chunk at
/// a time — sparse pages allocate only for non-zero content.
fn load_bitmap_streaming(
    cover: &[u8],
    map: &TensorMap,
    inode_ptr: Pointer,
    bitmap: &mut DirtyBitmap,
) -> Result<(), FsError> {
    let inode = read_inode(cover, map, inode_ptr)?;
    let length = inode.length;
    let expected = bitmap.total_bytes();
    if length != expected {
        return Err(DirtyBitmapError::ByteCountMismatch {
            expected,
            got: length,
        }
        .into());
    }
    let data_ptrs = collect_data_pointers(&inode, cover, map)?;
    let mut offset: u64 = 0;
    let mut buf = vec![0u8; 16 * 1024];
    for ptr in data_ptrs {
        if offset >= length {
            break;
        }
        let chunk_capacity = byte_capacity(ptr) as usize;
        let take = chunk_capacity.min((length - offset) as usize);
        if buf.len() < take {
            buf.resize(take, 0);
        }
        read_chunk(cover, map, ptr, 0, &mut buf[..take])?;
        bitmap.write_bytes_at(offset, &buf[..take]);
        offset += take as u64;
    }
    if offset != length {
        return Err(FsError::OutOfSpace {
            requested_bits: ((length - offset) * 8) as u32,
        });
    }
    Ok(())
}

/// Thread `data_chunk_ptrs` through direct + indirect blocks and
/// build an [`Inode`] describing them. Indirect blocks also route
/// through [`dedup_or_write`] — when the incoming pointer list has
/// a lot of repeats (the bitmap-of-zeros case), many indirect
/// blocks end up with byte-identical content and collapse to one
/// physical chunk.
#[allow(clippy::too_many_arguments)]
fn build_inode(
    cover: &mut [u8],
    map: &TensorMap,
    allocator: &mut Allocator,
    dirty: &mut DirtyBitmap,
    dedup: &mut DedupIndex,
    cdc_params: &FastCdcParams,
    length: u64,
    data_chunk_ptrs: &[Pointer],
) -> Result<Inode, FsError> {
    let ppb = ptrs_per_indirect_block(cdc_params.avg_size);
    let chunk_count = data_chunk_ptrs.len();

    let mut direct = [Pointer::NULL; NUM_DIRECT];
    let direct_used = chunk_count.min(NUM_DIRECT);
    direct[..direct_used].copy_from_slice(&data_chunk_ptrs[..direct_used]);

    let mut cursor = direct_used;
    let single_indirect = if cursor < chunk_count {
        let end = (cursor + ppb).min(chunk_count);
        let ptr = write_pointer_list(
            cover,
            map,
            allocator,
            dirty,
            dedup,
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
            cover,
            map,
            allocator,
            dirty,
            dedup,
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
            cover,
            map,
            allocator,
            dirty,
            dedup,
            &data_chunk_ptrs[cursor..end],
            ppb,
        )?;
        cursor = end;
        ptr
    } else {
        Pointer::NULL
    };
    debug_assert_eq!(cursor, chunk_count);

    Ok(Inode {
        length,
        direct,
        single_indirect,
        double_indirect,
        triple_indirect,
    })
}

fn write_pointer_list(
    cover: &mut [u8],
    map: &TensorMap,
    allocator: &mut Allocator,
    dirty: &mut DirtyBitmap,
    dedup: &mut DedupIndex,
    pointers: &[Pointer],
) -> Result<Pointer, FsError> {
    let mut bytes = Vec::with_capacity(pointers.len() * Pointer::SIZE);
    for p in pointers {
        bytes.extend_from_slice(&p.encode());
    }
    dedup_or_write(cover, map, allocator, dirty, dedup, &bytes)
}

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

fn build_double_indirect(
    cover: &mut [u8],
    map: &TensorMap,
    allocator: &mut Allocator,
    dirty: &mut DirtyBitmap,
    dedup: &mut DedupIndex,
    data_ptrs: &[Pointer],
    ppb: usize,
) -> Result<Pointer, FsError> {
    let mut single_indirect_ptrs = Vec::new();
    for group in data_ptrs.chunks(ppb) {
        let sip = write_pointer_list(cover, map, allocator, dirty, dedup, group)?;
        single_indirect_ptrs.push(sip);
    }
    write_pointer_list(cover, map, allocator, dirty, dedup, &single_indirect_ptrs)
}

fn build_triple_indirect(
    cover: &mut [u8],
    map: &TensorMap,
    allocator: &mut Allocator,
    dirty: &mut DirtyBitmap,
    dedup: &mut DedupIndex,
    data_ptrs: &[Pointer],
    ppb: usize,
) -> Result<Pointer, FsError> {
    let group_size = ppb * ppb;
    let mut double_indirect_ptrs = Vec::new();
    for group in data_ptrs.chunks(group_size) {
        let dip = build_double_indirect(cover, map, allocator, dirty, dedup, group, ppb)?;
        double_indirect_ptrs.push(dip);
    }
    write_pointer_list(cover, map, allocator, dirty, dedup, &double_indirect_ptrs)
}

/// Walk an inode's tree and collect every non-null data-chunk pointer
/// (leaves only — indirect blocks are traversed but not emitted).
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

/// Persist an arbitrary byte stream as CDC-chunked data through a
/// fresh inode and return the inode's chunk pointer. Every content
/// chunk and every indirect block is routed through
/// [`dedup_or_write`], so identical chunks across files, directories,
/// the dirty bitmap, or within a single write all collapse to one
/// physical allocation.
///
/// Only the inode chunk itself bypasses dedup: an inode's bytes are
/// a length + 15 pointers, which differ by construction between files
/// / directories / commits.
fn persist_as_byte_stream(
    cover: &mut [u8],
    map: &TensorMap,
    allocator: &mut Allocator,
    dirty: &mut DirtyBitmap,
    dedup: &mut DedupIndex,
    cdc_params: &FastCdcParams,
    data: &[u8],
) -> Result<Pointer, FsError> {
    let ranges = chunk_ranges(data, cdc_params);
    let max_chunks = max_chunks_for(cdc_params);
    if ranges.len() > max_chunks {
        return Err(FsError::FileTooLarge {
            bytes: data.len(),
            chunk_count: ranges.len(),
            max_chunks,
        });
    }

    let mut data_chunk_ptrs = Vec::with_capacity(ranges.len());
    for range in &ranges {
        let chunk = &data[range.clone()];
        let ptr = dedup_or_write(cover, map, allocator, dirty, dedup, chunk)?;
        data_chunk_ptrs.push(ptr);
    }

    let inode = build_inode(
        cover,
        map,
        allocator,
        dirty,
        dedup,
        cdc_params,
        data.len() as u64,
        &data_chunk_ptrs,
    )?;
    let inode_ptr = alloc_and_write(cover, map, allocator, &inode.encode())?;
    mark_pointer_dirty(dirty, map, inode_ptr);
    Ok(inode_ptr)
}

/// Decode an inode chunk, walk its data chunks, concatenate, and
/// truncate to `inode.length`.
fn load_byte_stream(cover: &[u8], map: &TensorMap, inode_ptr: Pointer) -> Result<Vec<u8>, FsError> {
    let inode = read_inode(cover, map, inode_ptr)?;
    let length = inode.length as usize;
    let data_ptrs = collect_data_pointers(&inode, cover, map)?;

    let mut out = Vec::with_capacity(length);
    for ptr in data_ptrs {
        if out.len() >= length {
            break;
        }
        let chunk_bytes = byte_capacity(ptr) as usize;
        let this_read = chunk_bytes.min(length - out.len());
        let mut buf = vec![0u8; this_read];
        read_chunk(cover, map, ptr, 0, &mut buf)?;
        out.extend_from_slice(&buf);
    }
    out.truncate(length);
    Ok(out)
}

/// Every chunk position reachable from an inode (the inode itself +
/// indirect blocks + data chunks). Same as the old
/// `inode_tree_chunks` — used during mount reservation and inside
/// `walk_tree_chunks`.
fn inode_tree_chunks(
    cover: &[u8],
    map: &TensorMap,
    inode_ptr: Pointer,
) -> Result<std::collections::HashSet<(u16, u32, u32)>, FsError> {
    let mut set = std::collections::HashSet::new();
    if inode_ptr.is_null() {
        return Ok(set);
    }
    set.insert((
        inode_ptr.slot,
        inode_ptr.start_weight,
        inode_ptr.length_in_bits,
    ));
    let inode = read_inode(cover, map, inode_ptr)?;
    visit_inode_chunks(&inode, cover, map, |ptr| {
        set.insert((ptr.slot, ptr.start_weight, ptr.length_in_bits));
        Ok(())
    })?;
    Ok(set)
}

/// Recursively visit every pointer reachable from a directory tree —
/// each directory's inode + indirect blocks + content chunks, and
/// recursively into every child directory. The `visit` callback
/// fires per-pointer **before** any chunk at that pointer is read,
/// so corrupted pointers fail fast (e.g. at mount-time reservation)
/// rather than producing an opaque chunk-read error.
fn walk_tree_pointers<F>(
    cover: &[u8],
    map: &TensorMap,
    dir_inode_ptr: Pointer,
    visit: &mut F,
) -> Result<(), FsError>
where
    F: FnMut(Pointer) -> Result<(), FsError>,
{
    if dir_inode_ptr.is_null() {
        return Ok(());
    }
    // Visit the directory's inode chunk first — reads to follow
    // rely on its position being valid.
    visit(dir_inode_ptr)?;
    let inode = read_inode(cover, map, dir_inode_ptr)?;
    // Visit every pointer in the inode tree (indirect blocks + the
    // directory's content chunks). Each pointer is validated by
    // `visit` before the corresponding chunk is read downstream.
    visit_inode_chunks(&inode, cover, map, &mut *visit)?;
    // With the directory's tree validated + (optionally) reserved,
    // we can safely read its content and recurse.
    let dir = read_directory(cover, map, dir_inode_ptr)?;
    for entry in dir.entries() {
        match entry.kind {
            EntryKind::File => {
                visit(entry.inode)?;
                let finode = read_inode(cover, map, entry.inode)?;
                visit_inode_chunks(&finode, cover, map, &mut *visit)?;
            }
            EntryKind::Directory => {
                walk_tree_pointers(cover, map, entry.inode, visit)?;
            }
        }
    }
    Ok(())
}

/// Collect every `(slot, start, length_in_bits)` reachable from a
/// directory tree. Thin wrapper over [`walk_tree_pointers`].
fn collect_tree_chunks(
    cover: &[u8],
    map: &TensorMap,
    dir_inode_ptr: Pointer,
) -> Result<std::collections::HashSet<(u16, u32, u32)>, FsError> {
    let mut set = std::collections::HashSet::new();
    walk_tree_pointers(cover, map, dir_inode_ptr, &mut |ptr| {
        set.insert((ptr.slot, ptr.start_weight, ptr.length_in_bits));
        Ok(())
    })?;
    Ok(set)
}

/// A snapshot of "what's reachable from this commit": the root
/// directory tree, the bitmap inode tree, the salience inode tree
/// (post-B1), and the super-root chunk.
struct TreeScope {
    root_dir_inode_ptr: Pointer,
    bitmap_inode_ptr: Pointer,
    /// Calibration salience; `Pointer::NULL` on un-calibrated covers.
    /// Null-tolerant so every call site can populate it
    /// unconditionally from the current `SuperRoot`.
    salience_inode_ptr: Pointer,
    super_root_ptr: Pointer,
}

fn collect_scope_chunks(
    cover: &[u8],
    map: &TensorMap,
    scope: &TreeScope,
) -> Result<std::collections::HashSet<(u16, u32, u32)>, FsError> {
    let mut set = collect_tree_chunks(cover, map, scope.root_dir_inode_ptr)?;
    for e in inode_tree_chunks(cover, map, scope.bitmap_inode_ptr)? {
        set.insert(e);
    }
    if !scope.salience_inode_ptr.is_null() {
        for e in inode_tree_chunks(cover, map, scope.salience_inode_ptr)? {
            set.insert(e);
        }
    }
    if !scope.super_root_ptr.is_null() {
        set.insert((
            scope.super_root_ptr.slot,
            scope.super_root_ptr.start_weight,
            scope.super_root_ptr.length_in_bits,
        ));
    }
    Ok(set)
}

fn reclaim_abandoned_chunks(
    allocator: &mut Allocator,
    map: &TensorMap,
    cover: &[u8],
    old: TreeScope,
    new: TreeScope,
) -> Result<(), FsError> {
    let old_chunks = collect_scope_chunks(cover, map, &old)?;
    let new_chunks = collect_scope_chunks(cover, map, &new)?;
    for (slot, start_weight, length_in_bits) in &old_chunks {
        if new_chunks.contains(&(*slot, *start_weight, *length_in_bits)) {
            continue;
        }
        let ptr = Pointer {
            slot: *slot,
            start_weight: *start_weight,
            length_in_bits: *length_in_bits,
            flags: 0,
            reserved: 0,
        };
        allocator.free(map, ptr)?;
    }
    Ok(())
}

/// Rebuild the dedup index by walking every **content chunk**
/// reachable from the root directory tree + the bitmap's inode
/// tree. Content chunks are data leaves plus indirect-block chunks;
/// inode chunks are skipped because routing a data chunk through an
/// inode's hash would alias a 248-byte inode blob to whatever the
/// caller thinks it's reading (even granting that CDC params keep
/// data chunks larger than `INODE_BYTES`, skipping avoids the
/// foot-gun).
fn build_dedup_index(
    cover: &[u8],
    map: &TensorMap,
    root_dir_inode_ptr: Pointer,
    bitmap_inode_ptr: Pointer,
    salience_inode_ptr: Pointer,
) -> Result<DedupIndex, FsError> {
    let mut idx = DedupIndex::new();
    let mut hash_and_insert = |ptr: Pointer| -> Result<(), FsError> {
        if ptr.is_null() {
            return Ok(());
        }
        let chunk_bytes = byte_capacity(ptr) as usize;
        let mut buf = vec![0u8; chunk_bytes];
        read_chunk(cover, map, ptr, 0, &mut buf)?;
        idx.insert(hash_chunk(&buf), ptr);
        Ok(())
    };

    // Directory tree: every file / directory's content chunks.
    walk_content_chunks(cover, map, root_dir_inode_ptr, &mut hash_and_insert)?;

    // Bitmap tree: bitmap's data chunks + its indirect blocks.
    if !bitmap_inode_ptr.is_null() {
        let inode = read_inode(cover, map, bitmap_inode_ptr)?;
        visit_inode_chunks(&inode, cover, map, &mut hash_and_insert)?;
    }

    // Salience tree (B1): same shape as the bitmap — a metadata
    // inode with its own content chunks + indirects. Indexing it
    // lets two calibration runs with overlapping salience
    // distributions share physical storage, and collapses all-zero
    // regions (uncalibrated layers) to a single chunk.
    if !salience_inode_ptr.is_null() {
        let inode = read_inode(cover, map, salience_inode_ptr)?;
        visit_inode_chunks(&inode, cover, map, &mut hash_and_insert)?;
    }

    Ok(idx)
}

/// Recursively walk the content chunks reachable from a directory
/// tree — each directory's indirect blocks + content chunks, each
/// file's indirect blocks + data chunks. Inode chunks themselves
/// (dir inode, file inode) are **not** emitted.
fn walk_content_chunks<F>(
    cover: &[u8],
    map: &TensorMap,
    dir_inode_ptr: Pointer,
    visit: &mut F,
) -> Result<(), FsError>
where
    F: FnMut(Pointer) -> Result<(), FsError>,
{
    if dir_inode_ptr.is_null() {
        return Ok(());
    }
    let inode = read_inode(cover, map, dir_inode_ptr)?;
    visit_inode_chunks(&inode, cover, map, &mut *visit)?;
    let dir = read_directory(cover, map, dir_inode_ptr)?;
    for entry in dir.entries() {
        match entry.kind {
            EntryKind::File => {
                let finode = read_inode(cover, map, entry.inode)?;
                visit_inode_chunks(&finode, cover, map, &mut *visit)?;
            }
            EntryKind::Directory => {
                walk_content_chunks(cover, map, entry.inode, visit)?;
            }
        }
    }
    Ok(())
}

/// Visit every non-null pointer an inode transitively references —
/// data chunks AND indirect block chunks. Used by mount-time
/// reservation + reclamation tree walks.
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
