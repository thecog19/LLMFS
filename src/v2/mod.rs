//! V2 storage layer: Unix-inode substrate + CoW over a GGUF cover.
//!
//! Per DESIGN-NEW §15, V2 replaces V1's block-indexed superblock +
//! redirection layout with a per-file inode model (direct + indirect
//! pointers), hierarchical directories, variable-length chunks with
//! content-defined chunking, and a two-slot generational root pointer
//! for atomic CoW commits.
//!
//! This module is being built up incrementally; see the implementation
//! plan at `/home/suero/.claude/plans/why-masked-to-zero-velvet-nova.md`
//! for the sequence of milestones. First landed:
//!
//! - [`ceiling`] — persisted 256-weight-bucket summary of ceiling
//!   magnitudes, used by the allocator to find low-max-ceiling free
//!   runs without a per-weight scan.

pub mod alloc;
pub mod anchor;
pub mod cdc;
pub mod ceiling;
pub mod chunk;
pub mod dedup;
pub mod directory;
pub mod dirty;
pub mod freelist;
pub mod fs;
pub mod inode;
pub mod pointer;
pub mod super_root;
