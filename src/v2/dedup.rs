//! Content-hash dedup index (DESIGN-NEW §15.6).
//!
//! Maps a chunk's 128-bit BLAKE3 content hash to the [`Pointer`]
//! where that chunk already lives in the cover. The filesystem's
//! write path consults this index before allocating fresh space —
//! a hit means "reuse that pointer, don't write anything," which
//! turns duplicate content into zero new cover perturbation.
//!
//! Hash width is 128 bits (upper half of the 256-bit BLAKE3
//! digest). Collision probability against a population of 10^9
//! chunks is ~5×10⁻²¹ — effectively zero. BLAKE3 itself is
//! cryptographically strong, so deliberate collisions would require
//! adversarial effort unrelated to anything V2 does.
//!
//! Persistence: the index is currently rebuilt in-memory from the
//! current inode's data chunks on every mount (step 9 scope — see
//! [`crate::v2::fs::Filesystem::mount`]). A persisted form lives
//! in the super-root's `dedup_index_inode` slot for a future
//! milestone.

use std::collections::HashMap;

use crate::v2::pointer::Pointer;

/// A chunk's content hash — BLAKE3 truncated to 128 bits. Derives
/// `Ord` lexicographically so it can go into `BTreeSet`/`BTreeMap`
/// for deterministic iteration when we eventually persist the
/// index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ContentHash([u8; 16]);

impl From<[u8; 16]> for ContentHash {
    fn from(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }
}

impl From<ContentHash> for [u8; 16] {
    fn from(hash: ContentHash) -> Self {
        hash.0
    }
}

impl AsRef<[u8]> for ContentHash {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// Hash a chunk's content with BLAKE3, truncated to 128 bits.
pub fn hash_chunk(data: &[u8]) -> ContentHash {
    let full = blake3::hash(data);
    let bytes: [u8; 16] = full.as_bytes()[..16]
        .try_into()
        .expect("BLAKE3 digest is 32 bytes; truncation to 16 is infallible");
    ContentHash(bytes)
}

/// In-memory content-addressable index of currently-referenced data
/// chunks. Rebuilt from the current inode at mount time; updated
/// on every write.
#[derive(Debug, Default, Clone)]
pub struct DedupIndex {
    entries: HashMap<ContentHash, Pointer>,
}

impl DedupIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of entries. Each entry is one distinct content hash —
    /// a chunk with that content is reachable via its stored
    /// [`Pointer`].
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up the pointer for content with the given hash. `None`
    /// means no dedup candidate — the caller must allocate + write
    /// fresh.
    pub fn lookup(&self, hash: &ContentHash) -> Option<Pointer> {
        self.entries.get(hash).copied()
    }

    /// Insert or update a (hash → pointer) entry. On a subsequent
    /// rewrite of the file, the dedup path reuses whatever pointer
    /// is associated with the hash at lookup time.
    pub fn insert(&mut self, hash: ContentHash, pointer: Pointer) {
        self.entries.insert(hash, pointer);
    }

    /// Remove an entry; returns the removed pointer if present.
    pub fn remove(&mut self, hash: &ContentHash) -> Option<Pointer> {
        self.entries.remove(hash)
    }

    /// Clear all entries. Used when a write rebuilds the index from
    /// the new inode (so stale entries for abandoned chunks don't
    /// linger and misdirect future lookups to positions whose
    /// content may be reclaimed).
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Iterate over (hash, pointer) entries. Order is unspecified.
    pub fn iter(&self) -> impl Iterator<Item = (&ContentHash, &Pointer)> {
        self.entries.iter()
    }
}
