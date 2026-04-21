//! V2 dedup — content-hash index (BLAKE3 → 128-bit digest).
//!
//! The index maps a chunk's content hash to the `Pointer` where the
//! chunk already lives in the cover. `Filesystem::write` consults it
//! before allocating; a hit means "reuse that pointer, don't alloc
//! or write," which turns duplicate content into zero new cover
//! perturbation (DESIGN-NEW §15.6 Content-defined chunking + dedup).
//!
//! Unit-level tests (this file): the `DedupIndex` and `hash_chunk`
//! primitives in isolation. Integration tests (live `Filesystem`
//! dedup-hit behaviour) land in `tests/v2_fs_dedup.rs`.

use llmdb::v2::dedup::{ContentHash, DedupIndex, hash_chunk};
use llmdb::v2::pointer::Pointer;

fn sample_pointer(seed: u64) -> Pointer {
    Pointer {
        slot: (seed & 0xFFFF) as u16,
        start_weight: ((seed >> 16) & 0xFFFF_FFFF) as u32,
        length_in_bits: 4096,
        flags: 0,
        reserved: 0,
    }
}

// ------------------------------------------------------------------
// hash_chunk — BLAKE3 truncated to 128 bits
// ------------------------------------------------------------------

#[test]
fn hash_chunk_is_deterministic() {
    let data = b"content-defined chunking is the best";
    let a = hash_chunk(data);
    let b = hash_chunk(data);
    assert_eq!(a, b);
}

#[test]
fn hash_chunk_differs_for_different_content() {
    let a = hash_chunk(b"aaaaa");
    let b = hash_chunk(b"aaaab");
    assert_ne!(a, b);
}

#[test]
fn hash_chunk_is_sixteen_bytes() {
    let h = hash_chunk(b"");
    let bytes: [u8; 16] = h.into();
    assert_eq!(bytes.len(), 16);
}

#[test]
fn hash_chunk_empty_input_is_stable() {
    // BLAKE3 of empty input is a specific well-known value; just
    // assert the first call matches subsequent calls to guard against
    // future refactors introducing salt / state.
    let a = hash_chunk(&[]);
    let b = hash_chunk(&[]);
    assert_eq!(a, b);
}

// ------------------------------------------------------------------
// DedupIndex
// ------------------------------------------------------------------

#[test]
fn new_index_is_empty() {
    let idx = DedupIndex::new();
    assert_eq!(idx.len(), 0);
    assert!(idx.is_empty());
}

#[test]
fn insert_then_lookup_returns_pointer() {
    let mut idx = DedupIndex::new();
    let data = b"hello";
    let hash = hash_chunk(data);
    let ptr = sample_pointer(0xDEAD);

    idx.insert(hash, ptr);
    assert_eq!(idx.len(), 1);
    assert_eq!(idx.lookup(&hash), Some(ptr));
}

#[test]
fn lookup_miss_returns_none() {
    let idx = DedupIndex::new();
    let hash = hash_chunk(b"anything");
    assert_eq!(idx.lookup(&hash), None);
}

#[test]
fn second_insert_overwrites_first() {
    let mut idx = DedupIndex::new();
    let hash = hash_chunk(b"data");
    let a = sample_pointer(1);
    let b = sample_pointer(2);

    idx.insert(hash, a);
    idx.insert(hash, b);
    assert_eq!(idx.len(), 1);
    assert_eq!(idx.lookup(&hash), Some(b));
}

#[test]
fn remove_drops_entry() {
    let mut idx = DedupIndex::new();
    let hash = hash_chunk(b"data");
    idx.insert(hash, sample_pointer(0));
    assert!(!idx.is_empty());

    let removed = idx.remove(&hash);
    assert!(removed.is_some());
    assert!(idx.is_empty());
    assert_eq!(idx.lookup(&hash), None);
}

#[test]
fn remove_miss_returns_none() {
    let mut idx = DedupIndex::new();
    let hash = hash_chunk(b"never inserted");
    assert!(idx.remove(&hash).is_none());
}

#[test]
fn content_hash_equality_by_bytes() {
    let a = hash_chunk(b"foo");
    let b = hash_chunk(b"foo");
    let c = hash_chunk(b"bar");
    assert_eq!(a, b);
    assert_ne!(a, c);

    // Ord is derived lexicographically for BTreeSet-ability.
    let _sorted: std::collections::BTreeSet<ContentHash> =
        [a, b, c].into_iter().collect();
}
