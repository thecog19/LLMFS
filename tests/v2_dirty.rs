//! V2 sparse-page DirtyBitmap integration tests.
//!
//! Tracks which weights have ever been written to (one bit per
//! eligible weight, packed). Allocator priority 2 (per DESIGN-NEW
//! §15.5) consults this to prefer already-perturbed positions over
//! pristine ones — writing over dirty bits adds less cover damage.
//!
//! The bitmap stores its bits in a sparse `BTreeMap` of 4 KB pages;
//! pages allocate lazily on first `mark` (or first non-zero byte
//! written via `write_bytes_at`). These tests cover the
//! sparse-allocation invariants alongside the original behavioural
//! contract from the dense V1.

use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};
use llmdb::v2::dirty::{DirtyBitmap, PAGE_BYTES};

fn f16_slot(weight_count: u64, name: &str) -> TensorSlot {
    TensorSlot {
        name: name.to_owned(),
        quant_type: GgufQuantType::F16,
        tier: TensorTier::Tier1,
        data_offset: 0,
        weight_count,
        stealable_bits_per_weight: GgufQuantType::F16.stealable_bits_hint(),
        capacity_bits: weight_count * GgufQuantType::F16.stealable_bits_hint() as u64,
        bit_start: 0,
        bit_end: weight_count * GgufQuantType::F16.stealable_bits_hint() as u64,
    }
}

fn two_slot_map() -> TensorMap {
    let a = f16_slot(100, "a");
    let b = f16_slot(50, "b");
    let total_bits = a.capacity_bits + b.capacity_bits;
    TensorMap {
        slots: vec![a, b],
        total_capacity_bits: total_bits,
        total_capacity_bytes: total_bits / 8,
    }
}

#[test]
fn new_bitmap_is_all_clean() {
    let map = two_slot_map();
    let bm = DirtyBitmap::new(&map);
    for slot in 0_u16..2 {
        let slot_len = map.slots[slot as usize].weight_count as u32;
        for w in 0..slot_len {
            assert!(!bm.is_dirty(slot, w), "fresh bitmap must be all-clean");
        }
    }
}

#[test]
fn new_bitmap_allocates_no_pages() {
    let map = two_slot_map();
    let bm = DirtyBitmap::new(&map);
    assert_eq!(bm.allocated_page_count(), 0);
    assert_eq!(bm.set_count(), 0);
}

#[test]
fn mark_then_is_dirty() {
    let map = two_slot_map();
    let mut bm = DirtyBitmap::new(&map);
    bm.mark(0, 17);
    assert!(bm.is_dirty(0, 17));
    assert!(!bm.is_dirty(0, 16));
    assert!(!bm.is_dirty(0, 18));
    assert!(!bm.is_dirty(1, 17), "different slot stays clean");
}

#[test]
fn mark_range_sets_every_bit_in_range() {
    let map = two_slot_map();
    let mut bm = DirtyBitmap::new(&map);
    bm.mark_range(0, 10, 20);
    for w in 10..30 {
        assert!(bm.is_dirty(0, w), "w={w} should be dirty");
    }
    assert!(!bm.is_dirty(0, 9), "just before range stays clean");
    assert!(!bm.is_dirty(0, 30), "just after range stays clean");
}

#[test]
fn is_range_dirty_requires_all_weights_dirty() {
    let map = two_slot_map();
    let mut bm = DirtyBitmap::new(&map);

    bm.mark_range(0, 0, 5);
    assert!(bm.is_range_dirty(0, 0, 5), "fully-dirty range reports true");
    assert!(
        !bm.is_range_dirty(0, 0, 6),
        "extending into clean bit reports false",
    );
}

#[test]
fn is_range_dirty_zero_length_is_vacuously_true() {
    let map = two_slot_map();
    let bm = DirtyBitmap::new(&map);
    assert!(bm.is_range_dirty(0, 0, 0));
}

#[test]
fn slot_offsets_keep_slots_independent() {
    // Slot 0 has 100 weights. Marking (1, 0) should set the bit
    // at logical index 100 — not at 0.
    let map = two_slot_map();
    let mut bm = DirtyBitmap::new(&map);
    bm.mark(1, 0);
    assert!(!bm.is_dirty(0, 0), "slot 1 bit must not leak into slot 0");
    assert!(bm.is_dirty(1, 0));
}

#[test]
fn total_bits_matches_sum_of_slot_weight_counts() {
    let map = two_slot_map();
    let bm = DirtyBitmap::new(&map);
    assert_eq!(bm.total_bits(), 100 + 50);
}

#[test]
fn mark_is_idempotent() {
    let map = two_slot_map();
    let mut bm = DirtyBitmap::new(&map);
    bm.mark(0, 42);
    bm.mark(0, 42);
    assert!(bm.is_dirty(0, 42));
    assert_eq!(bm.set_count(), 1);
}

#[test]
fn write_to_then_write_bytes_at_round_trips() {
    let map = two_slot_map();
    let mut bm = DirtyBitmap::new(&map);
    bm.mark(0, 3);
    bm.mark(0, 99);
    bm.mark(1, 0);
    bm.mark(1, 49);

    let mut buf = Vec::new();
    bm.write_to(&mut buf).expect("write_to");
    assert_eq!(buf.len(), bm.total_bytes() as usize);

    let mut restored = DirtyBitmap::new(&map);
    restored.write_bytes_at(0, &buf);

    for w in 0_u32..100 {
        assert_eq!(bm.is_dirty(0, w), restored.is_dirty(0, w), "slot 0 bit {w}");
    }
    for w in 0_u32..50 {
        assert_eq!(bm.is_dirty(1, w), restored.is_dirty(1, w), "slot 1 bit {w}");
    }
    assert_eq!(bm.set_count(), restored.set_count());
}

#[test]
fn write_to_emits_total_bytes_even_when_sparse() {
    let map = two_slot_map();
    let bm = DirtyBitmap::new(&map);
    let mut buf = Vec::new();
    bm.write_to(&mut buf).expect("write_to");
    assert_eq!(buf.len() as u64, bm.total_bytes());
    assert!(buf.iter().all(|&b| b == 0), "no marks → all zeros");
}

#[test]
fn marks_within_one_page_share_a_page() {
    // PAGE_BYTES * 8 weights fit in a single page.
    let weights_per_page = (PAGE_BYTES * 8) as u64;
    let big = TensorMap {
        slots: vec![f16_slot(weights_per_page * 4, "big")],
        total_capacity_bits: weights_per_page * 4,
        total_capacity_bytes: weights_per_page * 4 / 8,
    };
    let mut bm = DirtyBitmap::new(&big);
    for i in 0..100 {
        bm.mark(0, i);
    }
    assert_eq!(
        bm.allocated_page_count(),
        1,
        "100 marks within one page should share a page"
    );
    assert_eq!(bm.set_count(), 100);
}

#[test]
fn distant_marks_use_separate_pages() {
    let weights_per_page = (PAGE_BYTES * 8) as u64;
    let big = TensorMap {
        slots: vec![f16_slot(weights_per_page * 4, "big")],
        total_capacity_bits: weights_per_page * 4,
        total_capacity_bytes: weights_per_page * 4 / 8,
    };
    let mut bm = DirtyBitmap::new(&big);
    bm.mark(0, 0);
    bm.mark(0, weights_per_page as u32 + 1); // page 1
    bm.mark(0, weights_per_page as u32 * 3 + 1); // page 3
    assert_eq!(bm.allocated_page_count(), 3);
    assert_eq!(bm.set_count(), 3);
}

#[test]
fn write_bytes_at_zero_bytes_does_not_allocate() {
    let weights_per_page = (PAGE_BYTES * 8) as u64;
    let big = TensorMap {
        slots: vec![f16_slot(weights_per_page * 4, "big")],
        total_capacity_bits: weights_per_page * 4,
        total_capacity_bytes: weights_per_page * 4 / 8,
    };
    let mut bm = DirtyBitmap::new(&big);
    let zeros = vec![0u8; bm.total_bytes() as usize];
    bm.write_bytes_at(0, &zeros);
    assert_eq!(
        bm.allocated_page_count(),
        0,
        "writing all-zero bytes must allocate nothing"
    );
    assert_eq!(bm.set_count(), 0);
}

#[test]
fn write_bytes_at_partial_page_only_allocates_when_nonzero() {
    let weights_per_page = (PAGE_BYTES * 8) as u64;
    let big = TensorMap {
        slots: vec![f16_slot(weights_per_page * 4, "big")],
        total_capacity_bits: weights_per_page * 4,
        total_capacity_bytes: weights_per_page * 4 / 8,
    };
    let mut bm = DirtyBitmap::new(&big);

    // Page 0 stays zero. Page 1 gets a single non-zero byte. Page 2
    // and 3 stay zero.
    let mut bytes = vec![0u8; bm.total_bytes() as usize];
    bytes[PAGE_BYTES] = 0x80; // first byte of page 1
    bm.write_bytes_at(0, &bytes);
    assert_eq!(bm.allocated_page_count(), 1, "only page 1 should allocate");
    // Bit position: byte PAGE_BYTES, shift 7 (LSB-first) means bit
    // (PAGE_BYTES * 8) + 7.
    assert!(bm.is_dirty(0, (PAGE_BYTES * 8) as u32 + 7));
    assert!(!bm.is_dirty(0, 0));
}
