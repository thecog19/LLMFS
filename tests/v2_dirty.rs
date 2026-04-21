//! V2 DirtyBitmap unit tests.
//!
//! Tracks which weights have ever been written to (one bit per
//! eligible weight, packed). Allocator priority 2 (per DESIGN-NEW
//! §15.5) consults this to prefer already-perturbed positions over
//! pristine ones — writing over dirty bits adds less cover damage.

use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};
use llmdb::v2::dirty::DirtyBitmap;

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
}
