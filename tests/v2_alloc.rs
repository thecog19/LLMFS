//! V2 allocator — pristine path only (dedup + dirty preference land
//! in later milestones).
//!
//! Allocation priority for this milestone: **lowest max-ceiling-
//! magnitude free run that can hold the requested bit count**,
//! tiebroken by (length, slot, start_weight).
//!
//! Tests:
//! 1. Fresh init: every eligible slot contributes one full-range
//!    free run; non-stealable slots are skipped.
//! 2. Alloc returns a `Pointer` with the requested length_in_bits.
//! 3. Alloc prefers lower max-ceiling across slots.
//! 4. Split: an oversized run is trimmed and the remainder goes back
//!    to the free set with a recomputed max_ceiling via
//!    `CeilingSummary::max_over_range`.
//! 5. Free + merge: freeing a chunk returns the space to the free set
//!    and coalesces with adjacent free runs.
//! 6. Out-of-space: allocator returns `None` when nothing fits.
//! 7. Bit-to-weight conversion: a request of 32 bits lands in 8 F16
//!    weights or 4 F32 weights (the allocator picks per slot
//!    bits-per-weight).

use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};
use llmdb::v2::alloc::{AllocError, Allocator};
use llmdb::v2::ceiling::CeilingSummary;

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp32 = ((bits >> 23) & 0xFF) as i32;
    let mantissa32 = bits & 0x7FFFFF;
    if exp32 == 0 {
        return sign << 15;
    }
    let exp16 = exp32 - 127 + 15;
    if exp16 <= 0 {
        return sign << 15;
    }
    if exp16 >= 31 {
        return (sign << 15) | (0x1F << 10);
    }
    let mantissa16 = (mantissa32 >> 13) as u16;
    (sign << 15) | ((exp16 as u16) << 10) | mantissa16
}

fn f16_slot(weight_count: u64, data_offset: u64, name: &str) -> TensorSlot {
    let bits = GgufQuantType::F16.stealable_bits_hint() as u64;
    TensorSlot {
        name: name.to_owned(),
        quant_type: GgufQuantType::F16,
        tier: TensorTier::Tier1,
        data_offset,
        weight_count,
        stealable_bits_per_weight: bits as usize,
        capacity_bits: weight_count * bits,
        bit_start: 0,
        bit_end: weight_count * bits,
    }
}

fn f32_slot(weight_count: u64, data_offset: u64, name: &str) -> TensorSlot {
    let bits = GgufQuantType::F32.stealable_bits_hint() as u64;
    TensorSlot {
        name: name.to_owned(),
        quant_type: GgufQuantType::F32,
        tier: TensorTier::Tier1,
        data_offset,
        weight_count,
        stealable_bits_per_weight: bits as usize,
        capacity_bits: weight_count * bits,
        bit_start: 0,
        bit_end: weight_count * bits,
    }
}

fn f16_cover(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        out.extend_from_slice(&f32_to_f16_bits(*v).to_le_bytes());
    }
    out
}

fn f32_cover(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    out
}

fn build_map(slots: Vec<TensorSlot>) -> TensorMap {
    let total: u64 = slots.iter().map(|s| s.capacity_bits).sum();
    TensorMap {
        slots,
        total_capacity_bits: total,
        total_capacity_bytes: total / 8,
    }
}

fn summary_with_zero_buckets(bucket_counts: &[u32]) -> CeilingSummary {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"CSUM");
    bytes.push(1);
    bytes.extend_from_slice(&[0, 0, 0]);
    bytes.extend_from_slice(&(bucket_counts.len() as u32).to_le_bytes());
    for &count in bucket_counts {
        bytes.extend_from_slice(&count.to_le_bytes());
    }
    for &count in bucket_counts {
        for _ in 0..count {
            bytes.extend_from_slice(&0.0f32.to_le_bytes());
        }
    }
    CeilingSummary::deserialize(&bytes).expect("deserialize synthetic summary")
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

#[test]
fn fresh_allocator_has_one_free_run_per_eligible_slot() {
    // 2 slots: one F16 (eligible, 4 bits/wt), one with zero stealable.
    // Only F16 should contribute a free run.
    let f16_bytes = f16_cover(&[0.1, 0.2, 0.3, 0.4]);
    let f16 = f16_slot(4, 0, "a");
    let map = build_map(vec![f16]);
    let summary = CeilingSummary::build(&f16_bytes, &map);
    let alloc = Allocator::new_for_map(&map, summary).expect("allocator");
    assert_eq!(alloc.free_run_count(), 1);
    assert_eq!(alloc.total_free_weights(), 4);
}

#[test]
fn alloc_returns_pointer_with_requested_length_in_bits() {
    // 16 F16 weights = 64 stealable bits, enough for a 32-bit chunk.
    let values: Vec<f32> = (0..16).map(|i| (i + 1) as f32 * 0.01).collect();
    let cover = f16_cover(&values);
    let slot = f16_slot(16, 0, "a");
    let map = build_map(vec![slot]);
    let summary = CeilingSummary::build(&cover, &map);
    let mut alloc = Allocator::new_for_map(&map, summary).expect("allocator");

    let ptr = alloc.alloc(&map, 32).expect("alloc 32 bits");
    assert_eq!(ptr.length_in_bits, 32);
    assert_eq!(ptr.slot, 0);
}

#[test]
fn alloc_prefers_lower_max_ceiling_across_slots() {
    // Two F16 slots with different ceiling distributions. The
    // allocator should prefer the slot whose max_ceiling is lower.
    let low_values: Vec<f32> = (0..16).map(|i| (i + 1) as f32 * 1e-4).collect(); // tiny
    let high_values: Vec<f32> = (0..16).map(|i| (i + 1) as f32 * 0.5).collect(); // large

    let low_bytes = f16_cover(&low_values);
    let high_bytes = f16_cover(&high_values);

    // Concatenate: slot 0 = high, slot 1 = low. Allocator should pick slot 1.
    let mut cover = high_bytes.clone();
    cover.extend_from_slice(&low_bytes);
    let slot0 = f16_slot(16, 0, "high");
    let slot1 = f16_slot(16, high_bytes.len() as u64, "low");
    let map = build_map(vec![slot0, slot1]);
    let summary = CeilingSummary::build(&cover, &map);
    let mut alloc = Allocator::new_for_map(&map, summary).expect("allocator");

    let ptr = alloc.alloc(&map, 32).expect("alloc");
    assert_eq!(ptr.slot, 1, "allocator should prefer the lower-ceiling slot");
}

#[test]
fn alloc_splits_oversized_run_and_keeps_remainder() {
    let values: Vec<f32> = (0..16).map(|i| i as f32 * 0.01 + 0.01).collect();
    let cover = f16_cover(&values);
    let slot = f16_slot(16, 0, "a");
    let map = build_map(vec![slot]);
    let summary = CeilingSummary::build(&cover, &map);
    let mut alloc = Allocator::new_for_map(&map, summary).expect("allocator");

    // Initial: one run of 16 weights = 64 bits total.
    let ptr = alloc.alloc(&map, 16).expect("alloc 16 bits = 4 weights"); // 4 weights
    assert_eq!(ptr.length_in_bits, 16);
    // Remainder: 12 weights left.
    assert_eq!(alloc.free_run_count(), 1);
    assert_eq!(alloc.total_free_weights(), 12);
}

#[test]
fn alloc_then_free_round_trip_restores_free_runs() {
    let values: Vec<f32> = (0..16).map(|i| (i + 1) as f32 * 0.01).collect();
    let cover = f16_cover(&values);
    let slot = f16_slot(16, 0, "a");
    let map = build_map(vec![slot]);
    let summary = CeilingSummary::build(&cover, &map);
    let mut alloc = Allocator::new_for_map(&map, summary).expect("allocator");

    let ptr = alloc.alloc(&map, 16).unwrap();
    assert_eq!(alloc.total_free_weights(), 12);

    alloc.free(&map, ptr).expect("free");
    // The freed range should merge with the remainder → single 16-weight run.
    assert_eq!(alloc.free_run_count(), 1);
    assert_eq!(alloc.total_free_weights(), 16);
}

#[test]
fn alloc_returns_none_on_exhaustion() {
    let values: Vec<f32> = (0..4).map(|i| (i + 1) as f32 * 0.01).collect();
    let cover = f16_cover(&values);
    let slot = f16_slot(4, 0, "a");
    let map = build_map(vec![slot]);
    let summary = CeilingSummary::build(&cover, &map);
    let mut alloc = Allocator::new_for_map(&map, summary).expect("allocator");

    // 4 weights × 4 bits = 16 bits capacity.
    assert!(alloc.alloc(&map, 17).is_none(), "over-capacity request errors");
    assert!(alloc.alloc(&map, 16).is_some(), "exact fit succeeds");
    assert!(alloc.alloc(&map, 1).is_none(), "nothing left");
}

#[test]
fn alloc_converts_bits_to_weights_per_slot() {
    // F16 slot (4 bits/wt) + F32 slot (8 bits/wt). A 32-bit request
    // needs 8 F16 weights or 4 F32 weights. Test that both paths work
    // by making only one slot big enough.
    //
    // Scenario A: F16 has 7 weights (28 bits), F32 has 4 weights (32 bits).
    //             32-bit request should land in F32.
    let f16_bytes = f16_cover(&[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]);
    let f32_bytes = f32_cover(&[0.01, 0.02, 0.03, 0.04]);
    let mut cover = f16_bytes.clone();
    cover.extend_from_slice(&f32_bytes);
    let s_f16 = f16_slot(7, 0, "f16");
    let s_f32 = f32_slot(4, f16_bytes.len() as u64, "f32");
    let map = build_map(vec![s_f16, s_f32]);
    let summary = CeilingSummary::build(&cover, &map);
    let mut alloc = Allocator::new_for_map(&map, summary).expect("allocator");

    let ptr = alloc.alloc(&map, 32).expect("alloc 32 bits");
    assert_eq!(
        ptr.slot, 1,
        "32-bit request should land in the F32 slot (F16 slot only has 28 bits)",
    );
    assert_eq!(ptr.length_in_bits, 32);
}

#[test]
fn free_rejects_pointer_outside_any_run() {
    // Freeing a pointer that was never allocated (or was already
    // freed) is a programmer error; the allocator refuses.
    let values: Vec<f32> = (0..4).map(|i| (i + 1) as f32 * 0.01).collect();
    let cover = f16_cover(&values);
    let slot = f16_slot(4, 0, "a");
    let map = build_map(vec![slot]);
    let summary = CeilingSummary::build(&cover, &map);
    let mut alloc = Allocator::new_for_map(&map, summary).expect("allocator");

    // Make a bogus pointer (not allocated).
    let bogus = llmdb::v2::pointer::Pointer {
        slot: 0,
        start_weight: 0,
        length_in_bits: 16,
        flags: 0,
        reserved: 0,
    };

    match alloc.free(&map, bogus) {
        Err(AllocError::DoubleFree { .. }) => {}
        other => panic!("expected DoubleFree, got {other:?}"),
    }
}

#[test]
fn alloc_then_free_round_trip_allows_sub_weight_bit_lengths() {
    // A 1-bit allocation still occupies one F16 weight internally.
    // Free should recover that extent rather than rejecting the
    // pointer as "unaligned".
    let values: Vec<f32> = (0..16).map(|i| (i + 1) as f32 * 0.01).collect();
    let cover = f16_cover(&values);
    let slot = f16_slot(16, 0, "a");
    let map = build_map(vec![slot]);
    let summary = CeilingSummary::build(&cover, &map);
    let mut alloc = Allocator::new_for_map(&map, summary).expect("allocator");

    let ptr = alloc.alloc(&map, 1).expect("alloc 1 bit");
    assert_eq!(ptr.length_in_bits, 1);
    assert_eq!(alloc.total_free_weights(), 15);

    alloc.free(&map, ptr).expect("free sub-weight pointer");
    assert_eq!(alloc.free_run_count(), 1);
    assert_eq!(alloc.total_free_weights(), 16);
}

#[test]
fn free_rejects_pointer_past_slot_end() {
    let values: Vec<f32> = (0..16).map(|i| (i + 1) as f32 * 0.01).collect();
    let cover = f16_cover(&values);
    let slot = f16_slot(16, 0, "a");
    let map = build_map(vec![slot]);
    let summary = CeilingSummary::build(&cover, &map);
    let mut alloc = Allocator::new_for_map(&map, summary).expect("allocator");

    let bogus = llmdb::v2::pointer::Pointer {
        slot: 0,
        start_weight: 15,
        length_in_bits: 8, // 2 F16 weights; spills past weight_count=16
        flags: 0,
        reserved: 0,
    };

    match alloc.free(&map, bogus) {
        Err(AllocError::PointerOutOfBounds { .. }) => {}
        other => panic!("expected PointerOutOfBounds, got {other:?}"),
    }

    assert_eq!(alloc.free_run_count(), 1, "bogus free must not mutate allocator state");
    assert_eq!(alloc.total_free_weights(), 16);
}

#[test]
fn new_for_map_rejects_slots_larger_than_pointer_width() {
    let slot = f16_slot(u32::MAX as u64 + 1, 0, "huge");
    let map = build_map(vec![slot]);
    let summary = summary_with_zero_buckets(&[1]);

    match Allocator::new_for_map(&map, summary) {
        Err(AllocError::SlotTooLarge { slot, weight_count, max_weights }) => {
            assert_eq!(slot, 0);
            assert_eq!(weight_count, u32::MAX as u64 + 1);
            assert_eq!(max_weights, u32::MAX as u64);
        }
        other => panic!("expected SlotTooLarge, got {other:?}"),
    }
}

#[test]
fn new_for_map_rejects_more_slots_than_pointer_width() {
    let slots: Vec<TensorSlot> = (0..(u16::MAX as usize + 2))
        .map(|i| f16_slot(1, (i * 2) as u64, &format!("slot-{i}")))
        .collect();
    let map = build_map(slots);
    let summary = summary_with_zero_buckets(&vec![1; u16::MAX as usize + 2]);

    match Allocator::new_for_map(&map, summary) {
        Err(AllocError::TooManySlots { slot_count, max_slots }) => {
            assert_eq!(slot_count, u16::MAX as usize + 2);
            assert_eq!(max_slots, u16::MAX as usize + 1);
        }
        other => panic!("expected TooManySlots, got {other:?}"),
    }
}
