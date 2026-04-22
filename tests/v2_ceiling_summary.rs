//! V2 ceiling-magnitude bucket summary (`src/v2/ceiling.rs`).
//!
//! Summary records the max ceiling magnitude per 256-weight bucket,
//! per slot. Used by the V2 allocator to answer "max ceiling over
//! this free run ≥ L weights" without scanning every weight
//! individually; queries are slightly pessimistic (overestimate by at
//! most one bucket's range of weights) but precision-adequate for
//! the free-run priority queue.
//!
//! Invariants under test:
//! 1. **Bucket max is exact** — `summary[slot][bucket]` equals
//!    `max over weights in that bucket of ceiling_magnitude(w)`.
//! 2. **Range max is an upper bound** — `max_over_range(a, len)` is
//!    `>= max(ceiling_magnitude(w))` for `w ∈ [a, a+len)`.
//! 3. **Range max on full-bucket ranges is exact**.
//! 4. **Empty range returns 0**.
//! 5. **Bucket count** — one bucket per 256 weights (rounded up).
//! 6. **Serialization round-trips** — serialize then deserialize
//!    returns identical state.

use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::calibration::magnitude::read_weight_ceiling_abs;
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};
use llmdb::v2::ceiling::{BUCKET_SIZE, CeilingSummary};

// ------------------------------------------------------------------
// Fixture helpers
// ------------------------------------------------------------------

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

fn f16_slot(weight_count: u64, data_offset: u64) -> TensorSlot {
    let bits = GgufQuantType::F16.stealable_bits_hint() as u64;
    TensorSlot {
        name: format!("summary.f16.{weight_count}"),
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

fn synthesize_f16_cover(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        out.extend_from_slice(&f32_to_f16_bits(*v).to_le_bytes());
    }
    out
}

fn single_slot_map(slot: TensorSlot) -> TensorMap {
    let total = slot.capacity_bits;
    TensorMap {
        slots: vec![slot],
        total_capacity_bits: total,
        total_capacity_bytes: total / 8,
    }
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

#[test]
fn bucket_size_is_256() {
    assert_eq!(BUCKET_SIZE, 256);
}

#[test]
fn bucket_count_is_ceil_div_256() {
    // 300 weights → 2 buckets (256 full + 44 partial)
    let values: Vec<f32> = (0..300).map(|i| (i as f32) * 0.001).collect();
    let bytes = synthesize_f16_cover(&values);
    let map = single_slot_map(f16_slot(300, 0));
    let summary = CeilingSummary::build(&bytes, &map);
    assert_eq!(summary.bucket_count(0), 2);

    // 256 weights → 1 bucket (exactly full)
    let values: Vec<f32> = (0..256).map(|i| (i as f32) * 0.001).collect();
    let bytes = synthesize_f16_cover(&values);
    let map = single_slot_map(f16_slot(256, 0));
    let summary = CeilingSummary::build(&bytes, &map);
    assert_eq!(summary.bucket_count(0), 1);

    // 1 weight → 1 bucket (partial)
    let values: Vec<f32> = vec![0.123];
    let bytes = synthesize_f16_cover(&values);
    let map = single_slot_map(f16_slot(1, 0));
    let summary = CeilingSummary::build(&bytes, &map);
    assert_eq!(summary.bucket_count(0), 1);
}

#[test]
fn bucket_max_matches_brute_force() {
    // 512 weights, varied magnitudes to make bucket maxes non-trivial
    let values: Vec<f32> = (0..512)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            sign * ((i as f32) * 0.01 + 0.001)
        })
        .collect();
    let bytes = synthesize_f16_cover(&values);
    let slot = f16_slot(512, 0);
    let map = single_slot_map(slot.clone());
    let summary = CeilingSummary::build(&bytes, &map);

    // For each bucket, brute-force the max ceiling of its weights.
    for b in 0..summary.bucket_count(0) {
        let start = (b as u64) * BUCKET_SIZE;
        let end = (start + BUCKET_SIZE).min(slot.weight_count);
        let mut expected: f32 = 0.0;
        for w in start..end {
            let c = read_weight_ceiling_abs(&bytes, &slot, w);
            if c > expected {
                expected = c;
            }
        }
        let got = summary.bucket_max(0, b);
        assert!(
            (got - expected).abs() <= expected * 1e-6 + 1e-12,
            "bucket {b} mismatch: got {got}, expected {expected}",
        );
    }
}

#[test]
fn max_over_range_empty_is_zero() {
    let values: Vec<f32> = vec![1.0, 2.0, 3.0];
    let bytes = synthesize_f16_cover(&values);
    let map = single_slot_map(f16_slot(3, 0));
    let summary = CeilingSummary::build(&bytes, &map);
    assert_eq!(summary.max_over_range(0, 0, 0), 0.0);
    assert_eq!(summary.max_over_range(0, 1, 0), 0.0);
}

#[test]
fn max_over_range_full_bucket_is_exact() {
    let values: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
    let bytes = synthesize_f16_cover(&values);
    let slot = f16_slot(256, 0);
    let map = single_slot_map(slot.clone());
    let summary = CeilingSummary::build(&bytes, &map);

    let got = summary.max_over_range(0, 0, 256);
    let mut expected: f32 = 0.0;
    for w in 0..256 {
        let c = read_weight_ceiling_abs(&bytes, &slot, w);
        if c > expected {
            expected = c;
        }
    }
    assert!(
        (got - expected).abs() <= expected * 1e-6 + 1e-12,
        "full-bucket range max mismatch: got {got}, expected {expected}",
    );
}

#[test]
fn max_over_range_is_upper_bound_for_partial_buckets() {
    // 512 weights across 2 buckets. Query a range that crosses the
    // bucket boundary non-alignedly.
    let values: Vec<f32> = (0..512)
        .map(|i| if i % 7 == 0 { 10.0 } else { 0.001 })
        .collect();
    let bytes = synthesize_f16_cover(&values);
    let slot = f16_slot(512, 0);
    let map = single_slot_map(slot.clone());
    let summary = CeilingSummary::build(&bytes, &map);

    // Query [100, 400): crosses bucket 0 (256 weights) into bucket 1.
    let got = summary.max_over_range(0, 100, 300);
    let mut exact_in_range: f32 = 0.0;
    for w in 100..400 {
        let c = read_weight_ceiling_abs(&bytes, &slot, w);
        if c > exact_in_range {
            exact_in_range = c;
        }
    }
    assert!(
        got >= exact_in_range - exact_in_range * 1e-6 - 1e-12,
        "range max is not an upper bound: got {got}, exact {exact_in_range}",
    );
}

#[test]
fn max_over_range_respects_slot_boundary() {
    // 300 weights (2 buckets: 256 + 44). A query beyond 300 should
    // clamp, not panic.
    let values: Vec<f32> = (0..300).map(|i| (i as f32) * 0.01).collect();
    let bytes = synthesize_f16_cover(&values);
    let map = single_slot_map(f16_slot(300, 0));
    let summary = CeilingSummary::build(&bytes, &map);

    let got = summary.max_over_range(0, 0, 300);
    assert!(got > 0.0);
}

#[test]
fn multi_slot_summary() {
    // Two F16 slots side by side; summary should record each independently.
    let slot_a_values: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
    let slot_b_values: Vec<f32> = (0..512).map(|i| (i as f32) * 0.001 + 100.0).collect();
    let mut bytes = synthesize_f16_cover(&slot_a_values);
    let slot_a = f16_slot(256, 0);
    let slot_b_offset = bytes.len() as u64;
    bytes.extend(synthesize_f16_cover(&slot_b_values));
    let slot_b = f16_slot(512, slot_b_offset);

    let total_bits = slot_a.capacity_bits + slot_b.capacity_bits;
    let map = TensorMap {
        slots: vec![slot_a.clone(), slot_b.clone()],
        total_capacity_bits: total_bits,
        total_capacity_bytes: total_bits / 8,
    };
    let summary = CeilingSummary::build(&bytes, &map);

    assert_eq!(summary.bucket_count(0), 1); // slot_a: 256 weights
    assert_eq!(summary.bucket_count(1), 2); // slot_b: 512 weights

    // Slot A's bucket max should be less than slot B's (slot B has
    // much larger-magnitude weights).
    assert!(summary.bucket_max(0, 0) < summary.bucket_max(1, 0));
}

#[test]
fn serialize_roundtrip() {
    let values: Vec<f32> = (0..600).map(|i| ((i as f32) - 300.0) * 0.005).collect();
    let bytes = synthesize_f16_cover(&values);
    let map = single_slot_map(f16_slot(600, 0));
    let summary = CeilingSummary::build(&bytes, &map);

    let serialized = summary.serialize();
    let restored =
        CeilingSummary::deserialize(&serialized).expect("deserialize of serialized summary");

    assert_eq!(summary.slot_count(), restored.slot_count());
    for slot_idx in 0..summary.slot_count() {
        let buckets = summary.bucket_count(slot_idx);
        assert_eq!(buckets, restored.bucket_count(slot_idx));
        for b in 0..buckets {
            assert_eq!(
                summary.bucket_max(slot_idx, b),
                restored.bucket_max(slot_idx, b),
                "mismatch at slot {slot_idx} bucket {b}",
            );
        }
    }
}

#[test]
fn deserialize_rejects_bad_magic() {
    let bytes = vec![0u8; 16];
    assert!(CeilingSummary::deserialize(&bytes).is_err());
}

#[test]
fn deserialize_rejects_truncated_bucket_data() {
    // Build a valid summary then truncate the byte stream mid-bucket.
    let values: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
    let bytes = synthesize_f16_cover(&values);
    let map = single_slot_map(f16_slot(256, 0));
    let summary = CeilingSummary::build(&bytes, &map);

    let mut serialized = summary.serialize();
    serialized.truncate(serialized.len() - 2);
    assert!(CeilingSummary::deserialize(&serialized).is_err());
}
