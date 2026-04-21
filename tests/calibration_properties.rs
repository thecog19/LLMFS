//! Property-based invariants for the magnitude estimator and
//! placement layer. proptest generates random (but bounded) cover
//! topologies and weight values; every run re-asserts the core
//! contracts a year-from-now debugger would want to trust without
//! reading code:
//!
//! - `lowest_magnitude_weights` returns exactly `min(n, total)` items.
//! - Output is sorted ascending by magnitude.
//! - All `WeightRef`s are distinct (no weight appears twice).
//! - The output is globally optimal: no weight outside the result has
//!   a strictly smaller magnitude than the largest in the result.
//! - Repeated calls with identical inputs return identical output
//!   (deterministic — needed so reopen across sessions recovers the
//!   same Layer-0 metadata positions without persisting them).
//! - Prefix property: `result(n)` equals the first `n` of `result(n + k)`.
//!   Depends on the total-order tiebreaker between equal-magnitude
//!   weights; a regression here would manifest as metadata-position
//!   instability under small n changes.
//! - `compute_metadata_placement` always returns ≥ `needed_bits`
//!   positions when capacity allows.
//!
//! Scope: F16 and Q8_0 slots only, with finite non-NaN weight values.
//! Covers the hot path used by pristine + L0 metadata allocation;
//! K-quant property tests would need more elaborate generators to
//! keep the block structure valid.

use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::calibration::WeightRef;
use llmdb::stego::calibration::magnitude::{lowest_magnitude_weights, read_weight_abs};
use llmdb::stego::calibration::placement::compute_metadata_placement;
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};

use proptest::prelude::*;
use std::collections::HashSet;

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
        // Clamp to the largest finite f16 magnitude (0x7BFF)
        // rather than emit inf — keeps magnitude comparisons
        // well-defined for property tests.
        return (sign << 15) | (0x1E << 10) | 0x3FF;
    }
    let mantissa16 = (mantissa32 >> 13) as u16;
    (sign << 15) | ((exp16 as u16) << 10) | mantissa16
}

#[derive(Debug, Clone)]
enum SlotSpec {
    F16(Vec<f32>),
    Q8_0(Vec<i8>, f32),
}

/// Convert a spec into a (slot, bytes) pair at a given offset. The
/// slot's `bit_start` / `bit_end` are overwritten by `build_map`.
fn materialise(spec: &SlotSpec, data_offset: u64) -> (TensorSlot, Vec<u8>) {
    match spec {
        SlotSpec::F16(values) => {
            let mut bytes = vec![0_u8; values.len() * 2];
            for (i, v) in values.iter().enumerate() {
                let bits = f32_to_f16_bits(*v);
                bytes[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
            }
            let bits_per_weight = GgufQuantType::F16.stealable_bits_hint() as u64;
            let weight_count = values.len() as u64;
            (
                TensorSlot {
                    name: "prop.f16".to_owned(),
                    quant_type: GgufQuantType::F16,
                    tier: TensorTier::Tier1,
                    data_offset,
                    weight_count,
                    stealable_bits_per_weight: bits_per_weight as usize,
                    capacity_bits: weight_count * bits_per_weight,
                    bit_start: 0,
                    bit_end: weight_count * bits_per_weight,
                },
                bytes,
            )
        }
        SlotSpec::Q8_0(values, scale) => {
            // Block-aligned (32 weights per block). Generator guarantees this.
            let blocks = values.len() / 32;
            let mut bytes = vec![0_u8; blocks * 34];
            let scale_bits = f32_to_f16_bits(*scale);
            for b in 0..blocks {
                let base = b * 34;
                bytes[base..base + 2].copy_from_slice(&scale_bits.to_le_bytes());
                for i in 0..32 {
                    bytes[base + 2 + i] = values[b * 32 + i] as u8;
                }
            }
            let bits_per_weight = GgufQuantType::Q8_0.stealable_bits_hint() as u64;
            let weight_count = values.len() as u64;
            (
                TensorSlot {
                    name: "prop.q8_0".to_owned(),
                    quant_type: GgufQuantType::Q8_0,
                    tier: TensorTier::Tier1,
                    data_offset,
                    weight_count,
                    stealable_bits_per_weight: bits_per_weight as usize,
                    capacity_bits: weight_count * bits_per_weight,
                    bit_start: 0,
                    bit_end: weight_count * bits_per_weight,
                },
                bytes,
            )
        }
    }
}

/// Build a map from specs and return (map, mmap). Specs are laid out
/// contiguously in the mmap, each slot's `data_offset` set to its
/// byte start.
fn build_from_specs(specs: &[SlotSpec]) -> (TensorMap, Vec<u8>) {
    let mut mmap: Vec<u8> = Vec::new();
    let mut slots: Vec<TensorSlot> = Vec::with_capacity(specs.len());
    let mut bit_cursor = 0_u64;
    for spec in specs {
        let (mut slot, bytes) = materialise(spec, mmap.len() as u64);
        slot.bit_start = bit_cursor;
        bit_cursor += slot.capacity_bits;
        slot.bit_end = bit_cursor;
        slots.push(slot);
        mmap.extend(bytes);
    }
    (
        TensorMap {
            slots,
            total_capacity_bits: bit_cursor,
            total_capacity_bytes: bit_cursor / 8,
        },
        mmap,
    )
}

// ---- strategies ------------------------------------------------------

/// F16 values restricted to finite, small-magnitude floats. Rejects
/// NaN / inf by construction; the |w| distribution is the interesting
/// part of the ranking, not the extreme-value behaviour.
fn f16_value() -> impl Strategy<Value = f32> {
    (-100.0_f32..100.0_f32).prop_filter("finite only", |v| v.is_finite())
}

fn f16_slot_spec() -> impl Strategy<Value = SlotSpec> {
    // 1..32 weights: keep cases small so the O(N²) check loops stay
    // fast but large enough to exercise bucket+partition logic.
    prop::collection::vec(f16_value(), 1..32).prop_map(SlotSpec::F16)
}

fn q8_0_slot_spec() -> impl Strategy<Value = SlotSpec> {
    // Block-aligned: 1 or 2 blocks (32 or 64 weights).
    (1_usize..=2, -1.0_f32..1.0_f32, any::<u64>()).prop_flat_map(|(blocks, scale, seed)| {
        let n = blocks * 32;
        // Deterministic i8 vector seeded by `seed` — avoids
        // nesting proptest generators which slows things down.
        let values: Vec<i8> = (0..n)
            .map(|i| {
                ((seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(i as u64)
                    >> 56) as u8) as i8
            })
            .collect();
        Just(SlotSpec::Q8_0(values, scale))
    })
}

fn slot_spec() -> impl Strategy<Value = SlotSpec> {
    prop_oneof![f16_slot_spec(), q8_0_slot_spec()]
}

fn map_strategy() -> impl Strategy<Value = Vec<SlotSpec>> {
    prop::collection::vec(slot_spec(), 1..=3)
}

// ---- helpers ---------------------------------------------------------

/// Full ranking of every weight in the map — brute-force reference
/// for correctness assertions.
fn brute_force_rank(map: &TensorMap, mmap: &[u8]) -> Vec<(f32, WeightRef)> {
    let mut all: Vec<(f32, WeightRef)> = Vec::new();
    for (slot_idx, slot) in map.slots.iter().enumerate() {
        for weight_index in 0..slot.weight_count {
            let mag = read_weight_abs(mmap, slot, weight_index);
            all.push((
                mag,
                WeightRef {
                    slot_index: slot_idx as u32,
                    weight_index,
                },
            ));
        }
    }
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
    all
}

// ---- properties ------------------------------------------------------

proptest! {
    // Keep the case count modest — the brute-force reference is O(N²)
    // in weight count per run, and every property file runs in CI.
    #![proptest_config(ProptestConfig {
        cases: 64,
        .. ProptestConfig::default()
    })]

    #[test]
    fn returns_min_n_or_total(specs in map_strategy(), n in 0_usize..500) {
        let (map, mmap) = build_from_specs(&specs);
        let total: u64 = map.slots.iter().map(|s| s.weight_count).sum();
        let result = lowest_magnitude_weights(&mmap, &map, n);
        let expected_len = n.min(total as usize);
        prop_assert_eq!(result.len(), expected_len);
    }

    #[test]
    fn output_is_sorted_ascending_and_unique(specs in map_strategy(), n in 1_usize..300) {
        let (map, mmap) = build_from_specs(&specs);
        let result = lowest_magnitude_weights(&mmap, &map, n);
        let mags: Vec<f32> = result.iter().map(|r| {
            let slot = &map.slots[r.slot_index as usize];
            read_weight_abs(&mmap, slot, r.weight_index)
        }).collect();
        for w in mags.windows(2) {
            prop_assert!(w[0] <= w[1], "not sorted: {} > {}", w[0], w[1]);
        }
        let set: HashSet<WeightRef> = result.iter().copied().collect();
        prop_assert_eq!(set.len(), result.len(), "duplicate WeightRef in result");
    }

    #[test]
    fn result_is_globally_optimal(specs in map_strategy(), n in 1_usize..200) {
        let (map, mmap) = build_from_specs(&specs);
        let total: u64 = map.slots.iter().map(|s| s.weight_count).sum();
        let n = n.min(total as usize);
        if n == 0 { return Ok(()); }

        let result = lowest_magnitude_weights(&mmap, &map, n);
        let result_set: HashSet<WeightRef> = result.iter().copied().collect();
        let cutoff_mag = {
            let last = result.last().unwrap();
            let slot = &map.slots[last.slot_index as usize];
            read_weight_abs(&mmap, slot, last.weight_index)
        };
        // Anything left out must be >= cutoff_mag (with ties broken
        // by WeightRef, but proving strict globality just needs
        // magnitude dominance).
        for (slot_idx, slot) in map.slots.iter().enumerate() {
            for weight_index in 0..slot.weight_count {
                let r = WeightRef { slot_index: slot_idx as u32, weight_index };
                if result_set.contains(&r) { continue; }
                let mag = read_weight_abs(&mmap, slot, weight_index);
                prop_assert!(
                    mag >= cutoff_mag,
                    "excluded weight {:?} has mag {} < cutoff {}",
                    r, mag, cutoff_mag
                );
            }
        }
    }

    #[test]
    fn deterministic_across_calls(specs in map_strategy(), n in 0_usize..200) {
        let (map, mmap) = build_from_specs(&specs);
        let a = lowest_magnitude_weights(&mmap, &map, n);
        let b = lowest_magnitude_weights(&mmap, &map, n);
        let c = lowest_magnitude_weights(&mmap, &map, n);
        prop_assert_eq!(&a, &b);
        prop_assert_eq!(&b, &c);
    }

    #[test]
    fn prefix_property_n_is_prefix_of_n_plus_k(
        specs in map_strategy(),
        n in 0_usize..150,
        k in 1_usize..50
    ) {
        let (map, mmap) = build_from_specs(&specs);
        let total: u64 = map.slots.iter().map(|s| s.weight_count).sum();
        let n_ = n.min(total as usize);
        let m_ = (n + k).min(total as usize);
        let small = lowest_magnitude_weights(&mmap, &map, n_);
        let large = lowest_magnitude_weights(&mmap, &map, m_);
        prop_assert!(small.len() <= large.len());
        // Prefix equality: small[i] == large[i] for i in 0..small.len().
        for i in 0..small.len() {
            prop_assert_eq!(
                &small[i], &large[i],
                "prefix mismatch at {}: n={} vs n+k={}", i, n_, m_
            );
        }
    }

    #[test]
    fn ranking_agrees_with_brute_force(specs in map_strategy(), n in 1_usize..100) {
        let (map, mmap) = build_from_specs(&specs);
        let total: u64 = map.slots.iter().map(|s| s.weight_count).sum();
        let n = n.min(total as usize);
        if n == 0 { return Ok(()); }

        let result = lowest_magnitude_weights(&mmap, &map, n);
        let brute = brute_force_rank(&map, &mmap);
        // Brute force's first n items must equal our result, using
        // the same (mag, WeightRef) total order.
        for (i, r) in result.iter().enumerate() {
            prop_assert_eq!(r, &brute[i].1, "rank {} mismatch", i);
        }
    }

    #[test]
    fn placement_returns_at_least_needed_bits_when_capacity_allows(
        specs in map_strategy(),
        needed_bits in 0_u64..400
    ) {
        let (map, mmap) = build_from_specs(&specs);
        let capacity = map.total_capacity_bits;
        let p = compute_metadata_placement(&mmap, &map, needed_bits);
        if needed_bits <= capacity {
            prop_assert!(
                p.len_bits() >= needed_bits,
                "asked for {} bits, got {} (capacity {})",
                needed_bits, p.len_bits(), capacity
            );
        } else {
            // When out-of-capacity, placement returns what it can —
            // no more than total capacity.
            prop_assert!(p.len_bits() <= capacity);
        }
    }

    #[test]
    fn placement_positions_are_unique(specs in map_strategy(), needed_bits in 1_u64..200) {
        let (map, mmap) = build_from_specs(&specs);
        let p = compute_metadata_placement(&mmap, &map, needed_bits);
        let set: HashSet<_> = p.positions.iter().copied().collect();
        prop_assert_eq!(
            set.len(),
            p.positions.len(),
            "duplicate MetadataBitPos in placement"
        );
    }
}
