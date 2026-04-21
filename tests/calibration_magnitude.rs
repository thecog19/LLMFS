//! Integration tests for the magnitude-only salience estimator
//! (`stego::calibration::magnitude`). Synthesizes mmap-shaped byte
//! buffers with controlled weight values, builds a `TensorMap`
//! pointing into them, and asserts the ranking comes back in the
//! expected order.

use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::calibration::WeightRef;
use llmdb::stego::calibration::magnitude::{
    lowest_magnitude_weights, lowest_magnitude_weights_for_bits, read_weight_abs,
};
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};

/// Encode an f32 as fp16 bits. Round-to-nearest-even, no special-case
/// handling for inf/NaN/subnormal — fine for test fixtures.
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

fn f16_slot_with_values(values: &[f32], data_offset: u64) -> (TensorSlot, Vec<u8>) {
    let mut bytes = vec![0_u8; values.len() * 2];
    for (i, v) in values.iter().enumerate() {
        let bits = f32_to_f16_bits(*v);
        bytes[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
    }
    let weight_count = values.len() as u64;
    let bits_per_weight = GgufQuantType::F16.stealable_bits_hint() as u64;
    let slot = TensorSlot {
        name: "test.f16".to_owned(),
        quant_type: GgufQuantType::F16,
        tier: TensorTier::Tier1,
        data_offset,
        weight_count,
        stealable_bits_per_weight: bits_per_weight as usize,
        capacity_bits: weight_count * bits_per_weight,
        bit_start: 0,
        bit_end: weight_count * bits_per_weight,
    };
    (slot, bytes)
}

fn q8_0_slot_with_values(values: &[i8], scale: f32, data_offset: u64) -> (TensorSlot, Vec<u8>) {
    // Q8_0 block is exactly 32 weights; this helper packs `values.len()`
    // worth of one-block fixtures and expects len() <= 32.
    assert!(
        values.len() <= 32,
        "this helper is single-block; pass <= 32 values"
    );
    let mut block = vec![0_u8; 34];
    let scale_bits = f32_to_f16_bits(scale);
    block[0..2].copy_from_slice(&scale_bits.to_le_bytes());
    for (i, v) in values.iter().enumerate() {
        block[2 + i] = (*v) as u8;
    }
    let weight_count = values.len() as u64;
    let bits_per_weight = GgufQuantType::Q8_0.stealable_bits_hint() as u64;
    let slot = TensorSlot {
        name: "test.q8_0".to_owned(),
        quant_type: GgufQuantType::Q8_0,
        tier: TensorTier::Tier1,
        data_offset,
        weight_count,
        stealable_bits_per_weight: bits_per_weight as usize,
        capacity_bits: weight_count * bits_per_weight,
        bit_start: 0,
        bit_end: weight_count * bits_per_weight,
    };
    (slot, block)
}

fn build_map(slots: Vec<TensorSlot>) -> TensorMap {
    let mut bit_cursor = 0_u64;
    let mut adjusted = Vec::with_capacity(slots.len());
    for slot in slots {
        let bits = slot.capacity_bits;
        let mut s = slot;
        s.bit_start = bit_cursor;
        s.bit_end = bit_cursor + bits;
        bit_cursor += bits;
        adjusted.push(s);
    }
    TensorMap {
        slots: adjusted,
        total_capacity_bits: bit_cursor,
        total_capacity_bytes: bit_cursor / 8,
    }
}

#[test]
fn ranks_single_f16_tensor_by_magnitude_ascending() {
    let (slot, bytes) = f16_slot_with_values(&[10.0, 0.5, 100.0, 0.1], 0);
    let map = build_map(vec![slot]);

    let ranked = lowest_magnitude_weights(&bytes, &map, 4);
    let weight_indices: Vec<u64> = ranked.iter().map(|r| r.weight_index).collect();
    // |0.1| < |0.5| < |10.0| < |100.0|
    assert_eq!(weight_indices, vec![3, 1, 0, 2]);
}

#[test]
fn n_smaller_than_total_returns_only_n_smallest() {
    let (slot, bytes) = f16_slot_with_values(&[10.0, 0.5, 100.0, 0.1, 5.0], 0);
    let map = build_map(vec![slot]);

    let ranked = lowest_magnitude_weights(&bytes, &map, 2);
    assert_eq!(ranked.len(), 2);
    let weight_indices: Vec<u64> = ranked.iter().map(|r| r.weight_index).collect();
    assert_eq!(weight_indices, vec![3, 1]); // 0.1 then 0.5
}

#[test]
fn ranking_combines_across_multiple_slots() {
    // F16 slot has weights with magnitudes [10.0, 0.5]; bytes start at offset 0.
    let (f16_slot, f16_bytes) = f16_slot_with_values(&[10.0, 0.5], 0);
    // Q8_0 slot (single block) has scale 1.0 and weights [3, -1, 0, ...] giving
    // magnitudes [3.0, 1.0, 0.0, 0.0, ...]; bytes start at offset 4 (after F16).
    let (q8_slot, q8_bytes) = q8_0_slot_with_values(&[3, -1, 0, 0, 0], 1.0, 4);

    let mut combined = f16_bytes.clone();
    combined.extend(q8_bytes);

    let map = build_map(vec![f16_slot, q8_slot]);

    let ranked = lowest_magnitude_weights(&combined, &map, 4);
    let resolved: Vec<(u32, u64, f32)> = ranked
        .iter()
        .map(|r| {
            let slot = &map.slots[r.slot_index as usize];
            (
                r.slot_index,
                r.weight_index,
                read_weight_abs(&combined, slot, r.weight_index),
            )
        })
        .collect();

    // Smallest four magnitudes across both slots: q8[2]=0, q8[3]=0, q8[4]=0, f16[1]=0.5
    // The three Q8_0 zeros are tied; tiebreak is WeightRef's lexicographic Ord
    // (slot_index then weight_index), which puts q8 indices in ascending order.
    assert_eq!(resolved.len(), 4);
    assert_eq!(resolved[0], (1, 2, 0.0));
    assert_eq!(resolved[1], (1, 3, 0.0));
    assert_eq!(resolved[2], (1, 4, 0.0));
    assert_eq!(resolved[3].2, 0.5);
    assert_eq!(resolved[3].0, 0);
    assert_eq!(resolved[3].1, 1);
}

#[test]
fn deterministic_across_repeated_calls() {
    let (slot, bytes) = f16_slot_with_values(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 0);
    let map = build_map(vec![slot]);

    let a = lowest_magnitude_weights(&bytes, &map, 5);
    let b = lowest_magnitude_weights(&bytes, &map, 5);
    let c = lowest_magnitude_weights(&bytes, &map, 5);
    assert_eq!(a, b);
    assert_eq!(b, c);
}

#[test]
fn lowest_for_bits_returns_enough_capacity() {
    // F16 = 4 bits per weight. Asking for 16 bits → 4 weights minimum.
    let values: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let (slot, bytes) = f16_slot_with_values(&values, 0);
    let map = build_map(vec![slot]);

    let chosen = lowest_magnitude_weights_for_bits(&bytes, &map, 16);
    let total_bits: u64 = chosen
        .iter()
        .map(|r| {
            llmdb::stego::calibration::stealable_bits_for(
                map.slots[r.slot_index as usize].quant_type,
            ) as u64
        })
        .sum();
    assert!(
        total_bits >= 16,
        "got only {total_bits} bits from {} weights, expected ≥ 16",
        chosen.len()
    );
    // The first 4 weights (values 0.0..3.0) cover the budget exactly at 16
    // bits with F16's 4 stealable bits per weight; we expect either 4 or
    // (worst-case heap rounding) 5 weights chosen.
    assert!(chosen.len() <= 5);
}

#[test]
fn lowest_for_bits_zero_request_returns_empty() {
    let (slot, bytes) = f16_slot_with_values(&[1.0, 2.0], 0);
    let map = build_map(vec![slot]);
    assert!(lowest_magnitude_weights_for_bits(&bytes, &map, 0).is_empty());
}

#[test]
fn ranks_q6_k_block_via_dispatch() {
    // Synthesize a Q6_K block where weight 0 has |value| 16.0 and
    // weight 96 has |value| 32.0. ranking the two should place weight 0
    // first.
    use llmdb::stego::packing::q6_k;
    let mut block = vec![0_u8; q6_k::BLOCK_BYTES];
    // d = 1.0 (0x3C00)
    block[208] = 0x00;
    block[209] = 0x3C;
    // scales: all 1
    for i in 0..16 {
        block[128 + 64 + i] = 1;
    }
    // qh byte 0 = 0b00000011 = 3 → quadrant 0 high2bits = 3, others 0
    // For weight 0 (q=0,l=0,quad=0): q6 = (0 | 0x30) - 32 = 16. value = 1*1*16 = 16
    // For weight 96 (q=3,l=0,quad=3): qh_2bits = (3 >> 6) & 3 = 0, q6 = 0 - 32 = -32. value = -32. |value| = 32.
    block[128] = 0x03;

    let weight_count = q6_k::WEIGHTS_PER_BLOCK as u64;
    let bits_per_weight = GgufQuantType::Q6K.stealable_bits_hint() as u64;
    let slot = TensorSlot {
        name: "test.q6_k".to_owned(),
        quant_type: GgufQuantType::Q6K,
        tier: TensorTier::Tier1,
        data_offset: 0,
        weight_count,
        stealable_bits_per_weight: bits_per_weight as usize,
        capacity_bits: weight_count * bits_per_weight,
        bit_start: 0,
        bit_end: weight_count * bits_per_weight,
    };
    let map = build_map(vec![slot]);

    // weight 0 has magnitude 16, weight 96 has magnitude 32; everywhere
    // else is also -32 (since qh=0). Lowest 2 should be weight 0 (16.0)
    // then any of the -32s, with WeightRef tiebreak putting the
    // smallest index first.
    let ranked = lowest_magnitude_weights(&block, &map, 2);
    assert_eq!(ranked.len(), 2);
    assert_eq!(ranked[0].weight_index, 0); // |16| smaller than |32|
    let mag0 = read_weight_abs(&block, &map.slots[0], ranked[0].weight_index);
    let mag1 = read_weight_abs(&block, &map.slots[0], ranked[1].weight_index);
    assert_eq!(mag0, 16.0);
    assert_eq!(mag1, 32.0);
}

#[test]
fn weight_ref_orders_lexicographically() {
    let a = WeightRef {
        slot_index: 0,
        weight_index: 5,
    };
    let b = WeightRef {
        slot_index: 1,
        weight_index: 0,
    };
    let c = WeightRef {
        slot_index: 0,
        weight_index: 6,
    };
    assert!(a < b); // slot 0 before slot 1
    assert!(a < c); // same slot, smaller weight first
    assert!(c < b);
}
