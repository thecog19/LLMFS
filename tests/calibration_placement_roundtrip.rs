//! Placement → bit_io round-trip: the narrow contract between V2
//! calibration and the eventual Layer-0 metadata writer.
//!
//! Two properties under test:
//!   1. **Reversible.** A pattern written bit-by-bit via `bit_io::write_bit`
//!      at positions produced by `compute_metadata_placement` reads back
//!      bit-for-bit via `bit_io::read_bit`.
//!   2. **Non-aliasing.** Writing one metadata bit must not flip any
//!      other metadata bit in the same placement. Each `MetadataBitPos`
//!      targets a distinct physical bit; a bug that collapsed
//!      `(slot, weight, bit_index)` tuples to the same byte+mask would
//!      silently corrupt metadata.
//!
//! If these tests go red, something in `bit_io::locate` disagrees with
//! the per-packer bit numbering — either the packer changed, or a new
//! quant type was added in one place without its matching `locate`
//! arm. Start by diffing `bit_io.rs` against the packer whose `extract`
//! lines up with the failing type.
//!
//! Covers F16, F32, Q8_0, Q4_K, Q5_K, Q6_K — the intersection of
//! "magnitude dispatch decodes it" and "packer declares a nonzero
//! stealable-bit budget".

use std::collections::HashMap;

use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::calibration::bit_io::{read_bit, write_bit};
use llmdb::stego::calibration::placement::{MetadataBitPos, compute_metadata_placement};
use llmdb::stego::calibration::stealable_bits_for;
use llmdb::stego::packing::{q4_k, q5_k, q6_k};
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};

/// Encode an f32 as fp16 bits. Good enough for fixtures; no rounding
/// or subnormal handling. Copied from tests/calibration_magnitude.rs
/// to keep this file self-contained — if the helpers drift we'd
/// rather the drift stay visible in individual tests.
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

fn f32_slot_with_values(values: &[f32], data_offset: u64) -> (TensorSlot, Vec<u8>) {
    let mut bytes = vec![0_u8; values.len() * 4];
    for (i, v) in values.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_bits().to_le_bytes());
    }
    let weight_count = values.len() as u64;
    let bits_per_weight = GgufQuantType::F32.stealable_bits_hint() as u64;
    let slot = TensorSlot {
        name: "test.f32".to_owned(),
        quant_type: GgufQuantType::F32,
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

fn q8_0_slot_block(values: &[i8], scale: f32, data_offset: u64) -> (TensorSlot, Vec<u8>) {
    assert!(values.len() <= 32, "single Q8_0 block fixture");
    let mut block = vec![0_u8; 34];
    block[0..2].copy_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
    for (i, v) in values.iter().enumerate() {
        block[2 + i] = *v as u8;
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

fn q_k_like_slot(
    quant_type: GgufQuantType,
    weights_per_block: u64,
    block_bytes: usize,
    data_offset: u64,
) -> (TensorSlot, Vec<u8>) {
    // One block, all zero (valid if the packer parses it — they all
    // tolerate zeroed scales/qs since their reads just recover 0.0).
    let bytes = vec![0_u8; block_bytes];
    let bits_per_weight = quant_type.stealable_bits_hint() as u64;
    let slot = TensorSlot {
        name: format!("test.{quant_type:?}").to_lowercase(),
        quant_type,
        tier: TensorTier::Tier1,
        data_offset,
        weight_count: weights_per_block,
        stealable_bits_per_weight: bits_per_weight as usize,
        capacity_bits: weights_per_block * bits_per_weight,
        bit_start: 0,
        bit_end: weights_per_block * bits_per_weight,
    };
    (slot, bytes)
}

/// Single-slot TensorMap from an already-populated slot; recomputes
/// bit_start / bit_end / totals.
fn single_slot_map(mut slot: TensorSlot) -> TensorMap {
    slot.bit_start = 0;
    slot.bit_end = slot.capacity_bits;
    let total = slot.capacity_bits;
    TensorMap {
        slots: vec![slot],
        total_capacity_bits: total,
        total_capacity_bytes: total / 8,
    }
}

fn multi_slot_map(slots: Vec<TensorSlot>) -> TensorMap {
    let mut cursor = 0_u64;
    let mut adjusted = Vec::with_capacity(slots.len());
    for mut s in slots {
        s.bit_start = cursor;
        cursor += s.capacity_bits;
        s.bit_end = cursor;
        adjusted.push(s);
    }
    TensorMap {
        slots: adjusted,
        total_capacity_bits: cursor,
        total_capacity_bytes: cursor / 8,
    }
}

/// Walk every stealable `MetadataBitPos` reachable in the map. Used
/// by the non-aliasing tests to check that a write at one position
/// only flips that one position.
fn enumerate_all_positions(map: &TensorMap) -> Vec<MetadataBitPos> {
    let mut out = Vec::new();
    for (slot_idx, slot) in map.slots.iter().enumerate() {
        let bits = stealable_bits_for(slot.quant_type);
        for weight_index in 0..slot.weight_count {
            for bit_index in 0..bits {
                out.push(MetadataBitPos {
                    slot_index: slot_idx as u32,
                    weight_index,
                    bit_index: bit_index as u8,
                });
            }
        }
    }
    out
}

/// Write-then-read returns the value written. Asserted for all four
/// possible transitions (current=0/1, target=0/1) so the test catches
/// both the OR-set and AND-clear branches of `write_bit`.
fn assert_single_bit_roundtrip(mmap: &mut [u8], slot: &TensorSlot, pos: MetadataBitPos) {
    for target in [false, true, false, true] {
        write_bit(mmap, slot, pos, target);
        let got = read_bit(mmap, slot, pos);
        assert_eq!(got, target, "roundtrip mismatch at {pos:?} target={target}");
    }
}

#[test]
fn single_bit_roundtrip_f16() {
    let (slot, mut bytes) = f16_slot_with_values(&[0.1, 0.2, 0.3, 0.4], 0);
    let map = single_slot_map(slot.clone());
    // All 4 stealable bits of each weight, for 4 weights = 16 positions.
    let positions = enumerate_all_positions(&map);
    assert_eq!(positions.len(), 16);
    for pos in positions {
        assert_single_bit_roundtrip(&mut bytes, &slot, pos);
    }
}

#[test]
fn single_bit_roundtrip_f32() {
    let (slot, mut bytes) = f32_slot_with_values(&[0.1, 0.2, 0.3], 0);
    let map = single_slot_map(slot.clone());
    // 3 weights × 8 bits = 24 positions.
    let positions = enumerate_all_positions(&map);
    assert_eq!(positions.len(), 24);
    for pos in positions {
        assert_single_bit_roundtrip(&mut bytes, &slot, pos);
    }
}

#[test]
fn single_bit_roundtrip_q8_0() {
    let (slot, mut bytes) = q8_0_slot_block(&[3, -7, 0, 15, -1], 1.0, 0);
    let map = single_slot_map(slot.clone());
    // 5 weights × 4 bits = 20 positions.
    let positions = enumerate_all_positions(&map);
    assert_eq!(positions.len(), 20);
    for pos in positions {
        assert_single_bit_roundtrip(&mut bytes, &slot, pos);
    }
}

#[test]
fn single_bit_roundtrip_q4_k() {
    let (slot, mut bytes) = q_k_like_slot(
        GgufQuantType::Q4K,
        q4_k::WEIGHTS_PER_BLOCK as u64,
        q4_k::BLOCK_BYTES,
        0,
    );
    let map = single_slot_map(slot.clone());
    // 256 weights × 1 bit = 256 positions.
    let positions = enumerate_all_positions(&map);
    assert_eq!(positions.len(), 256);
    for pos in positions {
        assert_single_bit_roundtrip(&mut bytes, &slot, pos);
    }
}

#[test]
fn single_bit_roundtrip_q5_k() {
    let (slot, mut bytes) = q_k_like_slot(
        GgufQuantType::Q5K,
        q5_k::WEIGHTS_PER_BLOCK as u64,
        q5_k::BLOCK_BYTES,
        0,
    );
    let map = single_slot_map(slot.clone());
    let positions = enumerate_all_positions(&map);
    assert_eq!(positions.len(), 256);
    for pos in positions {
        assert_single_bit_roundtrip(&mut bytes, &slot, pos);
    }
}

#[test]
fn single_bit_roundtrip_q6_k() {
    let (slot, mut bytes) = q_k_like_slot(
        GgufQuantType::Q6K,
        q6_k::WEIGHTS_PER_BLOCK as u64,
        q6_k::BLOCK_BYTES,
        0,
    );
    let map = single_slot_map(slot.clone());
    // 256 weights × 2 bits = 512 positions.
    let positions = enumerate_all_positions(&map);
    assert_eq!(positions.len(), 512);
    for pos in positions {
        assert_single_bit_roundtrip(&mut bytes, &slot, pos);
    }
}

#[test]
fn write_does_not_alias_other_positions_f16() {
    // A distinct physical bit per `MetadataBitPos` is the load-bearing
    // invariant that lets placement.rs iterate bit by bit without
    // worrying about collisions. Exhaustively: flip each position
    // and verify no *other* position changed. Enumerated because any
    // collision is a footgun — random sampling might not hit it.
    let (slot, initial) = f16_slot_with_values(&[0.0, 0.0, 0.0, 0.0], 0);
    let map = single_slot_map(slot.clone());
    let positions = enumerate_all_positions(&map);
    for target in &positions {
        let mut bytes = initial.clone();
        write_bit(&mut bytes, &slot, *target, true);
        for other in &positions {
            let expected = other == target;
            let got = read_bit(&bytes, &slot, *other);
            assert_eq!(
                got, expected,
                "write at {target:?} leaked into {other:?} (expected {expected}, got {got})",
            );
        }
    }
}

#[test]
fn write_does_not_alias_other_positions_q4_k() {
    // Q4_K's 256 positions share 128 carrier bytes; verify the
    // low-half-bit-0 / high-half-bit-4 split doesn't alias. This is
    // the test I wish I'd had when I first transcribed the packer —
    // it took two debugging hours to notice I'd used `half*4` as the
    // bit position (which happens to work for half ∈ {0,1} and would
    // have broken silently if `half` grew to 2).
    let (slot, initial) = q_k_like_slot(
        GgufQuantType::Q4K,
        q4_k::WEIGHTS_PER_BLOCK as u64,
        q4_k::BLOCK_BYTES,
        0,
    );
    let map = single_slot_map(slot.clone());
    let positions = enumerate_all_positions(&map);
    // Pairs of positions known to share a carrier byte: weight N
    // (low nibble) and weight N+32 (high nibble) for N in 0..32, plus
    // similar for j_outer > 0. Spot-check one pair per outer group;
    // the exhaustive pass below covers the rest.
    for (a_idx, a) in positions.iter().enumerate() {
        let mut bytes = initial.clone();
        write_bit(&mut bytes, &slot, *a, true);
        for (b_idx, b) in positions.iter().enumerate() {
            let expected = a_idx == b_idx;
            let got = read_bit(&bytes, &slot, *b);
            assert_eq!(got, expected, "q4_k aliasing: write {a:?} flipped {b:?}");
        }
    }
}

#[test]
fn write_does_not_alias_other_positions_q6_k() {
    // Q6_K has 2 stealable bits per weight. Verify adjacent
    // bit_index=0 and bit_index=1 on the same weight are independent,
    // and that high-nibble weights don't alias low-nibble weights
    // they share a carrier byte with.
    let (slot, initial) = q_k_like_slot(
        GgufQuantType::Q6K,
        q6_k::WEIGHTS_PER_BLOCK as u64,
        q6_k::BLOCK_BYTES,
        0,
    );
    let map = single_slot_map(slot.clone());
    let positions = enumerate_all_positions(&map);
    for (a_idx, a) in positions.iter().enumerate() {
        let mut bytes = initial.clone();
        write_bit(&mut bytes, &slot, *a, true);
        for (b_idx, b) in positions.iter().enumerate() {
            let expected = a_idx == b_idx;
            let got = read_bit(&bytes, &slot, *b);
            assert_eq!(got, expected, "q6_k aliasing: write {a:?} flipped {b:?}");
        }
    }
}

#[test]
fn placement_roundtrip_preserves_pattern_f16() {
    // Compute a placement for a specific bit count, then write a
    // known pattern bit-by-bit through those positions and verify the
    // readback. This exercises the full calibration→bit_io pipeline.
    let (slot, mut bytes) =
        f16_slot_with_values(&[0.1, 0.01, 100.0, 0.001, 50.0, 0.0001, 25.0, 0.00001], 0);
    let map = single_slot_map(slot);
    let placement = compute_metadata_placement(&bytes, &map, 20);
    assert!(placement.positions.len() >= 20);

    // Pattern: alternating then all-ones in groups of 3, to hit
    // every transition combination. Fixed so failures are reproducible.
    let pattern: Vec<bool> = (0..placement.positions.len())
        .map(|i| matches!(i % 5, 0 | 2 | 3))
        .collect();

    for (bit, pos) in pattern.iter().zip(&placement.positions) {
        write_bit(&mut bytes, &map.slots[pos.slot_index as usize], *pos, *bit);
    }
    for (bit, pos) in pattern.iter().zip(&placement.positions) {
        let got = read_bit(&bytes, &map.slots[pos.slot_index as usize], *pos);
        assert_eq!(got, *bit, "pattern mismatch at {pos:?}");
    }
}

#[test]
fn placement_roundtrip_across_multiple_slots() {
    // Two slots of different quant types side by side. The placement
    // should address both transparently. Writing bit N then reading
    // bit N must return the same value regardless of which slot it
    // lives in.
    let (mut f16_slot, f16_bytes) = f16_slot_with_values(&[0.01, 100.0, 0.001, 50.0], 0);
    let (mut q8_slot, q8_bytes) =
        q8_0_slot_block(&[1, -3, 2, -4, 0, 0, 0, 0], 0.25, f16_bytes.len() as u64);
    f16_slot.data_offset = 0;
    q8_slot.data_offset = f16_bytes.len() as u64;

    let mut combined = f16_bytes;
    combined.extend(q8_bytes);

    let map = multi_slot_map(vec![f16_slot, q8_slot]);
    // F16 = 4 weights × 4 bits = 16 bits; Q8_0 = 8 weights × 4 bits = 32.
    // 24 bits samples across both slots.
    let placement = compute_metadata_placement(&combined, &map, 24);
    assert_eq!(placement.positions.len(), 24);
    let spanning_slots: std::collections::HashSet<u32> =
        placement.positions.iter().map(|p| p.slot_index).collect();
    assert!(
        spanning_slots.len() >= 2,
        "placement should span both slots — got positions only in {spanning_slots:?}",
    );

    let pattern: Vec<bool> = (0..placement.positions.len()).map(|i| i % 3 == 1).collect();
    for (bit, pos) in pattern.iter().zip(&placement.positions) {
        write_bit(
            &mut combined,
            &map.slots[pos.slot_index as usize],
            *pos,
            *bit,
        );
    }
    for (bit, pos) in pattern.iter().zip(&placement.positions) {
        let got = read_bit(&combined, &map.slots[pos.slot_index as usize], *pos);
        assert_eq!(got, *bit, "multi-slot readback mismatch at {pos:?}");
    }
}

#[test]
fn placement_writes_are_disjoint_across_positions() {
    // Track, per byte-offset, which bits are flipped by each position.
    // Every position must own a distinct (byte_offset, bit_mask) —
    // otherwise two metadata bits would corrupt each other at write
    // time. Detects a packer-convention drift that would alias two
    // different `MetadataBitPos` onto the same physical bit.
    let (slot, zero_bytes) = q_k_like_slot(
        GgufQuantType::Q6K,
        q6_k::WEIGHTS_PER_BLOCK as u64,
        q6_k::BLOCK_BYTES,
        0,
    );
    let map = single_slot_map(slot.clone());

    let mut observed: HashMap<(usize, u8), MetadataBitPos> = HashMap::new();
    let positions = enumerate_all_positions(&map);

    for pos in &positions {
        let mut bytes = zero_bytes.clone();
        write_bit(&mut bytes, &slot, *pos, true);
        // Find the unique (offset, bit) that changed.
        let mut diffs = Vec::new();
        for (i, (a, b)) in zero_bytes.iter().zip(bytes.iter()).enumerate() {
            if a != b {
                let mask = a ^ b;
                // Exactly one bit must differ.
                assert!(
                    mask.count_ones() == 1,
                    "write_bit at {pos:?} flipped {} bits in byte {i} (mask 0x{mask:02x})",
                    mask.count_ones()
                );
                diffs.push((i, mask.trailing_zeros() as u8));
            }
        }
        assert_eq!(
            diffs.len(),
            1,
            "write_bit at {pos:?} changed {} bytes",
            diffs.len()
        );
        let site = diffs[0];
        if let Some(prev) = observed.insert(site, *pos) {
            panic!(
                "two MetadataBitPos collide on byte offset {} bit {}: {:?} and {:?}",
                site.0, site.1, prev, pos,
            );
        }
    }

    // Sanity: we observed exactly one (offset, bit) per position.
    assert_eq!(observed.len(), positions.len());
}
