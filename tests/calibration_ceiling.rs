//! `read_weight_ceiling_abs` — per-quant-type ceiling magnitude.
//!
//! Ceiling magnitude: `max over stealable-bit values of |w|`. Invariant
//! under any write V2 makes, because stealable bits don't enter the
//! formula (they're maxed over). Used by Layer 0 anchor placement and
//! Layer 3 pristine allocation (DESIGN-NEW §15.2 and §15.5).
//!
//! The properties under test are per DESIGN-NEW §15.10:
//!
//! 1. **Invariance.** Writing every possible value into the stealable
//!    bits of a weight never changes that weight's ceiling magnitude.
//!    This is the load-bearing property: if it breaks, Layer 0's
//!    findability argument breaks.
//! 2. **Upper bound.** `ceiling(w) >= |w|` for any current bit state
//!    of the weight.
//! 3. **Tightness.** For at least one stealable-bit value, `|w| ==
//!    ceiling(w)`. (Otherwise we'd be picking a looser bound than
//!    necessary.)
//! 4. **Formula agreement** per quant type. The closed-form cases
//!    (F16, F32, Q8_0) match hand-computed values for edge cases.

use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::calibration::bit_io::write_bit;
use llmdb::stego::calibration::magnitude::{read_weight_abs, read_weight_ceiling_abs};
use llmdb::stego::calibration::placement::MetadataBitPos;
use llmdb::stego::calibration::stealable_bits_for;
use llmdb::stego::packing::{q3_k, q4_k, q5_k, q6_k};
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::TensorSlot;

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

fn slot_of(quant_type: GgufQuantType, weight_count: u64, data_offset: u64) -> TensorSlot {
    let bits = quant_type.stealable_bits_hint() as u64;
    TensorSlot {
        name: "ceiling.test".to_owned(),
        quant_type,
        tier: TensorTier::Tier1,
        data_offset,
        weight_count,
        stealable_bits_per_weight: bits as usize,
        capacity_bits: weight_count * bits,
        bit_start: 0,
        bit_end: weight_count * bits,
    }
}

/// Exhaustively set the stealable bits of `weight_index` to every
/// possible value and return the max observed `|w|`. This is the
/// oracle against which `read_weight_ceiling_abs` is compared.
fn brute_force_ceiling(mmap: &[u8], slot: &TensorSlot, weight_index: u64) -> f32 {
    let stealable = stealable_bits_for(slot.quant_type);
    if stealable == 0 {
        return 0.0;
    }
    let mut max_mag: f32 = 0.0;
    let variants = 1u32 << stealable;
    for v in 0..variants {
        let mut probe = mmap.to_vec();
        for bit_idx in 0..stealable {
            let pos = MetadataBitPos {
                slot_index: 0,
                weight_index,
                bit_index: bit_idx as u8,
            };
            write_bit(
                &mut probe,
                slot,
                pos,
                ((v >> bit_idx) & 1) == 1,
            );
        }
        let mag = read_weight_abs(&probe, slot, weight_index);
        if mag > max_mag {
            max_mag = mag;
        }
    }
    max_mag
}

// ------------------------------------------------------------------
// Invariance / upper-bound / tightness (property tests per quant type)
// ------------------------------------------------------------------

fn assert_invariance_and_tightness(mmap: &[u8], slot: &TensorSlot, weight_index: u64) {
    let stealable = stealable_bits_for(slot.quant_type);
    let variants = 1u32 << stealable;
    let expected = brute_force_ceiling(mmap, slot, weight_index);
    let mut saw_tightness = false;

    for v in 0..variants {
        let mut probe = mmap.to_vec();
        for bit_idx in 0..stealable {
            let pos = MetadataBitPos {
                slot_index: 0,
                weight_index,
                bit_index: bit_idx as u8,
            };
            write_bit(
                &mut probe,
                slot,
                pos,
                ((v >> bit_idx) & 1) == 1,
            );
        }
        // Invariance: ceiling is the same across all stealable-bit
        // configurations.
        let got = read_weight_ceiling_abs(&probe, slot, weight_index);
        assert!(
            (got - expected).abs() <= expected * 1e-6 + 1e-12,
            "invariance broken for {:?} weight {}: got ceiling {} with stealable={:b} but \
             expected {} (brute-force max)",
            slot.quant_type,
            weight_index,
            got,
            v,
            expected,
        );

        // Upper bound: ceiling >= current |w|.
        let current = read_weight_abs(&probe, slot, weight_index);
        assert!(
            got >= current - current * 1e-6 - 1e-12,
            "upper bound violated for {:?} weight {}: ceiling {} < |w| {}",
            slot.quant_type,
            weight_index,
            got,
            current,
        );

        // Tightness: at least one variant should hit equality.
        if (got - current).abs() <= got * 1e-6 + 1e-12 {
            saw_tightness = true;
        }
    }

    assert!(
        saw_tightness,
        "no stealable-bit configuration of {:?} weight {} reaches ceiling {} — \
         the ceiling is loose",
        slot.quant_type,
        weight_index,
        expected,
    );
}

// ------------------------------------------------------------------
// F16
// ------------------------------------------------------------------

#[test]
fn f16_ceiling_invariance_tightness() {
    // Mix of magnitudes (small, moderate, around 1, negative, zero).
    let values = [0.0_f32, 0.0001, 0.01, 0.5, -1.0, -0.001, 10.0];
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for v in &values {
        bytes.extend_from_slice(&f32_to_f16_bits(*v).to_le_bytes());
    }
    let slot = slot_of(GgufQuantType::F16, values.len() as u64, 0);
    for w in 0..values.len() as u64 {
        assert_invariance_and_tightness(&bytes, &slot, w);
    }
}

#[test]
fn f16_ceiling_matches_hand_computed_for_one_point_oh() {
    // F16 1.0 = 0x3C00. Mantissa = 0 (low nibble = 0). OR 0xF in low
    // nibble -> mantissa = 15. Decoded: 1.0 * (1 + 15/1024) ≈ 1.0146...
    let bytes = f32_to_f16_bits(1.0).to_le_bytes().to_vec();
    let slot = slot_of(GgufQuantType::F16, 1, 0);
    let ceiling = read_weight_ceiling_abs(&bytes, &slot, 0);
    let expected = 1.0 + 15.0 / 1024.0;
    assert!(
        (ceiling - expected).abs() < 1e-5,
        "f16 1.0 ceiling: got {}, expected ~{}",
        ceiling,
        expected,
    );
}

// ------------------------------------------------------------------
// F32
// ------------------------------------------------------------------

#[test]
fn f32_ceiling_invariance_tightness() {
    let values = [0.0_f32, 0.0001, 0.01, 0.5, -1.0, -0.001, 10.0];
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in &values {
        bytes.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    let slot = slot_of(GgufQuantType::F32, values.len() as u64, 0);
    for w in 0..values.len() as u64 {
        assert_invariance_and_tightness(&bytes, &slot, w);
    }
}

#[test]
fn f32_ceiling_matches_hand_computed_for_one_point_oh() {
    // f32 1.0 = 0x3F800000. Mantissa = 0 (low byte = 0). OR 0xFF:
    // decoded ~= 1.0 * (1 + 255 / 2^23).
    let bytes = 1.0_f32.to_bits().to_le_bytes().to_vec();
    let slot = slot_of(GgufQuantType::F32, 1, 0);
    let ceiling = read_weight_ceiling_abs(&bytes, &slot, 0);
    let expected = 1.0 + 255.0 / (1u32 << 23) as f32;
    assert!(
        (ceiling - expected).abs() < 1e-7,
        "f32 1.0 ceiling: got {}, expected ~{}",
        ceiling,
        expected,
    );
}

// ------------------------------------------------------------------
// Q8_0
// ------------------------------------------------------------------

fn q8_0_fixture_block(scale: f32, int8_values: &[i8]) -> Vec<u8> {
    assert!(int8_values.len() <= 32);
    let mut block = vec![0_u8; 34];
    block[0..2].copy_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
    for (i, v) in int8_values.iter().enumerate() {
        block[2 + i] = *v as u8;
    }
    block
}

#[test]
fn q8_0_ceiling_four_canonical_cases() {
    // From DESIGN-NEW §15.10:
    //   int8 = 0x08 (H=0):  max(0, 15) = 15
    //   int8 = 0xFF (H=-16): max(16, 1) = 16
    //   int8 = 0x80 (H=-128): max(128, 113) = 128
    //   int8 = 0x7F (H=112): max(112, 127) = 127
    let cases: &[(i8, u8)] = &[(0x08, 15), (-1, 16), (-128, 128), (0x7F, 127)];
    for &(val, expected_max) in cases {
        let block = q8_0_fixture_block(1.0, &[val]);
        let slot = slot_of(GgufQuantType::Q8_0, 1, 0);
        let ceiling = read_weight_ceiling_abs(&block, &slot, 0);
        assert!(
            (ceiling - expected_max as f32).abs() < 1e-5,
            "q8_0 int8={}: got ceiling {}, expected {}",
            val,
            ceiling,
            expected_max,
        );
    }
}

#[test]
fn q8_0_ceiling_scales_with_block_scale() {
    // scale = 0.5, int8 = 0x08 → H = 0 → max(0, 15) * 0.5 = 7.5.
    let block = q8_0_fixture_block(0.5, &[0x08]);
    let slot = slot_of(GgufQuantType::Q8_0, 1, 0);
    let ceiling = read_weight_ceiling_abs(&block, &slot, 0);
    assert!(
        (ceiling - 7.5).abs() < 1e-5,
        "q8_0 scale=0.5 int8=8: got {} expected 7.5",
        ceiling,
    );
}

#[test]
fn q8_0_ceiling_invariance_tightness() {
    // Multiple int8 values with non-unit scale to stress both the
    // high-nibble handling and the scale multiplication.
    let ints: Vec<i8> = (0..32).map(|i| ((i * 17 + 3) as i8).wrapping_sub(64)).collect();
    let block = q8_0_fixture_block(0.1, &ints);
    let slot = slot_of(GgufQuantType::Q8_0, 32, 0);
    for w in 0..32 {
        assert_invariance_and_tightness(&block, &slot, w);
    }
}

// ------------------------------------------------------------------
// K-quants — invariance + tightness via brute force
// ------------------------------------------------------------------

#[test]
fn q3_k_ceiling_invariance_tightness() {
    // Zeroed Q3_K block with non-zero d so weights have sensible
    // magnitude. Brute-force across stealable-bit configurations.
    let mut block = vec![0_u8; q3_k::BLOCK_BYTES];
    // Set d = 1.0 (fp16)
    block[q3_k::D_OFFSET..q3_k::D_OFFSET + 2]
        .copy_from_slice(&f32_to_f16_bits(1.0).to_le_bytes());
    // Set a few scales so weights aren't all zero-ceiling
    for i in 0..q3_k::SCALES_BYTES {
        block[q3_k::SCALES_OFFSET + i] = (i as u8).wrapping_add(5);
    }
    // Set some qs bytes
    for i in 0..16 {
        block[q3_k::QS_OFFSET + i] = 0xAA; // alternating bits in qs
    }
    let slot = slot_of(GgufQuantType::Q3K, q3_k::WEIGHTS_PER_BLOCK as u64, 0);
    // Sample a few weight indices across quadrants
    for w in [0_u64, 1, 31, 32, 64, 127, 128, 255] {
        assert_invariance_and_tightness(&block, &slot, w);
    }
}

#[test]
fn q4_k_ceiling_invariance_tightness() {
    let mut block = vec![0_u8; q4_k::BLOCK_BYTES];
    block[0..2].copy_from_slice(&f32_to_f16_bits(1.0).to_le_bytes()); // d
    block[2..4].copy_from_slice(&f32_to_f16_bits(0.1).to_le_bytes()); // dmin
    // Set scales + qs with some pattern.
    for i in 0..12 {
        block[4 + i] = (i as u8).wrapping_add(3);
    }
    for i in 0..128 {
        block[16 + i] = ((i * 7) as u8).wrapping_add(0x55);
    }
    let slot = slot_of(GgufQuantType::Q4K, q4_k::WEIGHTS_PER_BLOCK as u64, 0);
    for w in [0_u64, 1, 31, 32, 63, 128, 200, 255] {
        assert_invariance_and_tightness(&block, &slot, w);
    }
}

#[test]
fn q5_k_ceiling_invariance_tightness() {
    let mut block = vec![0_u8; q5_k::BLOCK_BYTES];
    block[0..2].copy_from_slice(&f32_to_f16_bits(1.0).to_le_bytes()); // d
    block[2..4].copy_from_slice(&f32_to_f16_bits(0.1).to_le_bytes()); // dmin
    for i in 0..12 {
        block[4 + i] = (i as u8).wrapping_add(3);
    }
    // qh[32] at offset 16, qs[128] at offset 48
    for i in 0..32 {
        block[16 + i] = (i as u8).wrapping_add(0x11);
    }
    for i in 0..128 {
        block[48 + i] = ((i * 11) as u8).wrapping_add(0x33);
    }
    let slot = slot_of(GgufQuantType::Q5K, q5_k::WEIGHTS_PER_BLOCK as u64, 0);
    for w in [0_u64, 1, 31, 32, 128, 255] {
        assert_invariance_and_tightness(&block, &slot, w);
    }
}

#[test]
fn q6_k_ceiling_invariance_tightness() {
    let mut block = vec![0_u8; q6_k::BLOCK_BYTES];
    // d at offset 208 (per q6_k layout: ql[128], qh[64], scales[16], d[2])
    block[208..210].copy_from_slice(&f32_to_f16_bits(1.0).to_le_bytes());
    // scales at offset 192
    for i in 0..16 {
        block[192 + i] = (i as u8).wrapping_add(2);
    }
    // ql[128] with pattern
    for (i, byte) in block.iter_mut().take(128).enumerate() {
        *byte = ((i * 5) as u8).wrapping_add(0x44);
    }
    // qh[64] with pattern
    for i in 0..64 {
        block[128 + i] = ((i * 3) as u8).wrapping_add(0x22);
    }
    let slot = slot_of(GgufQuantType::Q6K, q6_k::WEIGHTS_PER_BLOCK as u64, 0);
    for w in [0_u64, 1, 31, 32, 128, 200, 255] {
        assert_invariance_and_tightness(&block, &slot, w);
    }
}

// ------------------------------------------------------------------
// Stub quant types return 0.0 (same convention as read_weight_abs)
// ------------------------------------------------------------------

#[test]
fn stub_quant_types_return_zero_ceiling() {
    // Stubs have stealable_bits_hint == 0; ceiling returns 0.
    let stubs = [
        GgufQuantType::Q2K,
        GgufQuantType::Q4_0,
        GgufQuantType::Q4_1,
        GgufQuantType::Q5_0,
        GgufQuantType::Q5_1,
        GgufQuantType::Q8_1,
        GgufQuantType::Q8K,
    ];
    for t in stubs {
        let slot = slot_of(t, 1, 0);
        let bytes = vec![0_u8; 1024]; // arbitrary
        assert_eq!(
            read_weight_ceiling_abs(&bytes, &slot, 0),
            0.0,
            "expected stub {:?} to return 0.0 ceiling",
            t,
        );
    }
}
