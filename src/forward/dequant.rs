//! Inference-time dequantization: packed cover bytes → `f32` weights.
//!
//! Sibling to `src/stego/packing/`. The existing packing modules know
//! how to *embed stealable bits* into each quant block and how to
//! compute *ceiling magnitudes* for anchor placement; they don't
//! concern themselves with reconstructing the f32 weight the model
//! multiplies against.
//!
//! This module is that reconstruction. Covers seven quant types:
//!
//! - **F32, F16** (Milestone A) — byte-wise little-endian load.
//! - **Q8_0** (Milestone A) — per-32-weight block with f16 scale × int8.
//! - **Q3_K, Q4_K, Q5_K, Q6_K** (Milestone C) — 256-weight super-
//!   blocks with per-sub-block scales + packed quants. Dequant here
//!   delegates to each packing module's `read_weight_value`, which
//!   is the same primitive the magnitude estimator trusts. That
//!   path is validated against ggml-encoded fixtures by
//!   `tests/kquant_ggml_reference.rs`, so calling it in a loop
//!   inherits the correctness proof.
//!
//! Q2_K is defined in the enum but has no packing module; it
//! returns [`DequantError::Unsupported`] until we land a decoder.
//!
//! # Block layout recap
//!
//! ```text
//! F32   4 bytes / weight, little-endian IEEE-754 single.
//! F16   2 bytes / weight, little-endian IEEE-754 half.
//! Q8_0  34 bytes / 32 weights.   block = [scale:f16][q:int8×32].
//! Q3_K  110 bytes / 256 weights.
//! Q4_K  144 bytes / 256 weights.
//! Q5_K  176 bytes / 256 weights.
//! Q6_K  210 bytes / 256 weights.
//! ```

use thiserror::Error;

use crate::gguf::quant::GgufQuantType;
use crate::stego::packing::float::f16_to_f32;
use crate::stego::packing::{q3_k, q4_k, q5_k, q6_k};

/// Bytes per weight for F16.
pub const F16_BYTES_PER_WEIGHT: usize = 2;
/// Bytes per weight for F32.
pub const F32_BYTES_PER_WEIGHT: usize = 4;
/// Weights per Q8_0 block.
pub const Q8_0_WEIGHTS_PER_BLOCK: usize = 32;
/// Bytes per Q8_0 block: 2-byte f16 scale + 32 int8 weights.
pub const Q8_0_BYTES_PER_BLOCK: usize = 2 + Q8_0_WEIGHTS_PER_BLOCK;
/// Weights per K-quant super-block. Same for Q3_K/Q4_K/Q5_K/Q6_K.
pub const K_QUANT_WEIGHTS_PER_BLOCK: usize = 256;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DequantError {
    #[error("{quant:?}: unsupported in Milestone A — see `crate::forward::dequant` module docs")]
    Unsupported { quant: GgufQuantType },

    #[error("{quant:?}: source length {src_len} is not a multiple of the {unit}-byte unit")]
    SourceLengthMisaligned {
        quant: GgufQuantType,
        src_len: usize,
        unit: usize,
    },

    #[error("{quant:?}: destination holds {dst_weights} weights, source encodes {src_weights}")]
    DestinationLengthMismatch {
        quant: GgufQuantType,
        dst_weights: usize,
        src_weights: usize,
    },
}

/// How many f32 weights `src.len()` bytes of `quant` decodes to.
/// Returns `None` for quant types this module doesn't handle (Q2_K,
/// older non-K quants beyond Q8_0) or for byte lengths that aren't
/// a whole number of blocks.
pub fn weight_count(quant: GgufQuantType, src_len: usize) -> Option<usize> {
    match quant {
        GgufQuantType::F32 => src_len
            .is_multiple_of(F32_BYTES_PER_WEIGHT)
            .then_some(src_len / F32_BYTES_PER_WEIGHT),
        GgufQuantType::F16 => src_len
            .is_multiple_of(F16_BYTES_PER_WEIGHT)
            .then_some(src_len / F16_BYTES_PER_WEIGHT),
        GgufQuantType::Q8_0 => src_len
            .is_multiple_of(Q8_0_BYTES_PER_BLOCK)
            .then_some((src_len / Q8_0_BYTES_PER_BLOCK) * Q8_0_WEIGHTS_PER_BLOCK),
        GgufQuantType::Q3K => k_quant_weight_count(src_len, q3_k::BLOCK_BYTES),
        GgufQuantType::Q4K => k_quant_weight_count(src_len, q4_k::BLOCK_BYTES),
        GgufQuantType::Q5K => k_quant_weight_count(src_len, q5_k::BLOCK_BYTES),
        GgufQuantType::Q6K => k_quant_weight_count(src_len, q6_k::BLOCK_BYTES),
        _ => None,
    }
}

fn k_quant_weight_count(src_len: usize, block_bytes: usize) -> Option<usize> {
    src_len
        .is_multiple_of(block_bytes)
        .then_some((src_len / block_bytes) * K_QUANT_WEIGHTS_PER_BLOCK)
}

/// Dequantize a row of packed bytes into the provided destination
/// buffer. `dst.len()` must match the number of weights implied by
/// `src.len()` for the given quant type.
///
/// Reuse across calls: the caller can preallocate `dst` once per
/// row shape and reuse the buffer. No heap allocation per call.
pub fn dequantize_row_into(
    quant: GgufQuantType,
    src: &[u8],
    dst: &mut [f32],
) -> Result<(), DequantError> {
    match quant {
        GgufQuantType::F32 => dequant_f32(src, dst),
        GgufQuantType::F16 => dequant_f16(src, dst),
        GgufQuantType::Q8_0 => dequant_q8_0(src, dst),
        GgufQuantType::Q3K => dequant_k_quant(
            quant,
            src,
            dst,
            q3_k::BLOCK_BYTES,
            q3_k::read_weight_value,
        ),
        GgufQuantType::Q4K => dequant_k_quant(
            quant,
            src,
            dst,
            q4_k::BLOCK_BYTES,
            q4_k::read_weight_value,
        ),
        GgufQuantType::Q5K => dequant_k_quant(
            quant,
            src,
            dst,
            q5_k::BLOCK_BYTES,
            q5_k::read_weight_value,
        ),
        GgufQuantType::Q6K => dequant_k_quant(
            quant,
            src,
            dst,
            q6_k::BLOCK_BYTES,
            q6_k::read_weight_value,
        ),
        other => Err(DequantError::Unsupported { quant: other }),
    }
}

/// Allocate-on-demand variant. Use this in code paths where the
/// weight count isn't known up front; prefer the `_into` form in
/// hot loops.
pub fn dequantize_row(quant: GgufQuantType, src: &[u8]) -> Result<Vec<f32>, DequantError> {
    // Reject unsupported quant types before touching `src.len()`,
    // so callers get `Unsupported` for types we don't handle and
    // `SourceLengthMisaligned` for supported types with bad lengths.
    let unit = match quant {
        GgufQuantType::F32 => F32_BYTES_PER_WEIGHT,
        GgufQuantType::F16 => F16_BYTES_PER_WEIGHT,
        GgufQuantType::Q8_0 => Q8_0_BYTES_PER_BLOCK,
        GgufQuantType::Q3K => q3_k::BLOCK_BYTES,
        GgufQuantType::Q4K => q4_k::BLOCK_BYTES,
        GgufQuantType::Q5K => q5_k::BLOCK_BYTES,
        GgufQuantType::Q6K => q6_k::BLOCK_BYTES,
        other => return Err(DequantError::Unsupported { quant: other }),
    };
    if !src.len().is_multiple_of(unit) {
        return Err(DequantError::SourceLengthMisaligned {
            quant,
            src_len: src.len(),
            unit,
        });
    }
    let n = weight_count(quant, src.len())
        .expect("supported + aligned source always has a weight count");
    let mut dst = vec![0.0_f32; n];
    dequantize_row_into(quant, src, &mut dst)?;
    Ok(dst)
}

// ─── F32 ──────────────────────────────────────────────────────────────────

fn dequant_f32(src: &[u8], dst: &mut [f32]) -> Result<(), DequantError> {
    if !src.len().is_multiple_of(F32_BYTES_PER_WEIGHT) {
        return Err(DequantError::SourceLengthMisaligned {
            quant: GgufQuantType::F32,
            src_len: src.len(),
            unit: F32_BYTES_PER_WEIGHT,
        });
    }
    let n = src.len() / F32_BYTES_PER_WEIGHT;
    if dst.len() != n {
        return Err(DequantError::DestinationLengthMismatch {
            quant: GgufQuantType::F32,
            dst_weights: dst.len(),
            src_weights: n,
        });
    }
    for (i, chunk) in src.chunks_exact(F32_BYTES_PER_WEIGHT).enumerate() {
        let b = [chunk[0], chunk[1], chunk[2], chunk[3]];
        dst[i] = f32::from_le_bytes(b);
    }
    Ok(())
}

// ─── F16 ──────────────────────────────────────────────────────────────────

fn dequant_f16(src: &[u8], dst: &mut [f32]) -> Result<(), DequantError> {
    if !src.len().is_multiple_of(F16_BYTES_PER_WEIGHT) {
        return Err(DequantError::SourceLengthMisaligned {
            quant: GgufQuantType::F16,
            src_len: src.len(),
            unit: F16_BYTES_PER_WEIGHT,
        });
    }
    let n = src.len() / F16_BYTES_PER_WEIGHT;
    if dst.len() != n {
        return Err(DequantError::DestinationLengthMismatch {
            quant: GgufQuantType::F16,
            dst_weights: dst.len(),
            src_weights: n,
        });
    }
    for (i, chunk) in src.chunks_exact(F16_BYTES_PER_WEIGHT).enumerate() {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        dst[i] = f16_to_f32(bits);
    }
    Ok(())
}

// ─── Q8_0 ─────────────────────────────────────────────────────────────────

fn dequant_q8_0(src: &[u8], dst: &mut [f32]) -> Result<(), DequantError> {
    if !src.len().is_multiple_of(Q8_0_BYTES_PER_BLOCK) {
        return Err(DequantError::SourceLengthMisaligned {
            quant: GgufQuantType::Q8_0,
            src_len: src.len(),
            unit: Q8_0_BYTES_PER_BLOCK,
        });
    }
    let block_count = src.len() / Q8_0_BYTES_PER_BLOCK;
    let n = block_count * Q8_0_WEIGHTS_PER_BLOCK;
    if dst.len() != n {
        return Err(DequantError::DestinationLengthMismatch {
            quant: GgufQuantType::Q8_0,
            dst_weights: dst.len(),
            src_weights: n,
        });
    }
    for (block_idx, block) in src.chunks_exact(Q8_0_BYTES_PER_BLOCK).enumerate() {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let weights = &block[2..];
        let dst_base = block_idx * Q8_0_WEIGHTS_PER_BLOCK;
        for (i, &q) in weights.iter().enumerate() {
            // Interpret the byte as a signed int8 (two's complement).
            let signed = q as i8;
            dst[dst_base + i] = scale * signed as f32;
        }
    }
    Ok(())
}

// ─── K-quants (Q3_K / Q4_K / Q5_K / Q6_K) ────────────────────────────────

/// Generic K-quant dequant. Delegates to a per-quant
/// `read_weight_value(block, weight_idx)` primitive (the same one
/// `src/stego/calibration/magnitude.rs` uses, validated against
/// ggml-encoded fixtures in `tests/kquant_ggml_reference.rs`).
fn dequant_k_quant(
    quant: GgufQuantType,
    src: &[u8],
    dst: &mut [f32],
    block_bytes: usize,
    read: fn(&[u8], usize) -> Result<f32, crate::stego::packing::PackingError>,
) -> Result<(), DequantError> {
    if !src.len().is_multiple_of(block_bytes) {
        return Err(DequantError::SourceLengthMisaligned {
            quant,
            src_len: src.len(),
            unit: block_bytes,
        });
    }
    let block_count = src.len() / block_bytes;
    let n = block_count * K_QUANT_WEIGHTS_PER_BLOCK;
    if dst.len() != n {
        return Err(DequantError::DestinationLengthMismatch {
            quant,
            dst_weights: dst.len(),
            src_weights: n,
        });
    }
    for (block_idx, block) in src.chunks_exact(block_bytes).enumerate() {
        let dst_base = block_idx * K_QUANT_WEIGHTS_PER_BLOCK;
        for i in 0..K_QUANT_WEIGHTS_PER_BLOCK {
            // `read` can only fail on block-length or weight-index
            // errors, both ruled out by the block_bytes / iteration
            // bounds above. The expect is a sanity tripwire — a
            // regression in the packing module would surface here.
            dst[dst_base + i] = read(block, i).expect("read_weight_value on validated block");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── F32 ────────────────────────────────────────────────────────

    #[test]
    fn f32_dequant_round_trips_known_values() {
        // 1.0, -1.5, π ≈ 3.141593, 0.0 in little-endian f32.
        let values = [1.0_f32, -1.5, std::f32::consts::PI, 0.0];
        let mut src = Vec::new();
        for v in values {
            src.extend_from_slice(&v.to_le_bytes());
        }
        let got = dequantize_row(GgufQuantType::F32, &src).unwrap();
        assert_eq!(got, values);
    }

    #[test]
    fn f32_dequant_rejects_non_aligned_source() {
        let src = [0u8; 7]; // not a multiple of 4
        let err = dequantize_row(GgufQuantType::F32, &src).unwrap_err();
        assert!(matches!(err, DequantError::SourceLengthMisaligned { .. }));
    }

    #[test]
    fn f32_dequant_rejects_wrong_dst_length() {
        let mut dst = vec![0.0_f32; 3]; // expects 2 from 8 bytes
        let src = [0u8; 8];
        let err = dequantize_row_into(GgufQuantType::F32, &src, &mut dst).unwrap_err();
        assert!(matches!(
            err,
            DequantError::DestinationLengthMismatch { .. }
        ));
    }

    // ─── F16 ────────────────────────────────────────────────────────

    #[test]
    fn f16_dequant_zero_and_one_bits() {
        // f16 0x0000 = 0.0, 0x3C00 = 1.0, 0xBC00 = -1.0, 0x4000 = 2.0.
        let src = [0x00, 0x00, 0x00, 0x3C, 0x00, 0xBC, 0x00, 0x40];
        let got = dequantize_row(GgufQuantType::F16, &src).unwrap();
        assert_eq!(got, vec![0.0, 1.0, -1.0, 2.0]);
    }

    #[test]
    fn f16_dequant_preserves_nan_and_infinity() {
        // f16 +inf = 0x7C00, -inf = 0xFC00, NaN = 0x7E00.
        let src = [0x00, 0x7C, 0x00, 0xFC, 0x00, 0x7E];
        let got = dequantize_row(GgufQuantType::F16, &src).unwrap();
        assert_eq!(got[0], f32::INFINITY);
        assert_eq!(got[1], f32::NEG_INFINITY);
        assert!(got[2].is_nan());
    }

    #[test]
    fn f16_dequant_rejects_odd_length_source() {
        let src = [0u8; 3];
        let err = dequantize_row(GgufQuantType::F16, &src).unwrap_err();
        assert!(matches!(err, DequantError::SourceLengthMisaligned { .. }));
    }

    // ─── Q8_0 ───────────────────────────────────────────────────────

    /// Helper: build one Q8_0 block from a scale (f16 bits) and
    /// 32 signed-int8 weights.
    fn q8_0_block(scale_bits: u16, weights: [i8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q8_0_BYTES_PER_BLOCK);
        block.extend_from_slice(&scale_bits.to_le_bytes());
        for q in weights {
            block.push(q as u8);
        }
        block
    }

    #[test]
    fn q8_0_dequant_scale_one_is_identity_cast() {
        // scale = 1.0 (f16 bits 0x3C00). Weights 0..32 cast to f32.
        let mut weights = [0_i8; 32];
        for (i, w) in weights.iter_mut().enumerate() {
            *w = i as i8;
        }
        let src = q8_0_block(0x3C00, weights);
        let got = dequantize_row(GgufQuantType::Q8_0, &src).unwrap();
        for (i, &v) in got.iter().enumerate() {
            assert_eq!(v, i as f32, "weight {i}");
        }
    }

    #[test]
    fn q8_0_dequant_scale_applies_to_negative_weights() {
        // scale = 0.5 (f16 bits 0x3800). Weights span the int8 range.
        let mut weights = [0_i8; 32];
        weights[0] = -128;
        weights[1] = -1;
        weights[2] = 0;
        weights[3] = 1;
        weights[4] = 127;
        let src = q8_0_block(0x3800, weights);
        let got = dequantize_row(GgufQuantType::Q8_0, &src).unwrap();
        assert_eq!(got[0], -64.0);
        assert_eq!(got[1], -0.5);
        assert_eq!(got[2], 0.0);
        assert_eq!(got[3], 0.5);
        assert_eq!(got[4], 63.5);
        // Remaining positions (5..32) have weight=0 → dequant 0.
        for g in got.iter().skip(5) {
            assert_eq!(*g, 0.0);
        }
    }

    #[test]
    fn q8_0_dequant_two_blocks_concatenated() {
        // Two blocks: first with scale 1, weights [1; 32]; second
        // with scale 2, weights [-1; 32]. Expect 32 × 1.0 then
        // 32 × -2.0.
        let mut src = q8_0_block(0x3C00, [1_i8; 32]);
        src.extend_from_slice(&q8_0_block(0x4000, [-1_i8; 32]));
        let got = dequantize_row(GgufQuantType::Q8_0, &src).unwrap();
        assert_eq!(got.len(), 64);
        for g in &got[..32] {
            assert_eq!(*g, 1.0);
        }
        for g in &got[32..] {
            assert_eq!(*g, -2.0);
        }
    }

    #[test]
    fn q8_0_dequant_rejects_non_block_aligned_source() {
        // 33 bytes — not a multiple of 34.
        let src = vec![0u8; 33];
        let err = dequantize_row(GgufQuantType::Q8_0, &src).unwrap_err();
        assert!(matches!(err, DequantError::SourceLengthMisaligned { .. }));
    }

    // ─── K-quants ───────────────────────────────────────────────────

    use crate::stego::packing::{q3_k, q4_k, q5_k, q6_k};

    /// Hand-construct a Q4_K block with d=1, dmin=0, all scales+mins
    /// at an arbitrary non-zero value, and nibbles filled via
    /// `qs_value`. The actual dequantized values come from the
    /// packing module's `read_weight_value`; this test just asserts
    /// that `dequant_q4_k` produces the same 256 values in order.
    fn q4_k_synthetic_block() -> Vec<u8> {
        let mut b = vec![0u8; q4_k::BLOCK_BYTES];
        // d = 1.0 (f16 bits 0x3C00), dmin = 0.5 (f16 0x3800).
        b[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
        b[2..4].copy_from_slice(&0x3800_u16.to_le_bytes());
        // scales (12 bytes starting at offset 4): fill with 0x21 =
        // 6-bit scale=33, 6-bit min pair encoded — exact semantics
        // don't matter for self-consistency.
        for slot in &mut b[4..16] {
            *slot = 0x21;
        }
        // qs (128 bytes starting at offset 16): each byte holds
        // two 4-bit nibbles. Use 0x5A so nibbles are 0xA and 0x5.
        for slot in &mut b[16..] {
            *slot = 0x5A;
        }
        b
    }

    #[test]
    fn q4_k_dequant_matches_per_weight_reader() {
        let block = q4_k_synthetic_block();
        let dequanted = dequantize_row(GgufQuantType::Q4K, &block).unwrap();
        assert_eq!(dequanted.len(), 256);
        for (i, got) in dequanted.iter().enumerate() {
            let expected = q4_k::read_weight_value(&block, i).unwrap();
            assert_eq!(*got, expected, "weight {i} diverged");
            assert!(got.is_finite());
        }
    }

    #[test]
    fn q4_k_dequant_two_blocks_concatenated() {
        let mut src = q4_k_synthetic_block();
        // Second block with different qs values to force distinct
        // dequantized values.
        let mut b2 = vec![0u8; q4_k::BLOCK_BYTES];
        b2[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
        b2[2..4].copy_from_slice(&0x3800_u16.to_le_bytes());
        for slot in &mut b2[4..16] {
            *slot = 0x21;
        }
        for slot in &mut b2[16..] {
            *slot = 0xA5;
        }
        src.extend_from_slice(&b2);

        let dequanted = dequantize_row(GgufQuantType::Q4K, &src).unwrap();
        assert_eq!(dequanted.len(), 512);
        for (i, got) in dequanted.iter().take(256).enumerate() {
            let expected = q4_k::read_weight_value(&src[..q4_k::BLOCK_BYTES], i).unwrap();
            assert_eq!(*got, expected);
        }
        for (i, got) in dequanted.iter().skip(256).enumerate() {
            let expected = q4_k::read_weight_value(&src[q4_k::BLOCK_BYTES..], i).unwrap();
            assert_eq!(*got, expected);
        }
    }

    #[test]
    fn k_quants_reject_non_block_aligned_source() {
        // One byte shy of a single block for each K-quant.
        for (quant, block_bytes) in [
            (GgufQuantType::Q3K, q3_k::BLOCK_BYTES),
            (GgufQuantType::Q4K, q4_k::BLOCK_BYTES),
            (GgufQuantType::Q5K, q5_k::BLOCK_BYTES),
            (GgufQuantType::Q6K, q6_k::BLOCK_BYTES),
        ] {
            let src = vec![0u8; block_bytes - 1];
            let err = dequantize_row(quant, &src).unwrap_err();
            assert!(
                matches!(err, DequantError::SourceLengthMisaligned { .. }),
                "{quant:?}: expected SourceLengthMisaligned, got {err:?}",
            );
        }
    }

    #[test]
    fn all_k_quants_produce_finite_values_on_zero_block() {
        // All-zero K-quant blocks are legal (trained weights are
        // often near zero in some channels). Dequant must produce
        // finite outputs — no NaN from degenerate scales.
        for (quant, block_bytes) in [
            (GgufQuantType::Q3K, q3_k::BLOCK_BYTES),
            (GgufQuantType::Q4K, q4_k::BLOCK_BYTES),
            (GgufQuantType::Q5K, q5_k::BLOCK_BYTES),
            (GgufQuantType::Q6K, q6_k::BLOCK_BYTES),
        ] {
            let src = vec![0u8; block_bytes];
            let got = dequantize_row(quant, &src).unwrap();
            assert_eq!(got.len(), 256);
            for (i, v) in got.iter().enumerate() {
                assert!(v.is_finite(), "{quant:?} weight {i} = {v}");
            }
        }
    }

    // ─── Unsupported quants ─────────────────────────────────────────

    #[test]
    fn q2_k_returns_unsupported() {
        // Q2_K is in the enum but has no packing module yet.
        let err = dequantize_row(GgufQuantType::Q2K, &[0u8; 256]).unwrap_err();
        assert!(
            matches!(err, DequantError::Unsupported { quant: GgufQuantType::Q2K }),
            "expected Unsupported(Q2K), got {err:?}",
        );
    }

    // ─── Weight count helper ────────────────────────────────────────

    #[test]
    fn weight_count_reports_expected_values() {
        assert_eq!(weight_count(GgufQuantType::F32, 16), Some(4));
        assert_eq!(weight_count(GgufQuantType::F16, 16), Some(8));
        assert_eq!(weight_count(GgufQuantType::Q8_0, 68), Some(64));
        assert_eq!(weight_count(GgufQuantType::F32, 17), None);
        assert_eq!(weight_count(GgufQuantType::Q8_0, 33), None);
        // K-quants: one block = 256 weights.
        assert_eq!(weight_count(GgufQuantType::Q4K, q4_k::BLOCK_BYTES), Some(256));
        assert_eq!(
            weight_count(GgufQuantType::Q6K, q6_k::BLOCK_BYTES * 3),
            Some(768),
        );
        // Misaligned input.
        assert_eq!(weight_count(GgufQuantType::Q4K, 100), None);
        assert_eq!(weight_count(GgufQuantType::Q2K, q4_k::BLOCK_BYTES), None);
    }
}
