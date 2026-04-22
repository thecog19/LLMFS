//! Inference-time dequantization: packed cover bytes → `f32` weights.
//!
//! Sibling to `src/stego/packing/`. The existing packing modules know
//! how to *embed stealable bits* into each quant block and how to
//! compute *ceiling magnitudes* for anchor placement; they don't
//! concern themselves with reconstructing the f32 weight the model
//! multiplies against.
//!
//! This module is that reconstruction. Milestone A implements the
//! three quant types our target GGUFs use for F16/F32 inference:
//! `F32`, `F16`, and `Q8_0` (only F32 + F16 actually appear in
//! SmolLM2-135M-F16; `Q8_0` is included so the milestone-A forward
//! pass can mix quant types inside a model if needed, and because
//! the extension over `read_q8_0_abs` in `magnitude.rs` is trivial).
//!
//! K-quants (`Q3_K` / `Q4_K` / `Q5_K` / `Q6_K`) are deferred to
//! Milestone C. The existing `src/stego/packing/{q3_k,q4_k,q5_k,q6_k}.rs`
//! decoders can be adapted when that work lands.
//!
//! # Block layout recap (for the quant types this module handles)
//!
//! ```text
//! F32   4 bytes / weight, little-endian IEEE-754 single.
//! F16   2 bytes / weight, little-endian IEEE-754 half.
//! Q8_0  34 bytes / 32 weights. Block = [scale: f16 le][q: int8 × 32].
//!       Dequantized weight = f16_to_f32(scale) × int8_i.
//! ```

use thiserror::Error;

use crate::gguf::quant::GgufQuantType;
use crate::stego::packing::float::f16_to_f32;

/// Bytes per weight for F16.
pub const F16_BYTES_PER_WEIGHT: usize = 2;
/// Bytes per weight for F32.
pub const F32_BYTES_PER_WEIGHT: usize = 4;
/// Weights per Q8_0 block.
pub const Q8_0_WEIGHTS_PER_BLOCK: usize = 32;
/// Bytes per Q8_0 block: 2-byte f16 scale + 32 int8 weights.
pub const Q8_0_BYTES_PER_BLOCK: usize = 2 + Q8_0_WEIGHTS_PER_BLOCK;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DequantError {
    #[error("{quant:?}: unsupported in Milestone A — see `crate::forward::dequant` module docs")]
    Unsupported { quant: GgufQuantType },

    #[error(
        "{quant:?}: source length {src_len} is not a multiple of the {unit}-byte unit"
    )]
    SourceLengthMisaligned {
        quant: GgufQuantType,
        src_len: usize,
        unit: usize,
    },

    #[error(
        "{quant:?}: destination holds {dst_weights} weights, source encodes {src_weights}"
    )]
    DestinationLengthMismatch {
        quant: GgufQuantType,
        dst_weights: usize,
        src_weights: usize,
    },
}

/// How many f32 weights `src.len()` bytes of `quant` decodes to.
/// Returns `None` for quant types this module doesn't (yet) handle.
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
        _ => None,
    }
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
        other => Err(DequantError::Unsupported { quant: other }),
    }
}

/// Allocate-on-demand variant. Use this in code paths where the
/// weight count isn't known up front; prefer the `_into` form in
/// hot loops.
pub fn dequantize_row(quant: GgufQuantType, src: &[u8]) -> Result<Vec<f32>, DequantError> {
    // Reject unsupported quant types before touching `src.len()`,
    // so callers get `Unsupported` for K-quants and
    // `SourceLengthMisaligned` for F32/F16/Q8_0 with bad lengths.
    let unit = match quant {
        GgufQuantType::F32 => F32_BYTES_PER_WEIGHT,
        GgufQuantType::F16 => F16_BYTES_PER_WEIGHT,
        GgufQuantType::Q8_0 => Q8_0_BYTES_PER_BLOCK,
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
        assert!(matches!(err, DequantError::DestinationLengthMismatch { .. }));
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

    // ─── Unsupported quants ─────────────────────────────────────────

    #[test]
    fn k_quants_return_unsupported() {
        for q in [
            GgufQuantType::Q4K,
            GgufQuantType::Q5K,
            GgufQuantType::Q6K,
            GgufQuantType::Q3K,
            GgufQuantType::Q2K,
        ] {
            let err = dequantize_row(q, &[0u8; 256]).unwrap_err();
            assert!(
                matches!(err, DequantError::Unsupported { quant } if quant == q),
                "expected Unsupported for {q:?}, got {err:?}",
            );
        }
    }

    // ─── Weight count helper ────────────────────────────────────────

    #[test]
    fn weight_count_reports_expected_values() {
        assert_eq!(weight_count(GgufQuantType::F32, 16), Some(4));
        assert_eq!(weight_count(GgufQuantType::F16, 16), Some(8));
        assert_eq!(weight_count(GgufQuantType::Q8_0, 68), Some(64));
        assert_eq!(weight_count(GgufQuantType::F32, 17), None);
        assert_eq!(weight_count(GgufQuantType::Q8_0, 33), None);
        assert_eq!(weight_count(GgufQuantType::Q4K, 256), None);
    }
}
