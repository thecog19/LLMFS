//! Low-level bit I/O against mmap'd cover weights, keyed by
//! `MetadataBitPos`. This is the narrow contract between V2's
//! calibration layer (which ranks and selects bit positions) and the
//! eventual Layer-0 metadata writer in the device (which stores the
//! V2 scatter map there).
//!
//! The addressing convention: `bit_index` is always
//! `0..stealable_bits_for(quant_type)` and numbered from the
//! least-significant stealable bit of the weight upward. Where the
//! stealable bits land in the cover file is packer-specific, so each
//! quant type has its own `locate` branch. Keep this map in sync with
//! the packers in `src/stego/packing/*` — a divergence here means
//! metadata written via [`write_bit`] can't be recovered by the
//! V1-style extractors and vice versa.
//!
//! Currently implemented: F16, F32, Q8_0, Q4_K, Q5_K, Q6_K — the
//! same set the magnitude estimator decodes. Other GGUF quant types
//! panic because we have no dequantizer for them yet (mirrors the
//! TODO stubs in `calibration::magnitude::read_weight_abs`).

use crate::gguf::quant::GgufQuantType;
use crate::stego::calibration::placement::MetadataBitPos;
use crate::stego::calibration::stealable_bits_for;
use crate::stego::packing::{q4_k, q5_k, q6_k};
use crate::stego::tensor_map::TensorSlot;

/// Byte offset + bit position (0..7) of a stealable bit in the mmap.
struct BitSite {
    byte_offset: usize,
    bit_in_byte: u8,
}

/// Resolve a `MetadataBitPos` against one slot to a concrete byte +
/// bit in the mmap. Panics on unsupported quant types (keeps the
/// failure mode loud — the caller should have gated on
/// `stealable_bits_for > 0` before asking).
fn locate(slot: &TensorSlot, pos: MetadataBitPos) -> BitSite {
    debug_assert!(
        (pos.bit_index as u32) < stealable_bits_for(slot.quant_type),
        "bit_index {} out of stealable range for {:?}",
        pos.bit_index,
        slot.quant_type,
    );
    debug_assert!(
        pos.weight_index < slot.weight_count,
        "weight_index {} out of range for slot weight_count {}",
        pos.weight_index,
        slot.weight_count,
    );

    match slot.quant_type {
        GgufQuantType::F16 => {
            // Low 4 bits of the low byte of the fp16 (little-endian).
            // `F16Packer::embed` rewrites exactly this nibble.
            let byte_offset = slot.data_offset as usize + (pos.weight_index as usize) * 2;
            BitSite {
                byte_offset,
                bit_in_byte: pos.bit_index,
            }
        }
        GgufQuantType::F32 => {
            // Low byte of the f32 (little-endian); all 8 bits stealable.
            let byte_offset = slot.data_offset as usize + (pos.weight_index as usize) * 4;
            BitSite {
                byte_offset,
                bit_in_byte: pos.bit_index,
            }
        }
        GgufQuantType::Q8_0 => {
            // 34-byte block = 2-byte fp16 scale + 32 int8 quants.
            // Low nibble of the quant byte is stealable (4 bits).
            const BLOCK_WEIGHTS: u64 = 32;
            const BLOCK_BYTES: u64 = 34;
            const SCALE_BYTES: u64 = 2;
            let block_idx = pos.weight_index / BLOCK_WEIGHTS;
            let in_block = (pos.weight_index % BLOCK_WEIGHTS) as usize;
            let byte_offset = slot.data_offset as usize
                + (block_idx * BLOCK_BYTES) as usize
                + SCALE_BYTES as usize
                + in_block;
            BitSite {
                byte_offset,
                bit_in_byte: pos.bit_index,
            }
        }
        GgufQuantType::Q4K => {
            // 144-byte block; qs[128] at offset 16 (after d=2, dmin=2,
            // scales=12). 256 weights per block, 2 per qs byte: weights
            // with `half == 0` use the low nibble, `half == 1` the high.
            // The single stealable bit is the LSB of that 4-bit nibble,
            // i.e. bit 0 for low-half weights, bit 4 for high-half.
            let block_weights = q4_k::WEIGHTS_PER_BLOCK as u64;
            let block_bytes = q4_k::BLOCK_BYTES;
            let block_idx = pos.weight_index / block_weights;
            let in_block = (pos.weight_index % block_weights) as usize;
            let j_outer = in_block / 64;
            let within = in_block % 64;
            let half = within / 32;
            let l = within % 32;
            const QS_OFFSET: usize = 4 + 12; // d + dmin + scales
            let qs_byte_idx = QS_OFFSET + j_outer * 32 + l;
            let byte_offset =
                slot.data_offset as usize + block_idx as usize * block_bytes + qs_byte_idx;
            let bit_in_byte = if half == 0 { 0 } else { 4 };
            BitSite {
                byte_offset,
                bit_in_byte,
            }
        }
        GgufQuantType::Q5K => {
            // 176-byte block; qs[128] at offset 48 (after d=2, dmin=2,
            // scales=12, qh=32). Per-weight nibble addressing is
            // identical to Q4_K; the qh high-bit isn't stealable (it
            // would flip the weight's MSB and move its magnitude).
            let block_weights = q5_k::WEIGHTS_PER_BLOCK as u64;
            let block_bytes = q5_k::BLOCK_BYTES;
            let block_idx = pos.weight_index / block_weights;
            let in_block = (pos.weight_index % block_weights) as usize;
            let j_outer = in_block / 64;
            let within = in_block % 64;
            let half = within / 32;
            let l = within % 32;
            const QS_OFFSET: usize = 4 + 12 + 32;
            let qs_byte_idx = QS_OFFSET + j_outer * 32 + l;
            let byte_offset =
                slot.data_offset as usize + block_idx as usize * block_bytes + qs_byte_idx;
            let bit_in_byte = if half == 0 { 0 } else { 4 };
            BitSite {
                byte_offset,
                bit_in_byte,
            }
        }
        GgufQuantType::Q6K => {
            // 210-byte block; ql[128] at offset 0. Two stealable bits
            // per weight: the 2 LSBs of the 4-bit ql nibble. Low-nibble
            // weights (quadrant 0-1) land in bits 0-1 of the ql byte;
            // high-nibble weights (quadrant 2-3) in bits 4-5.
            let block_weights = q6_k::WEIGHTS_PER_BLOCK as u64;
            let block_bytes = q6_k::BLOCK_BYTES;
            let block_idx = pos.weight_index / block_weights;
            let in_block = (pos.weight_index % block_weights) as usize;
            let super_n = in_block / 128;
            let within = in_block % 128;
            let quadrant = within / 32;
            let l = within % 32;
            let ql_idx = super_n * 64 + l + (quadrant % 2) * 32;
            let byte_offset = slot.data_offset as usize + block_idx as usize * block_bytes + ql_idx;
            let base_bit = if quadrant < 2 { 0 } else { 4 };
            let bit_in_byte = base_bit + pos.bit_index;
            BitSite {
                byte_offset,
                bit_in_byte,
            }
        }
        other => panic!(
            "calibration::bit_io: no bit addressing for {:?} — add a decoder in \
             calibration::magnitude and a locate() arm here, in lockstep",
            other
        ),
    }
}

/// Read one stealable bit from the cover.
pub fn read_bit(mmap: &[u8], slot: &TensorSlot, pos: MetadataBitPos) -> bool {
    let site = locate(slot, pos);
    (mmap[site.byte_offset] >> site.bit_in_byte) & 1 == 1
}

/// Write one stealable bit in the cover, preserving every other bit of
/// the byte. Caller owns concurrency; this is a single byte-level RMW.
pub fn write_bit(mmap: &mut [u8], slot: &TensorSlot, pos: MetadataBitPos, value: bool) {
    let site = locate(slot, pos);
    let mask = 1_u8 << site.bit_in_byte;
    if value {
        mmap[site.byte_offset] |= mask;
    } else {
        mmap[site.byte_offset] &= !mask;
    }
}
