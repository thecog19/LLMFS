//! Magnitude-only salience estimator (Tier 0). Ranks eligible weights
//! by `|w|` ascending: the lowest-magnitude weights are the safest
//! places to steal bits from, because their contribution to any
//! forward activation is bounded by their own (small) magnitude.
//!
//! No calibration corpus, no forward pass — a single pass over the
//! mmap'd weights is enough. This is what Layer 0 (DESIGN-NEW §15.2)
//! uses to place metadata at deterministic, calibration-free
//! positions.

use crate::gguf::quant::GgufQuantType;
use crate::stego::calibration::{WeightRef, stealable_bits_for};
use crate::stego::packing::{float, q4_k, q5_k, q6_k};
use crate::stego::tensor_map::{TensorMap, TensorSlot};

/// Read the absolute value of a single weight from the mmap'd cover
/// file. Dispatches on quant type. Returns 0.0 for quant types we
/// don't yet decode — the caller treats those as "minimal salience"
/// (eligible first). Each unimplemented variant is its own concrete
/// TODO; calibration on covers using these types will silently
/// mis-rank until they're real.
pub fn read_weight_abs(mmap: &[u8], slot: &TensorSlot, weight_index: u64) -> f32 {
    match slot.quant_type {
        GgufQuantType::F16 => read_f16_abs(mmap, slot.data_offset, weight_index),
        GgufQuantType::F32 => read_f32_abs(mmap, slot.data_offset, weight_index),
        GgufQuantType::Q8_0 => read_q8_0_abs(mmap, slot.data_offset, weight_index),
        GgufQuantType::Q4K => read_q4_k_abs(mmap, slot.data_offset, weight_index),
        GgufQuantType::Q5K => read_q5_k_abs(mmap, slot.data_offset, weight_index),
        GgufQuantType::Q6K => read_q6_k_abs(mmap, slot.data_offset, weight_index),
        // TODO: Q3_K and the legacy Q4_0 / Q4_1 / Q5_0 / Q5_1 /
        // Q8_1 / Q2_K / Q8_K variants. Stubbed at 0.0 → ranked
        // first, which is incorrect; covers using these quant types
        // will silently mis-rank.
        GgufQuantType::Q3K
        | GgufQuantType::Q2K
        | GgufQuantType::Q4_0
        | GgufQuantType::Q4_1
        | GgufQuantType::Q5_0
        | GgufQuantType::Q5_1
        | GgufQuantType::Q8_1
        | GgufQuantType::Q8K => 0.0,
    }
}

fn read_q4_k_abs(mmap: &[u8], data_offset: u64, weight_index: u64) -> f32 {
    let block_weights = q4_k::WEIGHTS_PER_BLOCK as u64;
    let block_bytes = q4_k::BLOCK_BYTES;
    let block_idx = weight_index / block_weights;
    let in_block = (weight_index % block_weights) as usize;
    let block_start = data_offset as usize + block_idx as usize * block_bytes;
    let block = &mmap[block_start..block_start + block_bytes];
    q4_k::read_weight_value(block, in_block)
        .expect("q4_k weight read invariant violated")
        .abs()
}

fn read_q5_k_abs(mmap: &[u8], data_offset: u64, weight_index: u64) -> f32 {
    let block_weights = q5_k::WEIGHTS_PER_BLOCK as u64;
    let block_bytes = q5_k::BLOCK_BYTES;
    let block_idx = weight_index / block_weights;
    let in_block = (weight_index % block_weights) as usize;
    let block_start = data_offset as usize + block_idx as usize * block_bytes;
    let block = &mmap[block_start..block_start + block_bytes];
    q5_k::read_weight_value(block, in_block)
        .expect("q5_k weight read invariant violated")
        .abs()
}

fn read_f16_abs(mmap: &[u8], data_offset: u64, weight_index: u64) -> f32 {
    let off = (data_offset + weight_index * 2) as usize;
    let bits = u16::from_le_bytes([mmap[off], mmap[off + 1]]);
    float::f16_to_f32(bits).abs()
}

fn read_f32_abs(mmap: &[u8], data_offset: u64, weight_index: u64) -> f32 {
    let off = (data_offset + weight_index * 4) as usize;
    let bits = u32::from_le_bytes([mmap[off], mmap[off + 1], mmap[off + 2], mmap[off + 3]]);
    f32::from_bits(bits).abs()
}

fn read_q8_0_abs(mmap: &[u8], data_offset: u64, weight_index: u64) -> f32 {
    // Q8_0 layout: per 32-weight block, 2-byte fp16 scale followed by
    // 32 int8 values (34 bytes total).
    const BLOCK_WEIGHTS: u64 = 32;
    const BLOCK_BYTES: u64 = 34;
    let block_idx = weight_index / BLOCK_WEIGHTS;
    let in_block = (weight_index % BLOCK_WEIGHTS) as usize;
    let block_off = (data_offset + block_idx * BLOCK_BYTES) as usize;
    let scale_bits = u16::from_le_bytes([mmap[block_off], mmap[block_off + 1]]);
    let scale = float::f16_to_f32(scale_bits);
    let int8 = mmap[block_off + 2 + in_block] as i8;
    (int8 as f32 * scale).abs()
}

fn read_q6_k_abs(mmap: &[u8], data_offset: u64, weight_index: u64) -> f32 {
    let block_weights = q6_k::WEIGHTS_PER_BLOCK as u64;
    let block_bytes = q6_k::BLOCK_BYTES;
    let block_idx = weight_index / block_weights;
    let in_block = (weight_index % block_weights) as usize;
    let block_start = data_offset as usize + block_idx as usize * block_bytes;
    let block = &mmap[block_start..block_start + block_bytes];
    // The packer's value reader does its own bounds checks. If they
    // ever fail it's a programmer error (calibration walked off-tensor),
    // hence panic.
    q6_k::read_weight_value(block, in_block)
        .expect("q6_k weight read invariant violated")
        .abs()
}

/// Return the `n` lowest-`|w|` weights across all slots in `map`,
/// sorted ascending. Two passes over the mmap: one for a magnitude
/// histogram, one to collect candidates from the buckets at-or-below
/// the cutoff. Within the cutoff bucket we use `select_nth_unstable`
/// to pick exactly the right slice.
///
/// Total cost: `O(total_weights)` per pass; the constant factor is a
/// log2 + bucket lookup per weight on the hot path. About 20× faster
/// in practice than a size-`n` min-heap on million-scale weights
/// because there's no per-weight `log n` factor and the hot loop
/// stays branch-predictable.
///
/// Trained model weights don't contain NaN; `debug_assert` catches
/// a reader bug if one ever appears.
pub fn lowest_magnitude_weights(mmap: &[u8], map: &TensorMap, n: usize) -> Vec<WeightRef> {
    if n == 0 {
        return Vec::new();
    }
    let total: u64 = map.slots.iter().map(|s| s.weight_count).sum();
    let n = n.min(total as usize);
    if n == 0 {
        return Vec::new();
    }

    // Pass 1: histogram. Bucket 0 = magnitude == 0 (common in trained
    // models from pruning / init patterns); buckets 1..=NUM_BUCKETS-1
    // are log2-spaced.
    let mut histogram = [0_u64; NUM_BUCKETS];
    for slot in &map.slots {
        for weight_index in 0..slot.weight_count {
            let mag = read_weight_abs(mmap, slot, weight_index);
            debug_assert!(!mag.is_nan(), "weight magnitude unexpectedly NaN");
            histogram[bucket_for(mag)] += 1;
        }
    }

    // Find the cutoff bucket: the smallest k such that the running
    // sum of histogram[0..=k] reaches n.
    let mut cumulative = 0_u64;
    let mut cutoff_bucket = NUM_BUCKETS - 1;
    for (b, &count) in histogram.iter().enumerate() {
        cumulative = cumulative.saturating_add(count);
        if cumulative >= n as u64 {
            cutoff_bucket = b;
            break;
        }
    }

    // Pass 2: collect every weight in buckets 0..=cutoff_bucket,
    // tagged with magnitude so we can finalize the selection.
    let collected_capacity = cumulative.min(total) as usize;
    let mut collected: Vec<(F32Ord, WeightRef)> = Vec::with_capacity(collected_capacity);
    for (slot_idx, slot) in map.slots.iter().enumerate() {
        for weight_index in 0..slot.weight_count {
            let mag = read_weight_abs(mmap, slot, weight_index);
            if bucket_for(mag) <= cutoff_bucket {
                collected.push((
                    F32Ord(mag),
                    WeightRef {
                        slot_index: slot_idx as u32,
                        weight_index,
                    },
                ));
            }
        }
    }

    // The set we collected is a superset of the bottom-n: the cutoff
    // bucket holds all of its weights, but we only need enough to
    // reach n. Use `select_nth_unstable` to partition in O(N) and
    // then sort the lower partition for the deterministic output.
    if collected.len() > n {
        collected.select_nth_unstable_by(n, |a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        collected.truncate(n);
    }
    collected.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    collected.into_iter().map(|(_, r)| r).collect()
}

const NUM_BUCKETS: usize = 257;

/// Quantize a non-negative magnitude into one of `NUM_BUCKETS`
/// buckets. Bucket 0 is exactly zero (very common — pruned weights
/// and init patterns); buckets 1..=256 are log2-spaced over the
/// representable f32 range.
fn bucket_for(mag: f32) -> usize {
    if mag == 0.0 {
        return 0;
    }
    // log2 of any finite f32 is in roughly [-149, 128]; shift by 128
    // and clamp into [1, 256]. NaN won't reach here (debug_assert
    // upstream); inf falls into the top bucket.
    if !mag.is_finite() {
        return NUM_BUCKETS - 1;
    }
    let b = (mag.log2().floor() as i32 + 128).clamp(1, (NUM_BUCKETS - 1) as i32);
    b as usize
}

/// Return the lowest-magnitude weights whose combined stealable-bit
/// capacity is at least `needed_bits`. Used by Layer 0 (implicit
/// metadata addressing) to find a metadata-sized region in the
/// least-salient weights.
///
/// Picks an upper-bound candidate count assuming worst-case 1 bit per
/// weight (rules out asking for fewer than needed). May return slightly
/// more weights than strictly required — that's fine because callers
/// truncate to the bit budget at the per-bit allocation step.
pub fn lowest_magnitude_weights_for_bits(
    mmap: &[u8],
    map: &TensorMap,
    needed_bits: u64,
) -> Vec<WeightRef> {
    if needed_bits == 0 {
        return Vec::new();
    }
    // Q3K / Q4K / Q5K all advertise 1 stealable bit per weight, so the
    // worst-case candidate count is `needed_bits` itself. Cap at total
    // weight count to avoid asking for more than exists.
    let total_weights: u64 = map.slots.iter().map(|s| s.weight_count).sum();
    let estimate = (needed_bits as usize).min(total_weights as usize);
    let candidates = lowest_magnitude_weights(mmap, map, estimate);

    let mut accumulated_bits = 0_u64;
    let mut chosen = Vec::new();
    for r in candidates {
        let bits = stealable_bits_for(map.slots[r.slot_index as usize].quant_type) as u64;
        accumulated_bits = accumulated_bits.saturating_add(bits);
        chosen.push(r);
        if accumulated_bits >= needed_bits {
            return chosen;
        }
    }
    chosen
}

/// `f32` wrapper that implements `Ord` via `partial_cmp`. Magnitudes
/// are non-negative finite floats in trained models, so `partial_cmp`
/// is total in practice; NaN would panic via `expect`. PartialOrd is
/// implemented manually to keep clippy's `derive_ord_xor_partial_ord`
/// satisfied — it must agree with `Ord`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct F32Ord(f32);

impl Eq for F32Ord {}
impl Ord for F32Ord {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .expect("magnitude must not be NaN")
    }
}
impl PartialOrd for F32Ord {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::stego::packing::float::f16_to_f32;

    #[test]
    fn f16_to_f32_decodes_canonical_values() {
        // 1.0 = 0x3C00, -2.0 = 0xC000, 0.5 = 0x3800
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
    }

    #[test]
    fn f16_to_f32_handles_subnormal_and_specials() {
        // smallest positive subnormal = 2^-24 ≈ 5.96e-8
        let s = f16_to_f32(0x0001);
        assert!((s - 2.0_f32.powi(-24)).abs() < 1e-30);
        // +inf / -inf
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY);
    }

    #[test]
    fn read_f16_abs_pulls_from_offset() {
        // [0x00, 0x00, 0x00, 0x3C] → weight 0 = 0.0, weight 1 = 1.0
        let mmap = vec![0x00, 0x00, 0x00, 0x3C];
        assert_eq!(read_f16_abs(&mmap, 0, 0), 0.0);
        assert_eq!(read_f16_abs(&mmap, 0, 1), 1.0);
    }

    #[test]
    fn read_q8_0_abs_applies_per_block_scale() {
        // One Q8_0 block: 34 bytes = 2-byte fp16 scale + 32 int8 weights.
        // Scale = 0.5 (fp16 0x3800). Weights: [4, -8, 0, ...] → |0.5*4|=2.0, |0.5*-8|=4.0, 0.0
        let mut block = vec![0_u8; 34];
        block[0] = 0x00;
        block[1] = 0x38; // fp16 0x3800 = 0.5
        block[2] = 4; // int8 weight 0
        block[3] = (-8_i8) as u8; // int8 weight 1
        block[4] = 0; // int8 weight 2
        assert_eq!(read_q8_0_abs(&block, 0, 0), 2.0);
        assert_eq!(read_q8_0_abs(&block, 0, 1), 4.0);
        assert_eq!(read_q8_0_abs(&block, 0, 2), 0.0);
    }

    #[test]
    fn read_q8_0_abs_walks_to_second_block() {
        // Two blocks back-to-back (68 bytes).
        // Block 0 scale = 1.0 (0x3C00), weights [0..32] = 1
        // Block 1 scale = 2.0 (0x4000), weights [0..32] = 3
        let mut buf = vec![0_u8; 68];
        buf[0] = 0x00;
        buf[1] = 0x3C; // 1.0
        for i in 0..32 {
            buf[2 + i] = 1;
        }
        buf[34] = 0x00;
        buf[35] = 0x40; // 2.0
        for i in 0..32 {
            buf[36 + i] = 3;
        }
        // Block 0, weight 0: |1 * 1| = 1.0
        assert_eq!(read_q8_0_abs(&buf, 0, 0), 1.0);
        // Block 1, weight 0 (weight_index 32): |2 * 3| = 6.0
        assert_eq!(read_q8_0_abs(&buf, 0, 32), 6.0);
    }

    #[test]
    fn lowest_magnitude_weights_empty_request_returns_empty() {
        let map = TensorMap {
            slots: Vec::new(),
            total_capacity_bits: 0,
            total_capacity_bytes: 0,
        };
        assert!(lowest_magnitude_weights(&[], &map, 0).is_empty());
    }
}
