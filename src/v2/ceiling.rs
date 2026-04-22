//! Ceiling-magnitude bucket summary.
//!
//! V2's allocator needs to answer "what is the max ceiling magnitude
//! over weights `[start, start + len)`?" repeatedly during free-run
//! priority queue maintenance. Scanning every weight on each split /
//! merge is O(len); instead we precompute a **256-weight-bucket
//! summary**: one `f32` per bucket recording the max
//! [`read_weight_ceiling_abs`](crate::stego::calibration::magnitude::read_weight_ceiling_abs)
//! over weights inside it.
//!
//! Range queries touch at most `⌈len / 256⌉ + 1` buckets — O(1) in
//! practice since allocations rarely span more than a handful of
//! buckets. The trade-off: queries that span a partial bucket return
//! the max over the *whole* bucket, not just the queried sub-range.
//! That's an overestimate bounded by one bucket's worth of weights
//! (256) at each end of the range — acceptable pessimism for
//! allocator ranking (DESIGN-NEW §15.5).
//!
//! The summary is computed once at [`llmdb init`]-time and persisted
//! as a regular file inside V2's filesystem; mount reads the bytes
//! back into the in-memory structure. The ~5 MB size (for a 135M-
//! weight cover) is significant but tractable — significantly smaller
//! than the per-weight ceiling array (540 MB for the same cover).
//!
//! # Serialisation format
//!
//! ```text
//! offset   size   field
//! 0        4      magic = b"CSUM"
//! 4        1      version = 1
//! 5        3      reserved (zero)
//! 8        4      slot_count u32 LE
//! 12       4*S    per-slot bucket counts (u32 LE each, S = slot_count)
//! 12+4*S   ...    bucket data: concatenated f32 LE arrays per slot
//! ```
//!
//! No padding is required because the header is a multiple of 4 bytes
//! and both u32 and f32 are 4-byte-aligned.
//!
//! [`llmdb init`]: <https://example.invalid/>

use thiserror::Error;

use crate::stego::calibration::magnitude::read_weight_ceiling_abs;
use crate::stego::tensor_map::TensorMap;

/// Number of weights covered by each bucket in the summary.
pub const BUCKET_SIZE: u64 = 256;

const MAGIC: &[u8; 4] = b"CSUM";
const VERSION: u8 = 1;
const HEADER_LEN: usize = 12;

/// Persisted ceiling-magnitude bucket summary.
#[derive(Debug, Clone, PartialEq)]
pub struct CeilingSummary {
    /// `per_slot[slot_index][bucket_index]` = max
    /// `ceiling_magnitude(w)` over the weights in that bucket.
    per_slot: Vec<Vec<f32>>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CeilingError {
    #[error("bad magic: expected b\"CSUM\", got {found:?}")]
    BadMagic { found: [u8; 4] },
    #[error("unsupported version: {0}")]
    UnsupportedVersion(u8),
    #[error("truncated: {reason}")]
    Truncated { reason: &'static str },
}

impl CeilingSummary {
    /// Build the summary by scanning every eligible weight in every
    /// slot once. Cost is O(total_eligible_weights). Non-stealable
    /// quant types (where `read_weight_ceiling_abs` would return 0.0)
    /// still produce a slot entry — with all-zero buckets — so slot
    /// indexing stays aligned with [`TensorMap::slots`].
    pub fn build(mmap: &[u8], map: &TensorMap) -> Self {
        let mut per_slot = Vec::with_capacity(map.slots.len());
        for slot in &map.slots {
            let bucket_count = bucket_count_for_weights(slot.weight_count);
            let mut buckets = vec![0.0_f32; bucket_count];
            for weight_index in 0..slot.weight_count {
                let bucket = (weight_index / BUCKET_SIZE) as usize;
                let c = read_weight_ceiling_abs(mmap, slot, weight_index);
                if c > buckets[bucket] {
                    buckets[bucket] = c;
                }
            }
            per_slot.push(buckets);
        }
        Self { per_slot }
    }

    /// Number of slots covered by the summary.
    pub fn slot_count(&self) -> u32 {
        self.per_slot.len() as u32
    }

    /// Number of buckets for a given slot.
    pub fn bucket_count(&self, slot_index: u32) -> usize {
        self.per_slot[slot_index as usize].len()
    }

    /// Max ceiling magnitude in a single bucket. Exact.
    pub fn bucket_max(&self, slot_index: u32, bucket_index: usize) -> f32 {
        self.per_slot[slot_index as usize][bucket_index]
    }

    /// Max ceiling magnitude over weight range `[start, start + len)`.
    /// Returns `0.0` for `len == 0`. Over-approximates for ranges that
    /// don't fall on bucket boundaries: buckets partially intersected
    /// by the range contribute their full bucket max. Ranges beyond
    /// the slot's weight count are clamped at the last bucket — no
    /// panic.
    pub fn max_over_range(&self, slot_index: u32, start_weight: u64, len_weights: u64) -> f32 {
        if len_weights == 0 {
            return 0.0;
        }
        let buckets = &self.per_slot[slot_index as usize];
        if buckets.is_empty() {
            return 0.0;
        }
        let first = (start_weight / BUCKET_SIZE) as usize;
        let last_weight = start_weight.saturating_add(len_weights - 1);
        let last = ((last_weight / BUCKET_SIZE) as usize).min(buckets.len() - 1);
        if first >= buckets.len() {
            return 0.0;
        }
        buckets[first..=last]
            .iter()
            .copied()
            .fold(0.0_f32, f32::max)
    }

    /// Serialise to the format documented at module level.
    pub fn serialize(&self) -> Vec<u8> {
        let slot_count = self.per_slot.len() as u32;
        let total_buckets: usize = self.per_slot.iter().map(|s| s.len()).sum();
        let mut out = Vec::with_capacity(HEADER_LEN + 4 * slot_count as usize + 4 * total_buckets);
        out.extend_from_slice(MAGIC);
        out.push(VERSION);
        out.extend_from_slice(&[0_u8; 3]); // reserved
        out.extend_from_slice(&slot_count.to_le_bytes());
        for slot in &self.per_slot {
            out.extend_from_slice(&(slot.len() as u32).to_le_bytes());
        }
        for slot in &self.per_slot {
            for &v in slot {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    /// Deserialise from bytes produced by [`Self::serialize`]. Returns
    /// [`CeilingError`] on magic / version / length mismatch.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, CeilingError> {
        if bytes.len() < HEADER_LEN {
            return Err(CeilingError::Truncated {
                reason: "header shorter than 12 bytes",
            });
        }
        let magic: [u8; 4] = bytes[0..4].try_into().unwrap();
        if &magic != MAGIC {
            return Err(CeilingError::BadMagic { found: magic });
        }
        let version = bytes[4];
        if version != VERSION {
            return Err(CeilingError::UnsupportedVersion(version));
        }
        let slot_count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;

        let per_slot_counts_start = HEADER_LEN;
        let per_slot_counts_end = per_slot_counts_start + 4 * slot_count;
        if bytes.len() < per_slot_counts_end {
            return Err(CeilingError::Truncated {
                reason: "per-slot bucket count table truncated",
            });
        }

        let mut bucket_counts = Vec::with_capacity(slot_count);
        for i in 0..slot_count {
            let off = per_slot_counts_start + 4 * i;
            let n = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
            bucket_counts.push(n);
        }

        let total_buckets: usize = bucket_counts.iter().sum();
        let data_start = per_slot_counts_end;
        let data_end = data_start + 4 * total_buckets;
        if bytes.len() < data_end {
            return Err(CeilingError::Truncated {
                reason: "bucket data truncated",
            });
        }

        let mut per_slot = Vec::with_capacity(slot_count);
        let mut cursor = data_start;
        for &count in &bucket_counts {
            let mut buckets = Vec::with_capacity(count);
            for _ in 0..count {
                let v = f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
                buckets.push(v);
                cursor += 4;
            }
            per_slot.push(buckets);
        }

        Ok(Self { per_slot })
    }
}

/// Bucket count for a slot of `weight_count` weights. `⌈weight_count
/// / BUCKET_SIZE⌉`, with zero-weight slots mapping to zero buckets
/// (uncommon but tolerated).
fn bucket_count_for_weights(weight_count: u64) -> usize {
    if weight_count == 0 {
        0
    } else {
        weight_count.div_ceil(BUCKET_SIZE) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_count_zero_weight_slot() {
        assert_eq!(bucket_count_for_weights(0), 0);
    }

    #[test]
    fn bucket_count_one_weight_slot() {
        assert_eq!(bucket_count_for_weights(1), 1);
    }

    #[test]
    fn bucket_count_full_single_bucket() {
        assert_eq!(bucket_count_for_weights(256), 1);
    }

    #[test]
    fn bucket_count_spans_buckets() {
        assert_eq!(bucket_count_for_weights(257), 2);
        assert_eq!(bucket_count_for_weights(512), 2);
        assert_eq!(bucket_count_for_weights(513), 3);
    }
}
