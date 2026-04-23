//! Per-weight salience table + on-disk codec.
//!
//! B1's `AwqCollector` produces per-input-channel salience for each
//! linear weight tensor. For the allocator (B4) we need fast
//! `max_over_range` lookups keyed on **weight index**, mirroring
//! [`crate::v2::ceiling::CeilingSummary::max_over_range`] — so at
//! mount time we expand the per-channel salience into a dense
//! per-weight vector per slot. That's the in-memory cost; on disk
//! the cover stores a compact per-slot `Vec<f32>` (which is what
//! `Filesystem::commit_salience` persists).
//!
//! ## Wire format
//!
//! ```text
//! offset size  field
//! 0      4     version = 1            (u32 LE)
//! 4      4     total_slot_count       (u32 LE) — size of the dense per_slot vec
//! 8      4     populated_slot_count   (u32 LE)
//! …      —     per populated slot:
//!                u16 slot_idx
//!                u16 reserved (zero)
//!                u32 weight_count
//!                f32 × weight_count   (LE)
//! ```
//!
//! Empty / un-calibrated slots are implied by omission: any slot
//! index not in the populated list has `per_slot[idx] = None` and
//! [`SalienceTable::max_over_range`] returns `0.0` — which is the
//! neutral value for the allocator's compound `FitKey`.

use thiserror::Error;

pub const VERSION: u32 = 1;

/// Header bytes preceding the populated-slot entries.
/// = 4 (version) + 4 (total_slot_count) + 4 (populated_slot_count).
pub const HEADER_BYTES: usize = 12;

/// Size of one populated-slot entry's fixed prefix (before the f32 values).
/// = 2 (slot_idx) + 2 (reserved) + 4 (weight_count).
const SLOT_HEADER_BYTES: usize = 8;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SalienceTable {
    /// Per-slot dense salience, indexed by slot_idx. `None` means no
    /// salience data for that slot (uncalibrated). `Some(v)` holds
    /// one `f32` per weight in the slot.
    per_slot: Vec<Option<Vec<f32>>>,
}

#[derive(Debug, Error, PartialEq)]
pub enum SalienceError {
    #[error("truncated salience record: need {need} bytes, got {got}")]
    Truncated { need: usize, got: usize },

    #[error("unsupported salience version: {0}")]
    UnsupportedVersion(u32),

    #[error("slot_idx {slot} out of range (total_slot_count = {total})")]
    SlotOutOfRange { slot: u16, total: u32 },

    #[error("duplicate slot_idx {slot} in salience record")]
    DuplicateSlot { slot: u16 },

    #[error("weight_count {count} exceeds remaining buffer ({remaining} bytes)")]
    WeightCountOverflow { count: u32, remaining: usize },

    #[error("non-finite salience value at slot {slot}, index {index}: {value}")]
    NonFinite { slot: u16, index: usize, value: f32 },

    #[error(
        "negative salience value at slot {slot}, index {index}: {value} \
         (salience is |x| — must be ≥ 0)"
    )]
    NegativeValue { slot: u16, index: usize, value: f32 },
}

impl SalienceTable {
    pub const fn empty() -> Self {
        Self { per_slot: Vec::new() }
    }

    /// Build from a pre-sized per-slot vector. Each entry is
    /// `None` if the slot has no salience, `Some(values)` otherwise.
    /// Values must be finite and ≥ 0 — callers should have finalized
    /// `AwqCollector` output.
    pub fn new(per_slot: Vec<Option<Vec<f32>>>) -> Self {
        Self { per_slot }
    }

    /// Number of slot entries (populated or not).
    pub fn slot_count(&self) -> usize {
        self.per_slot.len()
    }

    /// Number of slots with populated salience.
    pub fn populated_slot_count(&self) -> usize {
        self.per_slot.iter().filter(|s| s.is_some()).count()
    }

    /// `max(salience[start..start+len])` for the given slot. Returns
    /// `0.0` for a null slot or an out-of-range request — the
    /// neutral value for the allocator's salience-secondary FitKey.
    pub fn max_over_range(&self, slot: u32, start: u64, len: u64) -> f32 {
        let Some(Some(values)) = self.per_slot.get(slot as usize) else {
            return 0.0;
        };
        let start = start as usize;
        let len = len as usize;
        if start >= values.len() || len == 0 {
            return 0.0;
        }
        let end = (start + len).min(values.len());
        let mut m = 0.0_f32;
        for &v in &values[start..end] {
            if v > m {
                m = v;
            }
        }
        m
    }

    /// Read the salience for one specific weight. `0.0` for null
    /// slots or out-of-range indices.
    pub fn get(&self, slot: u32, weight_index: u64) -> f32 {
        let Some(Some(values)) = self.per_slot.get(slot as usize) else {
            return 0.0;
        };
        values.get(weight_index as usize).copied().unwrap_or(0.0)
    }

    /// True if every slot is `None` — i.e. the cover has no
    /// calibration data. Allocator uses this to short-circuit the
    /// salience branch of the FitKey.
    pub fn is_uncalibrated(&self) -> bool {
        self.per_slot.iter().all(|s| s.is_none())
    }

    /// Serialize to the wire format.
    pub fn encode(&self) -> Vec<u8> {
        let populated: usize = self.populated_slot_count();
        let values_bytes: usize = self
            .per_slot
            .iter()
            .map(|s| s.as_ref().map_or(0, |v| SLOT_HEADER_BYTES + 4 * v.len()))
            .sum();
        let mut out = Vec::with_capacity(HEADER_BYTES + values_bytes);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(self.per_slot.len() as u32).to_le_bytes());
        out.extend_from_slice(&(populated as u32).to_le_bytes());
        for (idx, entry) in self.per_slot.iter().enumerate() {
            let Some(values) = entry else { continue };
            out.extend_from_slice(&(idx as u16).to_le_bytes());
            out.extend_from_slice(&[0u8; 2]); // reserved
            out.extend_from_slice(&(values.len() as u32).to_le_bytes());
            for v in values {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    /// Deserialize. Returns a `SalienceTable` with `per_slot` sized
    /// to the stored `total_slot_count`, populated at the slot
    /// indexes recorded on the wire.
    pub fn decode(bytes: &[u8]) -> Result<Self, SalienceError> {
        if bytes.len() < HEADER_BYTES {
            return Err(SalienceError::Truncated {
                need: HEADER_BYTES,
                got: bytes.len(),
            });
        }
        let version = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if version != VERSION {
            return Err(SalienceError::UnsupportedVersion(version));
        }
        let total = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let populated = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let mut per_slot: Vec<Option<Vec<f32>>> = vec![None; total as usize];

        let mut offset = HEADER_BYTES;
        for _ in 0..populated {
            if offset + SLOT_HEADER_BYTES > bytes.len() {
                return Err(SalienceError::Truncated {
                    need: offset + SLOT_HEADER_BYTES,
                    got: bytes.len(),
                });
            }
            let slot_idx = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
            // bytes[offset+2..offset+4] reserved, ignored
            let weight_count =
                u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
            offset += SLOT_HEADER_BYTES;

            if (slot_idx as u32) >= total {
                return Err(SalienceError::SlotOutOfRange {
                    slot: slot_idx,
                    total,
                });
            }
            if per_slot[slot_idx as usize].is_some() {
                return Err(SalienceError::DuplicateSlot { slot: slot_idx });
            }

            let need_bytes = (weight_count as usize).checked_mul(4).ok_or(
                SalienceError::WeightCountOverflow {
                    count: weight_count,
                    remaining: bytes.len().saturating_sub(offset),
                },
            )?;
            if offset + need_bytes > bytes.len() {
                return Err(SalienceError::WeightCountOverflow {
                    count: weight_count,
                    remaining: bytes.len() - offset,
                });
            }
            let mut values = Vec::with_capacity(weight_count as usize);
            for i in 0..weight_count as usize {
                let lo = offset + i * 4;
                let v = f32::from_le_bytes(bytes[lo..lo + 4].try_into().unwrap());
                if !v.is_finite() {
                    return Err(SalienceError::NonFinite {
                        slot: slot_idx,
                        index: i,
                        value: v,
                    });
                }
                if v < 0.0 {
                    return Err(SalienceError::NegativeValue {
                        slot: slot_idx,
                        index: i,
                        value: v,
                    });
                }
                values.push(v);
            }
            offset += need_bytes;
            per_slot[slot_idx as usize] = Some(values);
        }
        Ok(Self { per_slot })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_table_encodes_to_header_only() {
        let t = SalienceTable::empty();
        let bytes = t.encode();
        assert_eq!(bytes.len(), HEADER_BYTES);
        let round = SalienceTable::decode(&bytes).unwrap();
        assert!(round.is_uncalibrated());
        assert_eq!(round.slot_count(), 0);
    }

    #[test]
    fn round_trips_populated_and_null_slots() {
        let t = SalienceTable::new(vec![
            Some(vec![1.0, 2.5, 3.5]),
            None,
            Some(vec![0.0, 0.25]),
            None,
        ]);
        let bytes = t.encode();
        let round = SalienceTable::decode(&bytes).unwrap();
        assert_eq!(round, t);
    }

    #[test]
    fn max_over_range_returns_max_inside_window() {
        let t = SalienceTable::new(vec![Some(vec![1.0, 5.0, 2.0, 0.5, 3.0])]);
        assert_eq!(t.max_over_range(0, 0, 5), 5.0);
        assert_eq!(t.max_over_range(0, 2, 3), 3.0);
        assert_eq!(t.max_over_range(0, 3, 1), 0.5);
        // Out-of-range request → 0.0.
        assert_eq!(t.max_over_range(0, 10, 5), 0.0);
        assert_eq!(t.max_over_range(99, 0, 5), 0.0);
    }

    #[test]
    fn uncalibrated_slot_is_zero() {
        let t = SalienceTable::new(vec![None, Some(vec![1.0])]);
        assert_eq!(t.max_over_range(0, 0, 1), 0.0);
        assert_eq!(t.max_over_range(1, 0, 1), 1.0);
        assert!(!t.is_uncalibrated());
    }

    #[test]
    fn is_uncalibrated_true_for_all_none() {
        let t = SalienceTable::new(vec![None, None, None]);
        assert!(t.is_uncalibrated());
    }

    #[test]
    fn decode_rejects_bad_version() {
        let mut bytes = SalienceTable::empty().encode();
        bytes[0] = 99;
        assert!(matches!(
            SalienceTable::decode(&bytes),
            Err(SalienceError::UnsupportedVersion(_)),
        ));
    }

    #[test]
    fn decode_rejects_truncated_header() {
        let bytes = [0u8; 4];
        assert!(matches!(
            SalienceTable::decode(&bytes),
            Err(SalienceError::Truncated { .. }),
        ));
    }

    #[test]
    fn decode_rejects_slot_out_of_range() {
        // Manually craft a record: total_slot_count = 1, populated = 1,
        // slot_idx = 5 (out of range).
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1_u32.to_le_bytes()); // version
        bytes.extend_from_slice(&1_u32.to_le_bytes()); // total
        bytes.extend_from_slice(&1_u32.to_le_bytes()); // populated
        bytes.extend_from_slice(&5_u16.to_le_bytes()); // slot_idx
        bytes.extend_from_slice(&[0u8; 2]); // reserved
        bytes.extend_from_slice(&0_u32.to_le_bytes()); // weight_count
        assert!(matches!(
            SalienceTable::decode(&bytes),
            Err(SalienceError::SlotOutOfRange { slot: 5, total: 1 }),
        ));
    }

    #[test]
    fn decode_rejects_duplicate_slot() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1_u32.to_le_bytes()); // version
        bytes.extend_from_slice(&3_u32.to_le_bytes()); // total
        bytes.extend_from_slice(&2_u32.to_le_bytes()); // populated

        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 2]);
        bytes.extend_from_slice(&0_u32.to_le_bytes());

        bytes.extend_from_slice(&0_u16.to_le_bytes()); // duplicate
        bytes.extend_from_slice(&[0u8; 2]);
        bytes.extend_from_slice(&0_u32.to_le_bytes());

        assert!(matches!(
            SalienceTable::decode(&bytes),
            Err(SalienceError::DuplicateSlot { slot: 0 }),
        ));
    }

    #[test]
    fn decode_rejects_non_finite() {
        let t = SalienceTable::new(vec![Some(vec![f32::NAN])]);
        // Encode normally.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 2]);
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&t.per_slot[0].as_ref().unwrap()[0].to_le_bytes());
        assert!(matches!(
            SalienceTable::decode(&bytes),
            Err(SalienceError::NonFinite { .. }),
        ));
    }

    #[test]
    fn decode_rejects_negative() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 2]);
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&(-1.0_f32).to_le_bytes());
        assert!(matches!(
            SalienceTable::decode(&bytes),
            Err(SalienceError::NegativeValue { .. }),
        ));
    }

    #[test]
    fn decode_rejects_truncated_values() {
        // Declare 4 weights but only supply 2.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 2]);
        bytes.extend_from_slice(&4_u32.to_le_bytes()); // claim 4
        bytes.extend_from_slice(&1.0_f32.to_le_bytes()); // only 2 provided
        bytes.extend_from_slice(&2.0_f32.to_le_bytes());
        assert!(matches!(
            SalienceTable::decode(&bytes),
            Err(SalienceError::WeightCountOverflow { .. }),
        ));
    }
}
