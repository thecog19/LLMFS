//! Exact per-slot salience table + on-disk codec.
//!
//! B1's `AwqCollector` produces one salience value per **input
//! channel** of each linear tensor: `mean(|x_c|)`. In GGUF's
//! row-major `[out_dim, in_dim]` weight layout, that channel vector
//! repeats with period `in_dim` across the whole weight tensor:
//! `weight[i] -> channel[i % in_dim]`.
//!
//! The first B4 implementation expanded that periodic vector into a
//! dense per-weight `Vec<f32>` per slot. That preserved exact
//! ordering, but it also exploded storage and mount-time RAM. The
//! current format stores the period directly and answers
//! `max_over_range` queries against the repeating pattern exactly,
//! without materialising `f32 × weight_count`.
//!
//! Covers calibrated with the old dense format remain readable:
//! version 1 records decode into a legacy in-memory representation,
//! while new writes use version 2's periodic encoding.
//!
//! ## Wire format
//!
//! ```text
//! version 2:
//! offset size  field
//! 0      4     version = 2            (u32 LE)
//! 4      4     total_slot_count       (u32 LE)
//! 8      4     populated_slot_count   (u32 LE)
//! …      —     per populated slot:
//!                u16 slot_idx
//!                u16 reserved (zero)
//!                u32 weight_count
//!                u32 channel_count
//!                f32 × channel_count  (LE)
//!
//! version 1 (legacy dense, still accepted on decode):
//! offset size  field
//! 0      4     version = 1            (u32 LE)
//! 4      4     total_slot_count       (u32 LE)
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

pub const VERSION: u32 = 2;
const LEGACY_VERSION: u32 = 1;

/// Header bytes preceding the populated-slot entries.
/// = 4 (version) + 4 (total_slot_count) + 4 (populated_slot_count).
pub const HEADER_BYTES: usize = 12;

/// Size of one populated-slot entry's fixed prefix in version 2
/// (before the f32 channel values).
/// = 2 (slot_idx) + 2 (reserved) + 4 (weight_count) + 4 (channel_count).
const SLOT_HEADER_BYTES: usize = 12;

/// Size of one populated-slot entry's fixed prefix in legacy
/// version 1 (before the dense per-weight f32 values).
const LEGACY_SLOT_HEADER_BYTES: usize = 8;

#[derive(Debug, Clone, PartialEq)]
pub struct SalienceTable {
    /// Per-slot salience, indexed by slot_idx. `None` means no
    /// salience data for that slot (uncalibrated).
    per_slot: Vec<Option<SlotSalience>>,
}

impl Default for SalienceTable {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Debug, Clone, PartialEq)]
enum SlotSalience {
    Periodic(PeriodicSlotSalience),
    Dense(LegacyDenseSlotSalience),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PeriodicSlotSalience {
    weight_count: usize,
    period: Vec<f32>,
    max_value: f32,
}

#[derive(Debug, Clone, PartialEq)]
struct LegacyDenseSlotSalience {
    values: Vec<f32>,
    max_value: f32,
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

    #[error("channel_count {count} exceeds remaining buffer ({remaining} bytes)")]
    ChannelCountOverflow { count: u32, remaining: usize },

    #[error("non-finite salience value at slot {slot}, index {index}: {value}")]
    NonFinite { slot: u16, index: usize, value: f32 },

    #[error(
        "negative salience value at slot {slot}, index {index}: {value} \
         (salience is |x| — must be ≥ 0)"
    )]
    NegativeValue { slot: u16, index: usize, value: f32 },

    #[error(
        "periodic salience slot with {weight_count} weights must have at least one channel value"
    )]
    EmptyPeriodicSlot { weight_count: u64 },

    #[error("periodic salience weight_count {count} exceeds platform limits")]
    PeriodicWeightCountTooLarge { count: u64 },

    #[error("non-finite periodic salience value at index {index}: {value}")]
    PeriodicNonFinite { index: usize, value: f32 },

    #[error("negative periodic salience value at index {index}: {value}")]
    PeriodicNegative { index: usize, value: f32 },
}

impl PeriodicSlotSalience {
    /// Exact periodic salience for one slot: `period[i]` applies to
    /// every weight whose index is congruent to `i mod period.len()`.
    pub fn new(weight_count: u64, period: Vec<f32>) -> Result<Self, SalienceError> {
        if weight_count > 0 && period.is_empty() {
            return Err(SalienceError::EmptyPeriodicSlot { weight_count });
        }
        let weight_count = usize::try_from(weight_count).map_err(|_| {
            SalienceError::PeriodicWeightCountTooLarge {
                count: weight_count,
            }
        })?;
        let mut max_value = 0.0_f32;
        for (index, value) in period.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(SalienceError::PeriodicNonFinite { index, value });
            }
            if value < 0.0 {
                return Err(SalienceError::PeriodicNegative { index, value });
            }
            if value > max_value {
                max_value = value;
            }
        }
        Ok(Self {
            weight_count,
            period,
            max_value,
        })
    }

    fn max_over_range(&self, start: u64, len: u64) -> f32 {
        if len == 0 {
            return 0.0;
        }
        let start = start as usize;
        if start >= self.weight_count {
            return 0.0;
        }
        let len = len.min((self.weight_count - start) as u64) as usize;
        let period_len = self.period.len();
        if len >= period_len {
            return self.max_value;
        }

        let mut max_value = 0.0_f32;
        let mut period_index = start % period_len;
        for _ in 0..len {
            let value = self.period[period_index];
            if value > max_value {
                max_value = value;
            }
            period_index += 1;
            if period_index == period_len {
                period_index = 0;
            }
        }
        max_value
    }

    fn get(&self, weight_index: u64) -> f32 {
        let weight_index = weight_index as usize;
        if weight_index >= self.weight_count || self.period.is_empty() {
            return 0.0;
        }
        self.period[weight_index % self.period.len()]
    }
}

impl LegacyDenseSlotSalience {
    fn new(slot: u16, values: Vec<f32>) -> Result<Self, SalienceError> {
        let mut max_value = 0.0_f32;
        for (index, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(SalienceError::NonFinite { slot, index, value });
            }
            if value < 0.0 {
                return Err(SalienceError::NegativeValue { slot, index, value });
            }
            if value > max_value {
                max_value = value;
            }
        }
        Ok(Self { values, max_value })
    }
}

impl SlotSalience {
    fn weight_count(&self) -> usize {
        match self {
            Self::Periodic(slot) => slot.weight_count,
            Self::Dense(slot) => slot.values.len(),
        }
    }

    fn max_over_range(&self, start: u64, len: u64) -> f32 {
        match self {
            Self::Periodic(slot) => slot.max_over_range(start, len),
            Self::Dense(slot) => {
                let start = start as usize;
                let len = len as usize;
                if start >= slot.values.len() || len == 0 {
                    return 0.0;
                }
                let end = (start + len).min(slot.values.len());
                let mut max_value = 0.0_f32;
                for &value in &slot.values[start..end] {
                    if value > max_value {
                        max_value = value;
                    }
                }
                max_value
            }
        }
    }

    fn get(&self, weight_index: u64) -> f32 {
        match self {
            Self::Periodic(slot) => slot.get(weight_index),
            Self::Dense(slot) => slot
                .values
                .get(weight_index as usize)
                .copied()
                .unwrap_or(0.0),
        }
    }

    fn encoded_v2_bytes(&self) -> usize {
        match self {
            Self::Periodic(slot) => SLOT_HEADER_BYTES + 4 * slot.period.len(),
            Self::Dense(_) => 0,
        }
    }

    fn encoded_v1_bytes(&self) -> usize {
        LEGACY_SLOT_HEADER_BYTES + 4 * self.weight_count()
    }

    fn encode_v2(&self, slot_idx: usize, out: &mut Vec<u8>) {
        let Self::Periodic(slot) = self else {
            unreachable!("legacy dense slots are encoded through the v1 path");
        };
        out.extend_from_slice(&(slot_idx as u16).to_le_bytes());
        out.extend_from_slice(&[0u8; 2]);
        out.extend_from_slice(&(slot.weight_count as u32).to_le_bytes());
        out.extend_from_slice(&(slot.period.len() as u32).to_le_bytes());
        for value in &slot.period {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn encode_v1(&self, slot_idx: usize, out: &mut Vec<u8>) {
        out.extend_from_slice(&(slot_idx as u16).to_le_bytes());
        out.extend_from_slice(&[0u8; 2]);
        out.extend_from_slice(&(self.weight_count() as u32).to_le_bytes());
        match self {
            Self::Dense(slot) => {
                for value in &slot.values {
                    out.extend_from_slice(&value.to_le_bytes());
                }
            }
            Self::Periodic(slot) => {
                for weight_index in 0..slot.weight_count {
                    out.extend_from_slice(&slot.get(weight_index as u64).to_le_bytes());
                }
            }
        }
    }
}

impl SalienceTable {
    pub const fn empty() -> Self {
        Self {
            per_slot: Vec::new(),
        }
    }

    /// Build from a pre-sized per-slot vector. Each populated entry
    /// stores the exact repeating channel-salience period for that
    /// slot.
    pub fn new(per_slot: Vec<Option<PeriodicSlotSalience>>) -> Self {
        Self {
            per_slot: per_slot
                .into_iter()
                .map(|slot| slot.map(SlotSalience::Periodic))
                .collect(),
        }
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
        let Some(Some(slot_salience)) = self.per_slot.get(slot as usize) else {
            return 0.0;
        };
        slot_salience.max_over_range(start, len)
    }

    /// Read the salience for one specific weight. `0.0` for null
    /// slots or out-of-range indices.
    pub fn get(&self, slot: u32, weight_index: u64) -> f32 {
        let Some(Some(slot_salience)) = self.per_slot.get(slot as usize) else {
            return 0.0;
        };
        slot_salience.get(weight_index)
    }

    /// True if every slot is `None` — i.e. the cover has no
    /// calibration data. Allocator uses this to short-circuit the
    /// salience branch of the FitKey.
    pub fn is_uncalibrated(&self) -> bool {
        self.per_slot.iter().all(|s| s.is_none())
    }

    /// Serialize to the wire format.
    pub fn encode(&self) -> Vec<u8> {
        if self
            .per_slot
            .iter()
            .flatten()
            .any(|slot| matches!(slot, SlotSalience::Dense(_)))
        {
            self.encode_v1_dense()
        } else {
            self.encode_v2_periodic()
        }
    }

    fn encode_v2_periodic(&self) -> Vec<u8> {
        let populated = self.populated_slot_count();
        let values_bytes: usize = self
            .per_slot
            .iter()
            .map(|slot| slot.as_ref().map_or(0, SlotSalience::encoded_v2_bytes))
            .sum();
        let mut out = Vec::with_capacity(HEADER_BYTES + values_bytes);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(self.per_slot.len() as u32).to_le_bytes());
        out.extend_from_slice(&(populated as u32).to_le_bytes());
        for (idx, entry) in self.per_slot.iter().enumerate() {
            let Some(slot) = entry else { continue };
            slot.encode_v2(idx, &mut out);
        }
        out
    }

    fn encode_v1_dense(&self) -> Vec<u8> {
        let populated = self.populated_slot_count();
        let values_bytes: usize = self
            .per_slot
            .iter()
            .map(|slot| slot.as_ref().map_or(0, SlotSalience::encoded_v1_bytes))
            .sum();
        let mut out = Vec::with_capacity(HEADER_BYTES + values_bytes);
        out.extend_from_slice(&LEGACY_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.per_slot.len() as u32).to_le_bytes());
        out.extend_from_slice(&(populated as u32).to_le_bytes());
        for (idx, entry) in self.per_slot.iter().enumerate() {
            let Some(slot) = entry else { continue };
            slot.encode_v1(idx, &mut out);
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
        match version {
            LEGACY_VERSION => Self::decode_v1_dense(bytes),
            VERSION => Self::decode_v2_periodic(bytes),
            _ => Err(SalienceError::UnsupportedVersion(version)),
        }
    }

    fn decode_v1_dense(bytes: &[u8]) -> Result<Self, SalienceError> {
        let total = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let populated = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let mut per_slot: Vec<Option<SlotSalience>> = vec![None; total as usize];

        let mut offset = HEADER_BYTES;
        for _ in 0..populated {
            if offset + LEGACY_SLOT_HEADER_BYTES > bytes.len() {
                return Err(SalienceError::Truncated {
                    need: offset + LEGACY_SLOT_HEADER_BYTES,
                    got: bytes.len(),
                });
            }
            let slot_idx = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
            // bytes[offset+2..offset+4] reserved, ignored
            let weight_count =
                u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
            offset += LEGACY_SLOT_HEADER_BYTES;

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
                values.push(v);
            }
            offset += need_bytes;
            per_slot[slot_idx as usize] = Some(SlotSalience::Dense(LegacyDenseSlotSalience::new(
                slot_idx, values,
            )?));
        }
        Ok(Self { per_slot })
    }

    fn decode_v2_periodic(bytes: &[u8]) -> Result<Self, SalienceError> {
        let total = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let populated = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let mut per_slot: Vec<Option<SlotSalience>> = vec![None; total as usize];

        let mut offset = HEADER_BYTES;
        for _ in 0..populated {
            if offset + SLOT_HEADER_BYTES > bytes.len() {
                return Err(SalienceError::Truncated {
                    need: offset + SLOT_HEADER_BYTES,
                    got: bytes.len(),
                });
            }
            let slot_idx = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
            let weight_count =
                u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
            let channel_count =
                u32::from_le_bytes(bytes[offset + 8..offset + 12].try_into().unwrap());
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

            let need_bytes = (channel_count as usize).checked_mul(4).ok_or(
                SalienceError::ChannelCountOverflow {
                    count: channel_count,
                    remaining: bytes.len().saturating_sub(offset),
                },
            )?;
            if offset + need_bytes > bytes.len() {
                return Err(SalienceError::ChannelCountOverflow {
                    count: channel_count,
                    remaining: bytes.len() - offset,
                });
            }

            let mut period = Vec::with_capacity(channel_count as usize);
            for i in 0..channel_count as usize {
                let lo = offset + i * 4;
                let value = f32::from_le_bytes(bytes[lo..lo + 4].try_into().unwrap());
                if !value.is_finite() {
                    return Err(SalienceError::NonFinite {
                        slot: slot_idx,
                        index: i,
                        value,
                    });
                }
                if value < 0.0 {
                    return Err(SalienceError::NegativeValue {
                        slot: slot_idx,
                        index: i,
                        value,
                    });
                }
                period.push(value);
            }
            offset += need_bytes;
            per_slot[slot_idx as usize] = Some(SlotSalience::Periodic(PeriodicSlotSalience::new(
                weight_count as u64,
                period,
            )?));
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
            Some(PeriodicSlotSalience::new(9, vec![1.0, 2.5, 3.5]).unwrap()),
            None,
            Some(PeriodicSlotSalience::new(4, vec![0.0, 0.25]).unwrap()),
            None,
        ]);
        let bytes = t.encode();
        let round = SalienceTable::decode(&bytes).unwrap();
        assert_eq!(round, t);
    }

    #[test]
    fn max_over_range_is_exact_for_periodic_slots() {
        let t = SalienceTable::new(vec![Some(
            PeriodicSlotSalience::new(12, vec![1.0, 5.0, 2.0, 0.5, 3.0]).unwrap(),
        )]);
        assert_eq!(t.max_over_range(0, 0, 5), 5.0);
        assert_eq!(t.max_over_range(0, 2, 3), 3.0);
        assert_eq!(t.max_over_range(0, 3, 1), 0.5);
        assert_eq!(t.max_over_range(0, 4, 3), 5.0);
        assert_eq!(t.max_over_range(0, 5, 5), 5.0);
        assert_eq!(t.max_over_range(0, 7, 6), 5.0);
        // Out-of-range request → 0.0.
        assert_eq!(t.max_over_range(0, 20, 5), 0.0);
        assert_eq!(t.max_over_range(99, 0, 5), 0.0);
    }

    #[test]
    fn periodic_encoding_stores_only_one_period() {
        let t = SalienceTable::new(vec![Some(
            PeriodicSlotSalience::new(6, vec![1.0, 2.0, 3.0]).unwrap(),
        )]);
        let bytes = t.encode();
        assert_eq!(bytes.len(), HEADER_BYTES + SLOT_HEADER_BYTES + 3 * 4);
    }

    #[test]
    fn uncalibrated_slot_is_zero() {
        let t = SalienceTable::new(vec![
            None,
            Some(PeriodicSlotSalience::new(4, vec![1.0]).unwrap()),
        ]);
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
    fn decode_v1_dense_payload_remains_supported() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1_u32.to_le_bytes()); // version 1 dense
        bytes.extend_from_slice(&1_u32.to_le_bytes()); // total
        bytes.extend_from_slice(&1_u32.to_le_bytes()); // populated
        bytes.extend_from_slice(&0_u16.to_le_bytes()); // slot_idx
        bytes.extend_from_slice(&[0u8; 2]);
        bytes.extend_from_slice(&6_u32.to_le_bytes()); // weight_count
        for v in [1.0_f32, 2.0, 3.0, 1.0, 2.0, 3.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        let table = SalienceTable::decode(&bytes).unwrap();
        assert_eq!(table.max_over_range(0, 0, 6), 3.0);
        assert_eq!(table.max_over_range(0, 4, 2), 3.0);
        assert_eq!(table.get(0, 5), 3.0);
    }

    #[test]
    fn decode_rejects_non_finite() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 2]);
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&f32::NAN.to_le_bytes());
        assert!(matches!(
            SalienceTable::decode(&bytes),
            Err(SalienceError::NonFinite { .. }),
        ));
    }

    #[test]
    fn decode_rejects_negative() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 2]);
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&(-1.0_f32).to_le_bytes());
        assert!(matches!(
            SalienceTable::decode(&bytes),
            Err(SalienceError::NegativeValue { .. }),
        ));
    }

    #[test]
    fn decode_rejects_truncated_values() {
        // Declare 4 channels but only supply 2.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 2]);
        bytes.extend_from_slice(&4_u32.to_le_bytes()); // weight_count
        bytes.extend_from_slice(&4_u32.to_le_bytes()); // claim 4 channels
        bytes.extend_from_slice(&1.0_f32.to_le_bytes()); // only 2 provided
        bytes.extend_from_slice(&2.0_f32.to_le_bytes());
        assert!(matches!(
            SalienceTable::decode(&bytes),
            Err(SalienceError::ChannelCountOverflow { .. }),
        ));
    }
}
