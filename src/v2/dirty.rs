//! Dirty-weight bitmap (DESIGN-NEW §15.5 allocation priority 2).
//!
//! One bit per eligible weight in the cover. A bit is **set** iff
//! that weight's stealable bits have been written to at some point
//! — "dirty" in the "has been perturbed" sense. Never cleared:
//! once perturbed, always perturbed (the cover can't go back to
//! pristine).
//!
//! The allocator prefers runs whose first-N weights are already
//! dirty over pristine runs of the same ceiling magnitude —
//! writing to already-perturbed weights is cheaper than adding new
//! perturbation to still-pristine ones. See
//! `Allocator::alloc_preferring_dirty`.
//!
//! **Persistence.** In-memory only in this milestone (step 10a).
//! Cross-session dirty state is rebuilt at mount by marking every
//! weight the current inode transitively references (those were
//! definitely written). Positions used by past sessions but no
//! longer referenced lose their dirty status — step 10b will
//! persist the bitmap via `super_root.dirty_bitmap_inode` and
//! close that gap.

use thiserror::Error;

use crate::stego::tensor_map::TensorMap;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DirtyBitmapError {
    #[error("bitmap byte count mismatch: map expects {expected} bytes, deserialize got {got}")]
    ByteCountMismatch { expected: usize, got: usize },
}

/// Packed bit-per-weight dirty bitmap. Per-slot bit offsets are
/// computed from `TensorMap::slots` — slot 0's weights occupy bits
/// `0..slot0.weight_count`, slot 1's continue from there, etc.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirtyBitmap {
    /// Packed bits, LSB-first within each byte.
    bits: Vec<u8>,
    /// `slot_offsets[slot_index]` = starting bit index for that slot.
    slot_offsets: Vec<u64>,
    total_bits: u64,
}

impl DirtyBitmap {
    /// Build an all-zero bitmap sized for every eligible weight in
    /// `map`. Non-stealable slots still get a 0-length offset entry
    /// so `slot_offsets[i]` is always valid.
    pub fn new(map: &TensorMap) -> Self {
        let mut slot_offsets = Vec::with_capacity(map.slots.len());
        let mut total: u64 = 0;
        for slot in &map.slots {
            slot_offsets.push(total);
            total = total.saturating_add(slot.weight_count);
        }
        let byte_len = total.div_ceil(8) as usize;
        Self {
            bits: vec![0; byte_len],
            slot_offsets,
            total_bits: total,
        }
    }

    /// Total number of weights this bitmap can track.
    pub fn total_bits(&self) -> u64 {
        self.total_bits
    }

    /// Number of bits currently set (i.e. weights ever marked dirty).
    pub fn set_count(&self) -> u64 {
        self.bits.iter().map(|b| b.count_ones() as u64).sum()
    }

    /// Is `(slot, weight_index)` marked dirty?
    pub fn is_dirty(&self, slot: u16, weight_index: u32) -> bool {
        let bit = self.bit_index(slot, weight_index);
        let byte = (bit / 8) as usize;
        let shift = (bit % 8) as u8;
        self.bits.get(byte).is_some_and(|b| b & (1 << shift) != 0)
    }

    /// Mark `(slot, weight_index)` dirty. Idempotent.
    pub fn mark(&mut self, slot: u16, weight_index: u32) {
        let bit = self.bit_index(slot, weight_index);
        let byte = (bit / 8) as usize;
        let shift = (bit % 8) as u8;
        if byte < self.bits.len() {
            self.bits[byte] |= 1 << shift;
        }
    }

    /// Mark every weight in `[start, start + len)` of `slot` dirty.
    /// O(len) at current simplicity; a byte-stride implementation
    /// could be ~8× faster but isn't needed yet (callers mark
    /// chunks of a few hundred weights at most).
    pub fn mark_range(&mut self, slot: u16, start: u32, len: u32) {
        for i in 0..len {
            self.mark(slot, start + i);
        }
    }

    /// True iff every weight in `[start, start + len)` of `slot` is
    /// already marked dirty. Zero-length range is vacuously true.
    pub fn is_range_dirty(&self, slot: u16, start: u32, len: u32) -> bool {
        for i in 0..len {
            if !self.is_dirty(slot, start + i) {
                return false;
            }
        }
        true
    }

    fn bit_index(&self, slot: u16, weight_index: u32) -> u64 {
        self.slot_offsets[slot as usize] + weight_index as u64
    }

    /// Serialise the packed bit array. The map's slot layout is what
    /// tells a decoder how to re-derive slot offsets, so only the
    /// raw bytes are persisted — no header.
    pub fn serialize(&self) -> Vec<u8> {
        self.bits.clone()
    }

    /// Reconstruct a `DirtyBitmap` from persisted bytes + the
    /// cover's `TensorMap` (which supplies per-slot offsets). Errors
    /// if the byte count doesn't match what the map implies —
    /// either the map changed between sessions (cover was reshaped,
    /// shouldn't happen) or the persistence was corrupted.
    pub fn deserialize(bytes: &[u8], map: &TensorMap) -> Result<Self, DirtyBitmapError> {
        let fresh = Self::new(map);
        if bytes.len() != fresh.bits.len() {
            return Err(DirtyBitmapError::ByteCountMismatch {
                expected: fresh.bits.len(),
                got: bytes.len(),
            });
        }
        Ok(Self {
            bits: bytes.to_vec(),
            ..fresh
        })
    }
}
