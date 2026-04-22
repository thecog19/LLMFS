//! Sparse dirty-weight bitmap (DESIGN-NEW §15.5 allocation priority 2).
//!
//! One bit per eligible weight in the cover. A bit is **set** iff
//! that weight's stealable bits have been written to at some point —
//! "dirty" in the "has been perturbed" sense. Never cleared: once
//! perturbed, always perturbed.
//!
//! The allocator prefers runs whose first-N weights are already
//! dirty over pristine runs of the same ceiling magnitude — writing
//! to already-perturbed weights is cheaper than adding new
//! perturbation. See `Allocator::alloc_preferring_dirty`.
//!
//! # Memory layout
//!
//! Storage is sparse: a `BTreeMap<u32, Box<[u8; PAGE_BYTES]>>` keyed
//! by page index. A "page" is `PAGE_BYTES` worth of bits
//! (`PAGE_BYTES * 8` weights, default 32 768). Pages are allocated
//! lazily on first `mark` and never freed within a session.
//! Pages that are entirely zero never allocate at all — `is_dirty`
//! returns `false` for them, `is_range_dirty` returns `false`, and
//! `set_count` skips them.
//!
//! For a 280 GB F16 cover (~140 G weights, ~17.5 GB if dense), an
//! all-zero bitmap costs ~bytes; a bitmap with 1 % set bits
//! distributed across the cover costs at most
//! `set_pages × PAGE_BYTES`. This is the structural fix for the
//! "can't even mount the cover" RAM ceiling.
//!
//! # Persistence
//!
//! [`Self::write_to`] streams the dense bit array to a `Write`
//! sink one page at a time, emitting zeros for missing pages
//! without ever materialising the full byte array. [`Self::write_bytes_at`]
//! performs the inverse — write a slice of dense bytes at an
//! absolute offset, allocating pages only when the source bytes are
//! non-zero. [`crate::v2::fs`] uses these to round-trip the bitmap
//! through the persisted V2 inode without ever holding a 17 GB
//! `Vec<u8>`.

use std::collections::BTreeMap;
use std::io::{self, Write};

use thiserror::Error;

use crate::stego::tensor_map::TensorMap;

/// Bytes per sparse page. 4 KB matches a typical OS page; gives
/// 32 768 weights per page.
pub const PAGE_BYTES: usize = 4096;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DirtyBitmapError {
    #[error("bitmap byte count mismatch: map expects {expected} bytes, source provided {got}")]
    ByteCountMismatch { expected: u64, got: u64 },
}

/// Sparse-page dirty-weight bitmap. Per-slot bit offsets are
/// computed from `TensorMap::slots` — slot 0's weights occupy bits
/// `0..slot0.weight_count`, slot 1's continue from there, etc.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DirtyBitmap {
    /// Sparse pages keyed by page index. Missing pages are
    /// implicitly all-zero — `is_dirty` returns `false`,
    /// `is_range_dirty` returns `false`, and `set_count` excludes
    /// them.
    pages: BTreeMap<u32, Box<[u8; PAGE_BYTES]>>,
    /// `slot_offsets[slot_index]` = starting bit index for that slot.
    slot_offsets: Vec<u64>,
    /// Total tracked bits across all slots.
    total_bits: u64,
}

impl DirtyBitmap {
    /// Build an empty bitmap sized for every eligible weight in
    /// `map`. **Allocates zero pages** — the bitmap occupies a
    /// constant ~bytes regardless of how many weights `map`
    /// describes. Pages are allocated lazily on the first `mark`
    /// or `write_bytes_at` that touches them.
    pub fn new(map: &TensorMap) -> Self {
        let mut slot_offsets = Vec::with_capacity(map.slots.len());
        let mut total: u64 = 0;
        for slot in &map.slots {
            slot_offsets.push(total);
            total = total.saturating_add(slot.weight_count);
        }
        Self {
            pages: BTreeMap::new(),
            slot_offsets,
            total_bits: total,
        }
    }

    /// Total number of weights this bitmap can track.
    pub fn total_bits(&self) -> u64 {
        self.total_bits
    }

    /// Total dense byte length of the bitmap (`ceil(total_bits / 8)`).
    /// This is what `write_to` would emit and what `write_bytes_at`
    /// should be filled with end-to-end.
    pub fn total_bytes(&self) -> u64 {
        self.total_bits.div_ceil(8)
    }

    /// Number of bits currently set across all allocated pages.
    /// Skips missing pages (they contribute zero by definition).
    pub fn set_count(&self) -> u64 {
        self.pages
            .values()
            .map(|p| p.iter().map(|b| b.count_ones() as u64).sum::<u64>())
            .sum()
    }

    /// Number of physical pages currently allocated. Test-only
    /// observability — exposed for the RAM-bound proof tests.
    pub fn allocated_page_count(&self) -> usize {
        self.pages.len()
    }

    /// Is `(slot, weight_index)` marked dirty?
    pub fn is_dirty(&self, slot: u16, weight_index: u32) -> bool {
        let bit = self.bit_index(slot, weight_index);
        let (page_idx, in_byte, shift) = bit_position(bit);
        match self.pages.get(&page_idx) {
            Some(p) => p[in_byte] & (1 << shift) != 0,
            None => false,
        }
    }

    /// Mark `(slot, weight_index)` dirty. Idempotent. Allocates the
    /// page lazily on first touch.
    pub fn mark(&mut self, slot: u16, weight_index: u32) {
        let bit = self.bit_index(slot, weight_index);
        if bit >= self.total_bits {
            return;
        }
        let (page_idx, in_byte, shift) = bit_position(bit);
        let page = self
            .pages
            .entry(page_idx)
            .or_insert_with(|| Box::new([0u8; PAGE_BYTES]));
        page[in_byte] |= 1 << shift;
    }

    /// Mark every weight in `[start, start + len)` of `slot` dirty.
    /// O(len) over the bit range.
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

    /// Borrow the page at `page_idx`, if allocated. Used by the V2
    /// fs's streaming persist to snapshot one page at a time without
    /// materialising the dense byte array.
    pub fn page_at(&self, page_idx: u32) -> Option<&[u8; PAGE_BYTES]> {
        self.pages.get(&page_idx).map(|b| b.as_ref())
    }

    /// Write `bytes` into the bitmap starting at absolute byte
    /// `offset`. Pages whose source slice is entirely zero (and
    /// don't already exist) are **not allocated** — that's how the
    /// streaming load path keeps RAM proportional to non-zero
    /// content rather than to total cover size.
    ///
    /// Panics if `offset + bytes.len() > total_bytes()`.
    pub fn write_bytes_at(&mut self, offset: u64, bytes: &[u8]) {
        let end = offset + bytes.len() as u64;
        assert!(
            end <= self.total_bytes(),
            "write_bytes_at: range [{offset}, {end}) exceeds bitmap byte length {}",
            self.total_bytes()
        );
        let mut consumed = 0_usize;
        while consumed < bytes.len() {
            let abs = offset + consumed as u64;
            let page_idx = (abs / PAGE_BYTES as u64) as u32;
            let in_page = (abs % PAGE_BYTES as u64) as usize;
            let page_remaining = PAGE_BYTES - in_page;
            let take = (bytes.len() - consumed).min(page_remaining);
            let src = &bytes[consumed..consumed + take];

            let any_nonzero = src.iter().any(|&b| b != 0);
            let exists = self.pages.contains_key(&page_idx);
            if exists || any_nonzero {
                let page = self
                    .pages
                    .entry(page_idx)
                    .or_insert_with(|| Box::new([0u8; PAGE_BYTES]));
                page[in_page..in_page + take].copy_from_slice(src);
            }
            consumed += take;
        }
    }

    /// Stream the dense byte form to `writer`, page by page. Missing
    /// pages emit zeros from a stack-resident buffer — never
    /// allocates the full bitmap as a `Vec<u8>`.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let total = self.total_bytes();
        let zeros = [0u8; PAGE_BYTES];
        let total_pages = total.div_ceil(PAGE_BYTES as u64) as u32;
        for idx in 0..total_pages {
            let page_first_byte = idx as u64 * PAGE_BYTES as u64;
            let bytes_in_page = (total - page_first_byte).min(PAGE_BYTES as u64) as usize;
            match self.pages.get(&idx) {
                Some(p) => writer.write_all(&p[..bytes_in_page])?,
                None => writer.write_all(&zeros[..bytes_in_page])?,
            }
        }
        Ok(())
    }

    fn bit_index(&self, slot: u16, weight_index: u32) -> u64 {
        self.slot_offsets[slot as usize] + weight_index as u64
    }
}

/// Decompose a bit index into (page index, byte within page, bit
/// within byte). Bits within a byte are LSB-first to match the V1
/// persistence convention; the high bit of each byte is bit 7 of
/// the byte's eight tracked weights.
fn bit_position(bit: u64) -> (u32, usize, u8) {
    let byte = bit / 8;
    let shift = (bit % 8) as u8;
    let page_idx = (byte / PAGE_BYTES as u64) as u32;
    let in_byte = (byte % PAGE_BYTES as u64) as usize;
    (page_idx, in_byte, shift)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::quant::GgufQuantType;
    use crate::stego::planner::TensorTier;
    use crate::stego::tensor_map::TensorSlot;

    fn f16_slot(weight_count: u64, name: &str) -> TensorSlot {
        TensorSlot {
            name: name.to_owned(),
            quant_type: GgufQuantType::F16,
            tier: TensorTier::Tier1,
            data_offset: 0,
            weight_count,
            stealable_bits_per_weight: GgufQuantType::F16.stealable_bits_hint(),
            capacity_bits: weight_count * GgufQuantType::F16.stealable_bits_hint() as u64,
            bit_start: 0,
            bit_end: weight_count * GgufQuantType::F16.stealable_bits_hint() as u64,
        }
    }

    fn one_slot_map(weights: u64) -> TensorMap {
        let slot = f16_slot(weights, "test");
        TensorMap {
            slots: vec![slot.clone()],
            total_capacity_bits: slot.capacity_bits,
            total_capacity_bytes: slot.capacity_bits / 8,
        }
    }

    #[test]
    fn empty_bitmap_allocates_no_pages() {
        let map = one_slot_map(1_000_000);
        let bitmap = DirtyBitmap::new(&map);
        assert_eq!(bitmap.total_bits(), 1_000_000);
        assert_eq!(bitmap.set_count(), 0);
        assert_eq!(bitmap.allocated_page_count(), 0);
        assert_eq!(bitmap.total_bytes(), 125_000);
    }

    #[test]
    fn mark_allocates_one_page() {
        let map = one_slot_map(1_000_000);
        let mut bitmap = DirtyBitmap::new(&map);
        bitmap.mark(0, 100);
        assert_eq!(bitmap.allocated_page_count(), 1);
        assert!(bitmap.is_dirty(0, 100));
        assert!(!bitmap.is_dirty(0, 99));
        assert_eq!(bitmap.set_count(), 1);
    }

    #[test]
    fn distant_marks_allocate_separate_pages() {
        let map = one_slot_map(10_000_000);
        let mut bitmap = DirtyBitmap::new(&map);
        bitmap.mark(0, 0);
        bitmap.mark(0, 9_000_000);
        assert_eq!(bitmap.allocated_page_count(), 2);
        assert_eq!(bitmap.set_count(), 2);
    }

    #[test]
    fn marks_within_one_page_share_a_page() {
        let map = one_slot_map(1_000_000);
        let mut bitmap = DirtyBitmap::new(&map);
        for i in 0..100 {
            bitmap.mark(0, i);
        }
        assert_eq!(bitmap.allocated_page_count(), 1);
        assert_eq!(bitmap.set_count(), 100);
    }

    #[test]
    fn mark_is_idempotent() {
        let map = one_slot_map(1_000);
        let mut bitmap = DirtyBitmap::new(&map);
        bitmap.mark(0, 5);
        bitmap.mark(0, 5);
        assert_eq!(bitmap.set_count(), 1);
    }

    #[test]
    fn out_of_range_mark_is_silently_ignored() {
        let map = one_slot_map(100);
        let mut bitmap = DirtyBitmap::new(&map);
        bitmap.mark(0, 200); // past total_bits
        assert_eq!(bitmap.set_count(), 0);
    }

    #[test]
    fn mark_range_and_is_range_dirty() {
        let map = one_slot_map(10_000);
        let mut bitmap = DirtyBitmap::new(&map);
        bitmap.mark_range(0, 100, 50);
        assert!(bitmap.is_range_dirty(0, 100, 50));
        assert!(!bitmap.is_range_dirty(0, 99, 51));
        assert!(!bitmap.is_range_dirty(0, 100, 51));
        assert!(bitmap.is_range_dirty(0, 100, 0)); // empty range vacuous
    }

    #[test]
    fn write_bytes_at_zero_skips_allocation() {
        let map = one_slot_map(1_000_000);
        let mut bitmap = DirtyBitmap::new(&map);
        bitmap.write_bytes_at(0, &vec![0u8; 100_000]);
        assert_eq!(bitmap.allocated_page_count(), 0);
        assert_eq!(bitmap.set_count(), 0);
    }

    #[test]
    fn write_bytes_at_nonzero_allocates_only_touched_pages() {
        let map = one_slot_map(1_000_000); // 125_000 bytes total
        let mut bitmap = DirtyBitmap::new(&map);
        let mut bytes = vec![0u8; 125_000];
        bytes[0] = 0x01; // page 0
        bytes[100_000] = 0xFF; // page 24 (100000 / 4096 = 24.4)
        bitmap.write_bytes_at(0, &bytes);
        assert_eq!(bitmap.allocated_page_count(), 2);
        assert!(bitmap.is_dirty(0, 0)); // bit 0
        assert!(bitmap.is_dirty(0, 100_000 * 8)); // bit at byte 100_000
    }

    #[test]
    fn write_to_round_trips_through_write_bytes_at() {
        let map = one_slot_map(50_000);
        let mut a = DirtyBitmap::new(&map);
        a.mark_range(0, 10, 100);
        a.mark(0, 40_000);

        let mut buf = Vec::new();
        a.write_to(&mut buf).expect("write_to");
        assert_eq!(buf.len(), a.total_bytes() as usize);

        let mut b = DirtyBitmap::new(&map);
        b.write_bytes_at(0, &buf);
        // a and b should agree on every set bit.
        for w in [0_u32, 9, 10, 50, 109, 110, 39_999, 40_000, 40_001] {
            assert_eq!(a.is_dirty(0, w), b.is_dirty(0, w), "bit {w} mismatch");
        }
        assert_eq!(a.set_count(), b.set_count());
    }
}
