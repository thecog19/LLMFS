//! V2 allocator — pristine path only.
//!
//! For this milestone the allocator implements priority level 3 of
//! DESIGN-NEW §15.5: "pick the lowest-max-ceiling-magnitude free run
//! of sufficient length." Dedup (priority 1) and dirty-preference
//! (priority 2) layer on top in later steps and re-use the same
//! free-run set.
//!
//! Flow:
//!
//! 1. **Init** — [`Allocator::new_for_map`] walks every eligible
//!    slot, inserts one full-range free run per slot (with max_ceiling
//!    read from the [`CeilingSummary`]).
//! 2. **Alloc** — [`Allocator::alloc`] calls
//!    [`FreeRunSet::pop_best_fit_where`] with a bits-to-weights
//!    predicate; returns the run's first `requested_weights` as an
//!    allocated `Pointer`, and pushes any remainder back to the
//!    freelist with a ceiling recomputed via
//!    [`CeilingSummary::max_over_range`] (tighter than the parent
//!    run's ceiling in general).
//! 3. **Free** — [`Allocator::free`] looks up the run's extent from
//!    the pointer, rebuilds a `FreeRun`, and pushes via
//!    `insert_with_merge` so adjacent free runs coalesce.
//!
//! Failure modes:
//! - `AllocError::OutOfSpace` — no free run can hold the request.
//! - `AllocError::DoubleFree` — a pointer's extent overlaps an
//!   existing free run (allocator assumption: pointers come from a
//!   prior alloc call; freeing the same pointer twice is a programmer
//!   error).

use thiserror::Error;

use crate::stego::calibration::stealable_bits_for;
use crate::stego::tensor_map::TensorMap;
use crate::v2::ceiling::CeilingSummary;
use crate::v2::freelist::{FreeRun, FreeRunSet};
use crate::v2::pointer::Pointer;

/// Pristine-path allocator over a cover's free-run set.
#[derive(Debug)]
pub struct Allocator {
    freelist: FreeRunSet,
    ceiling: CeilingSummary,
}

#[derive(Debug, Error, PartialEq)]
pub enum AllocError {
    #[error("no free run with length ≥ {requested_bits} bits")]
    OutOfSpace { requested_bits: u32 },

    #[error("free called on a pointer that overlaps an existing free run (slot {slot}, start {start})")]
    DoubleFree { slot: u16, start: u32 },

    #[error("pointer references slot {slot} but map has only {slot_count} slots")]
    SlotOutOfRange { slot: u16, slot_count: usize },

    #[error("pointer's length_in_bits ({length}) is not a multiple of the slot's stealable bits per weight ({bpw})")]
    UnalignedLength { length: u32, bpw: u32 },
}

impl Allocator {
    /// Build an allocator over a fresh cover: each eligible slot
    /// contributes one initial free run spanning its full weight
    /// range. Non-stealable slots are skipped.
    pub fn new_for_map(map: &TensorMap, ceiling: CeilingSummary) -> Self {
        let mut freelist = FreeRunSet::new();
        for (slot_idx, slot) in map.slots.iter().enumerate() {
            let bpw = stealable_bits_for(slot.quant_type);
            if bpw == 0 || slot.weight_count == 0 {
                continue;
            }
            let max_ceiling = ceiling.max_over_range(
                slot_idx as u32,
                0,
                slot.weight_count,
            );
            freelist.insert(FreeRun {
                slot: slot_idx as u16,
                start_weight: 0,
                length_in_weights: slot.weight_count as u32,
                max_ceiling,
            });
        }
        Self { freelist, ceiling }
    }

    /// Count of free runs, for diagnostics and tests.
    pub fn free_run_count(&self) -> usize {
        self.freelist.len()
    }

    /// Sum of free-run lengths, for diagnostics and tests.
    pub fn total_free_weights(&self) -> u64 {
        // Iterate by popping requires mutation; instead expose a
        // counting helper. For now, just alloc+free a sentinel — but
        // that's ugly. Simpler: add a public accessor to FreeRunSet.
        // Kept here as a sum over an iterator (requires a freelist
        // method to support it).
        self.freelist.total_weights()
    }

    /// Allocate a chunk of `length_in_bits` bits. Picks the lowest
    /// `max_ceiling` free run across all eligible slots that can hold
    /// the request, splits if necessary, and returns a `Pointer` to
    /// the allocated prefix.
    pub fn alloc(&mut self, map: &TensorMap, length_in_bits: u32) -> Option<Pointer> {
        if length_in_bits == 0 {
            return Some(Pointer::NULL);
        }

        let picked = self.freelist.pop_best_fit_where(|run| {
            let slot = &map.slots[run.slot as usize];
            let bpw = slot.stealable_bits_per_weight as u32;
            if bpw == 0 {
                return false;
            }
            let run_bits = run.length_in_weights.saturating_mul(bpw);
            run_bits >= length_in_bits
        })?;

        let slot = &map.slots[picked.slot as usize];
        let bpw = slot.stealable_bits_per_weight as u32;
        // ceil-div: cover any request that isn't an exact multiple of bpw
        let needed_weights = length_in_bits.div_ceil(bpw);

        let allocated_pointer = Pointer {
            slot: picked.slot,
            start_weight: picked.start_weight,
            length_in_bits,
            flags: 0,
            reserved: 0,
        };

        // If the picked run is longer than needed, split and push
        // the remainder back with a recomputed ceiling.
        if picked.length_in_weights > needed_weights {
            let remainder_start = picked.start_weight + needed_weights;
            let remainder_len = picked.length_in_weights - needed_weights;
            let remainder_ceiling = self.ceiling.max_over_range(
                picked.slot as u32,
                remainder_start as u64,
                remainder_len as u64,
            );
            self.freelist.insert(FreeRun {
                slot: picked.slot,
                start_weight: remainder_start,
                length_in_weights: remainder_len,
                max_ceiling: remainder_ceiling,
            });
        }

        Some(allocated_pointer)
    }

    /// Free a previously-allocated chunk. Returns `Err(DoubleFree)`
    /// if the pointer's extent overlaps an existing free run.
    pub fn free(&mut self, map: &TensorMap, pointer: Pointer) -> Result<(), AllocError> {
        if pointer.is_null() {
            return Ok(());
        }
        if pointer.slot as usize >= map.slots.len() {
            return Err(AllocError::SlotOutOfRange {
                slot: pointer.slot,
                slot_count: map.slots.len(),
            });
        }
        let slot = &map.slots[pointer.slot as usize];
        let bpw = slot.stealable_bits_per_weight as u32;
        if bpw == 0 || !pointer.length_in_bits.is_multiple_of(bpw) {
            return Err(AllocError::UnalignedLength {
                length: pointer.length_in_bits,
                bpw,
            });
        }
        let length_in_weights = pointer.length_in_bits / bpw;
        if self
            .freelist
            .overlaps_run(pointer.slot, pointer.start_weight, length_in_weights)
        {
            return Err(AllocError::DoubleFree {
                slot: pointer.slot,
                start: pointer.start_weight,
            });
        }
        let max_ceiling = self.ceiling.max_over_range(
            pointer.slot as u32,
            pointer.start_weight as u64,
            length_in_weights as u64,
        );
        self.freelist.insert_with_merge(FreeRun {
            slot: pointer.slot,
            start_weight: pointer.start_weight,
            length_in_weights,
            max_ceiling,
        });
        Ok(())
    }
}
