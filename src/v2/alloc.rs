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
//! - `AllocError::PointerOutOfBounds` — the pointer's covered weights
//!   extend past the end of its slot.
//! - `AllocError::SlotTooLarge` / `AllocError::TooManySlots` — the
//!   cover shape exceeds V2's current `Pointer`/freelist address widths.

use thiserror::Error;

use crate::stego::calibration::stealable_bits_for;
use crate::stego::tensor_map::TensorMap;
use crate::v2::ceiling::CeilingSummary;
use crate::v2::freelist::{FreeRun, FreeRunSet, ReserveError};
use crate::v2::pointer::Pointer;
use crate::v2::salience::SalienceTable;

/// Pristine-path allocator over a cover's free-run set.
#[derive(Debug)]
pub struct Allocator {
    freelist: FreeRunSet,
    ceiling: CeilingSummary,
    /// Per-weight salience (B4). Empty on un-calibrated covers; the
    /// allocator's FitKey degenerates to ceiling-only in that case.
    salience: SalienceTable,
}

#[derive(Debug, Error, PartialEq)]
pub enum AllocError {
    #[error("no free run with length ≥ {requested_bits} bits")]
    OutOfSpace { requested_bits: u32 },

    #[error(
        "free called on a pointer that overlaps an existing free run (slot {slot}, start {start})"
    )]
    DoubleFree { slot: u16, start: u32 },

    #[error("pointer references slot {slot} but map has only {slot_count} slots")]
    SlotOutOfRange { slot: u16, slot_count: usize },

    #[error("pointer targets slot {slot}, which has no stealable bits")]
    NonStealableSlot { slot: u16 },

    #[error(
        "pointer range [{start_weight}, {end_weight}) lies outside slot {slot} with {weight_count} weights"
    )]
    PointerOutOfBounds {
        slot: u16,
        start_weight: u32,
        end_weight: u64,
        weight_count: u64,
    },

    #[error("slot {slot} has {weight_count} weights, exceeding V2's current max of {max_weights}")]
    SlotTooLarge {
        slot: usize,
        weight_count: u64,
        max_weights: u64,
    },

    #[error("map has {slot_count} slots, exceeding V2's current max of {max_slots}")]
    TooManySlots { slot_count: usize, max_slots: usize },

    #[error("reserve failed: {source}")]
    Reserve {
        #[from]
        source: ReserveError,
    },
}

impl Allocator {
    /// Build an allocator over a fresh cover: each eligible slot
    /// contributes one initial free run spanning its full weight
    /// range. Non-stealable slots are skipped.
    ///
    /// The allocator starts with an empty [`SalienceTable`] —
    /// [`Self::set_salience`] populates it at mount time on
    /// calibrated covers.
    pub fn new_for_map(map: &TensorMap, ceiling: CeilingSummary) -> Result<Self, AllocError> {
        Self::new_for_map_with_salience(map, ceiling, SalienceTable::empty())
    }

    /// Like [`Self::new_for_map`] but seeds the salience table at
    /// construction. Used by mount paths that decode `salience_inode`
    /// before installing free runs, so every run's initial
    /// `max_salience` reflects the per-weight data.
    pub fn new_for_map_with_salience(
        map: &TensorMap,
        ceiling: CeilingSummary,
        salience: SalienceTable,
    ) -> Result<Self, AllocError> {
        const MAX_SLOTS: usize = u16::MAX as usize + 1;
        const MAX_WEIGHTS_PER_SLOT: u64 = u32::MAX as u64;

        if map.slots.len() > MAX_SLOTS {
            return Err(AllocError::TooManySlots {
                slot_count: map.slots.len(),
                max_slots: MAX_SLOTS,
            });
        }

        let mut freelist = FreeRunSet::new();
        for (slot_idx, slot) in map.slots.iter().enumerate() {
            let bpw = stealable_bits_for(slot.quant_type);
            if bpw == 0 || slot.weight_count == 0 {
                continue;
            }
            if slot.weight_count > MAX_WEIGHTS_PER_SLOT {
                return Err(AllocError::SlotTooLarge {
                    slot: slot_idx,
                    weight_count: slot.weight_count,
                    max_weights: MAX_WEIGHTS_PER_SLOT,
                });
            }
            let max_ceiling = ceiling.max_over_range(slot_idx as u32, 0, slot.weight_count);
            let max_salience =
                salience.max_over_range(slot_idx as u32, 0, slot.weight_count);
            freelist.insert(FreeRun {
                slot: slot_idx as u16,
                start_weight: 0,
                length_in_weights: slot.weight_count as u32,
                max_ceiling,
                max_salience,
            });
        }
        Ok(Self {
            freelist,
            ceiling,
            salience,
        })
    }

    /// Remove every `(slot, weight_index)` in `weights` from the free
    /// set, splitting containing runs as needed. Used at `llmdb init`
    /// to carve out the anchor positions so the allocator never
    /// returns them to data-chunk callers. Errors as soon as any
    /// individual reservation fails (NotFree means the weight was
    /// already outside a free run).
    pub fn reserve_weights<I>(&mut self, weights: I) -> Result<(), AllocError>
    where
        I: IntoIterator<Item = (u16, u32)>,
    {
        for (slot, weight_index) in weights {
            self.freelist
                .reserve_weight(slot, weight_index, &self.ceiling, &self.salience)?;
        }
        Ok(())
    }

    /// Borrow the cover's ceiling-magnitude bucket summary — the
    /// `Filesystem` layer needs it at mount time to rebuild its own
    /// view.
    pub fn ceiling(&self) -> &CeilingSummary {
        &self.ceiling
    }

    /// Borrow the per-weight salience table. Empty on un-calibrated
    /// covers.
    pub fn salience(&self) -> &SalienceTable {
        &self.salience
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

        let picked = self
            .freelist
            .pop_best_fit_where(|run| run_fits_length(map, run, length_in_bits))?;
        Some(self.finalize_alloc(map, picked, length_in_bits))
    }

    /// Allocate preferring runs whose first-N weights are already
    /// dirty. Priority 2 per DESIGN-NEW §15.5: writing to
    /// already-perturbed weights costs less cover damage than
    /// perturbing pristine ones. Falls back to plain
    /// [`Self::alloc`] when no dirty-sufficient run exists.
    pub fn alloc_preferring_dirty(
        &mut self,
        map: &TensorMap,
        length_in_bits: u32,
        dirty: &crate::v2::dirty::DirtyBitmap,
    ) -> Option<Pointer> {
        if length_in_bits == 0 {
            return Some(Pointer::NULL);
        }

        // Pass 1: smallest-max-ceiling run where `length fits` AND
        // the first-needed_weights are dirty.
        let picked = self.freelist.pop_best_fit_where(|run| {
            if !run_fits_length(map, run, length_in_bits) {
                return false;
            }
            let slot = &map.slots[run.slot as usize];
            let bpw = slot.stealable_bits_per_weight as u32;
            let needed = weights_for_bits(length_in_bits, bpw);
            dirty.is_range_dirty(run.slot, run.start_weight, needed)
        });
        if let Some(run) = picked {
            return Some(self.finalize_alloc(map, run, length_in_bits));
        }

        // Pass 2: fall back to plain alloc (pristine ordering).
        self.alloc(map, length_in_bits)
    }

    /// Split a picked run if longer than needed, push the remainder
    /// back to the free list, and return a `Pointer` to the
    /// allocated prefix. Shared by every alloc path.
    fn finalize_alloc(&mut self, map: &TensorMap, picked: FreeRun, length_in_bits: u32) -> Pointer {
        let slot = &map.slots[picked.slot as usize];
        let bpw = slot.stealable_bits_per_weight as u32;
        let needed_weights = weights_for_bits(length_in_bits, bpw);

        let allocated_pointer = Pointer {
            slot: picked.slot,
            start_weight: picked.start_weight,
            length_in_bits,
            flags: 0,
            reserved: 0,
        };

        if picked.length_in_weights > needed_weights {
            let remainder_start = picked.start_weight + needed_weights;
            let remainder_len = picked.length_in_weights - needed_weights;
            let remainder_ceiling = self.ceiling.max_over_range(
                picked.slot as u32,
                remainder_start as u64,
                remainder_len as u64,
            );
            let remainder_salience = self.salience.max_over_range(
                picked.slot as u32,
                remainder_start as u64,
                remainder_len as u64,
            );
            self.freelist.insert(FreeRun {
                slot: picked.slot,
                start_weight: remainder_start,
                length_in_weights: remainder_len,
                max_ceiling: remainder_ceiling,
                max_salience: remainder_salience,
            });
        }

        allocated_pointer
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
        if bpw == 0 {
            return Err(AllocError::NonStealableSlot { slot: pointer.slot });
        }
        let length_in_weights = weights_for_bits(pointer.length_in_bits, bpw);
        let end_weight = pointer_end_weight(pointer.start_weight, length_in_weights);
        if end_weight > slot.weight_count {
            return Err(AllocError::PointerOutOfBounds {
                slot: pointer.slot,
                start_weight: pointer.start_weight,
                end_weight,
                weight_count: slot.weight_count,
            });
        }
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
        let max_salience = self.salience.max_over_range(
            pointer.slot as u32,
            pointer.start_weight as u64,
            length_in_weights as u64,
        );
        self.freelist.insert_with_merge(FreeRun {
            slot: pointer.slot,
            start_weight: pointer.start_weight,
            length_in_weights,
            max_ceiling,
            max_salience,
        });
        Ok(())
    }
}

fn weights_for_bits(length_in_bits: u32, bits_per_weight: u32) -> u32 {
    if length_in_bits == 0 {
        0
    } else {
        length_in_bits.div_ceil(bits_per_weight)
    }
}

/// True when `run` has enough capacity (in bits) to hold a chunk of
/// `length_in_bits`. Shared by every alloc path.
fn run_fits_length(map: &TensorMap, run: &FreeRun, length_in_bits: u32) -> bool {
    let slot = &map.slots[run.slot as usize];
    let bpw = slot.stealable_bits_per_weight as u32;
    if bpw == 0 {
        return false;
    }
    let run_bits = run.length_in_weights.saturating_mul(bpw);
    run_bits >= length_in_bits
}

fn pointer_end_weight(start_weight: u32, length_in_weights: u32) -> u64 {
    start_weight as u64 + length_in_weights as u64
}
