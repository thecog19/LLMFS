//! Per-cover free-run set (DESIGN-NEW §15.5).
//!
//! Tracks the unallocated contiguous `WeightRef` runs of the cover
//! and answers two queries:
//!
//! - `pop_best_fit(min_len)` — remove and return the free run with
//!   the smallest `(max_ceiling, max_salience)` among those whose
//!   length is `≥ min_len`. O(log N) in the number of free runs.
//! - `insert_with_merge(run)` — add a free run; if it abuts existing
//!   free run(s) in the same slot, coalesce them. Merged maxes are
//!   `max(neighbour.max_{ceiling,salience}, run.max_{ceiling,salience})`
//!   — splitting a run can only reduce max, so combining two runs
//!   whose originals were correctly sized means the merged max is
//!   an exact upper bound.
//!
//! The dual-index scheme:
//!
//! - `by_position: BTreeMap<(slot, start_weight), FreeRunEntry>` —
//!   primary storage, answers the adjacency queries used in
//!   merge-on-free in O(log N).
//! - `by_fit: BTreeSet<(max_ceiling_bits, max_salience_bits,
//!   length_in_weights, slot, start_weight)>` — keyed to make
//!   "smallest-cost with length ≥ L" a lower-bound range query in
//!   O(log N). Ceiling-primary, salience-secondary. When the cover
//!   isn't calibrated every run has `max_salience = 0.0` and the
//!   salience component is a no-op, so behavior matches
//!   pre-B4 exactly.
//!
//! Both `max_ceiling` and `max_salience` are stored as `f32::to_bits
//! () as u32` for the Ord key; this preserves ordering for
//! non-negative non-NaN floats (both are non-negative by
//! construction, and a `debug_assert` catches NaN).

use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use crate::v2::ceiling::CeilingSummary;
use crate::v2::salience::SalienceTable;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ReserveError {
    #[error("weight ({slot}, {weight_index}) is not in any free run")]
    NotFree { slot: u16, weight_index: u32 },
}

/// Key type for the `by_fit` index. Packed tuple for lexicographic
/// ordering: `(max_ceiling_bits, max_salience_bits,
/// length_in_weights, slot, start_weight)`. Ceiling-primary,
/// salience-secondary; ties then prefer shorter runs + canonical
/// `(slot, start_weight)`.
type FitKey = (u32, u32, u32, u16, u32);

/// Unallocated contiguous run of WeightRef positions in a single slot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FreeRun {
    pub slot: u16,
    pub start_weight: u32,
    pub length_in_weights: u32,
    /// `max(ceiling_magnitude(w))` over weights `[start..start+length)`.
    /// Invariant: non-negative and not NaN.
    pub max_ceiling: f32,
    /// `max(salience(w))` over weights `[start..start+length)`.
    /// `0.0` for runs on un-calibrated covers (the neutral value in
    /// the compound FitKey — ordering degenerates to ceiling-only).
    /// Invariant: non-negative and not NaN.
    pub max_salience: f32,
}

impl FreeRun {
    fn end_weight(&self) -> u32 {
        self.start_weight + self.length_in_weights
    }

    fn fit_key(&self) -> FitKey {
        debug_assert!(
            !self.max_ceiling.is_nan(),
            "FreeRun.max_ceiling must not be NaN",
        );
        debug_assert!(
            self.max_ceiling >= 0.0,
            "FreeRun.max_ceiling must be non-negative, got {}",
            self.max_ceiling,
        );
        debug_assert!(
            !self.max_salience.is_nan(),
            "FreeRun.max_salience must not be NaN",
        );
        debug_assert!(
            self.max_salience >= 0.0,
            "FreeRun.max_salience must be non-negative, got {}",
            self.max_salience,
        );
        (
            self.max_ceiling.to_bits(),
            self.max_salience.to_bits(),
            self.length_in_weights,
            self.slot,
            self.start_weight,
        )
    }
}

/// Per-cover free-run tracker with dual ordering.
#[derive(Debug, Default, Clone)]
pub struct FreeRunSet {
    by_position: BTreeMap<(u16, u32), FreeRunEntry>,
    by_fit: BTreeSet<FitKey>,
}

#[derive(Debug, Clone, Copy)]
struct FreeRunEntry {
    length_in_weights: u32,
    max_ceiling: f32,
    max_salience: f32,
}

impl FreeRunSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.by_position.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_position.is_empty()
    }

    /// Sum of `length_in_weights` across all free runs. O(N) — meant
    /// for diagnostics and tests, not the allocation hot path.
    pub fn total_weights(&self) -> u64 {
        self.by_position
            .values()
            .map(|e| e.length_in_weights as u64)
            .sum()
    }

    /// Remove the single weight `(slot, weight_index)` from the free
    /// set, splitting the containing run if necessary. Left and
    /// right halves' `max_ceiling` and `max_salience` are recomputed
    /// over the sub-range so the allocator's fit-ordering stays
    /// accurate after the split.
    ///
    /// Used at init to carve out the anchor positions so the
    /// allocator never hands them back as data chunks.
    ///
    /// Returns `ReserveError::NotFree` if the weight isn't in any
    /// free run (already allocated, or outside the slot's range).
    pub fn reserve_weight(
        &mut self,
        slot: u16,
        weight_index: u32,
        ceiling: &CeilingSummary,
        salience: &SalienceTable,
    ) -> Result<(), ReserveError> {
        // Containing run: rightmost entry whose key is ≤ (slot, weight_index).
        let containing_key = self
            .by_position
            .range(..=(slot, weight_index))
            .next_back()
            .map(|(k, _)| *k);
        let key = containing_key.ok_or(ReserveError::NotFree { slot, weight_index })?;
        if key.0 != slot {
            return Err(ReserveError::NotFree { slot, weight_index });
        }
        let entry = *self.by_position.get(&key).unwrap();
        let run_start = key.1;
        let run_end = run_start + entry.length_in_weights;
        if weight_index < run_start || weight_index >= run_end {
            return Err(ReserveError::NotFree { slot, weight_index });
        }

        // Remove the whole run.
        self.remove_raw(key).expect("run was in by_position");

        // Reinsert the left half, if any.
        if weight_index > run_start {
            let left_len = weight_index - run_start;
            let left_max = ceiling.max_over_range(slot as u32, run_start as u64, left_len as u64);
            let left_sal = salience.max_over_range(slot as u32, run_start as u64, left_len as u64);
            self.insert_raw(FreeRun {
                slot,
                start_weight: run_start,
                length_in_weights: left_len,
                max_ceiling: left_max,
                max_salience: left_sal,
            });
        }

        // Reinsert the right half, if any.
        if weight_index + 1 < run_end {
            let right_start = weight_index + 1;
            let right_len = run_end - right_start;
            let right_max =
                ceiling.max_over_range(slot as u32, right_start as u64, right_len as u64);
            let right_sal =
                salience.max_over_range(slot as u32, right_start as u64, right_len as u64);
            self.insert_raw(FreeRun {
                slot,
                start_weight: right_start,
                length_in_weights: right_len,
                max_ceiling: right_max,
                max_salience: right_sal,
            });
        }

        Ok(())
    }

    /// True if any free run in `slot` overlaps the weight range
    /// `[start, start + length)`. Used by the allocator to detect
    /// double-free: freeing a pointer whose extent is already marked
    /// free is a bug.
    pub fn overlaps_run(&self, slot: u16, start: u32, length: u32) -> bool {
        let end = start + length;
        // Check the first run at-or-before `start` — it might extend
        // past `start`.
        if let Some(((s, run_start), entry)) = self
            .by_position
            .range(..=(slot, start))
            .next_back()
            .map(|(k, v)| (*k, *v))
            && s == slot
            && run_start + entry.length_in_weights > start
        {
            return true;
        }
        // Check runs that start strictly after `start` but before `end`.
        self.by_position
            .range((slot, start + 1)..(slot, end))
            .next()
            .is_some()
    }

    /// Insert a free run without adjacency merge. Used when pushing
    /// a split remainder back into the set — the caller has already
    /// decided this run shouldn't coalesce with neighbours.
    pub fn insert(&mut self, run: FreeRun) {
        self.insert_raw(run);
    }

    /// Insert a free run and coalesce with adjacent runs in the same
    /// slot. This is the correct operation when freeing a chunk.
    /// Merged `max_ceiling` / `max_salience` are the pairwise max
    /// of the combining runs' values.
    pub fn insert_with_merge(&mut self, mut run: FreeRun) {
        // Look left: run ending at run.start_weight?
        let left_key = self
            .by_position
            .range(..(run.slot, run.start_weight))
            .next_back()
            .map(|(k, _)| *k);
        if let Some(key) = left_key {
            let entry = self.by_position.get(&key).copied().unwrap();
            if key.0 == run.slot && key.1 + entry.length_in_weights == run.start_weight {
                // Adjacent. Remove it and merge into `run`.
                let prev = self.remove_raw(key).unwrap();
                run = FreeRun {
                    slot: prev.slot,
                    start_weight: prev.start_weight,
                    length_in_weights: prev.length_in_weights + run.length_in_weights,
                    max_ceiling: prev.max_ceiling.max(run.max_ceiling),
                    max_salience: prev.max_salience.max(run.max_salience),
                };
            }
        }

        // Look right: run starting at run.end_weight?
        let right_key = (run.slot, run.end_weight());
        if let Some(entry) = self.by_position.get(&right_key).copied() {
            let next = self.remove_raw(right_key).unwrap();
            debug_assert_eq!(next.length_in_weights, entry.length_in_weights);
            run = FreeRun {
                slot: run.slot,
                start_weight: run.start_weight,
                length_in_weights: run.length_in_weights + next.length_in_weights,
                max_ceiling: run.max_ceiling.max(next.max_ceiling),
                max_salience: run.max_salience.max(next.max_salience),
            };
        }

        self.insert_raw(run);
    }

    /// Remove and return the free run with the smallest `max_ceiling`
    /// whose length is `≥ min_length` (measured in weights). Ties
    /// break by `(length, slot, start_weight)` — same key order as
    /// the BTreeSet index, so the result is deterministic.
    pub fn pop_best_fit(&mut self, min_length: u32) -> Option<FreeRun> {
        self.pop_best_fit_where(|run| run.length_in_weights >= min_length)
    }

    /// Walk free runs in ascending `(max_ceiling, length, slot,
    /// start_weight)` order and pop the first one for which
    /// `predicate` returns true. Lets callers filter by cross-slot
    /// criteria (e.g. "length in BITS ≥ K" where the weights-to-bits
    /// conversion depends on the slot's quant type) without needing
    /// the freelist itself to know about quant types.
    pub fn pop_best_fit_where(
        &mut self,
        mut predicate: impl FnMut(&FreeRun) -> bool,
    ) -> Option<FreeRun> {
        let mut winner_key: Option<FitKey> = None;
        for key in &self.by_fit {
            let (_ceiling_bits, _salience_bits, length, slot, start) = *key;
            let entry = match self.by_position.get(&(slot, start)).copied() {
                Some(e) => e,
                None => continue, // by_fit / by_position invariant violated; skip
            };
            debug_assert_eq!(entry.length_in_weights, length);
            let run = FreeRun {
                slot,
                start_weight: start,
                length_in_weights: entry.length_in_weights,
                max_ceiling: entry.max_ceiling,
                max_salience: entry.max_salience,
            };
            if predicate(&run) {
                winner_key = Some(*key);
                break;
            }
        }
        let (_c, _s, _len, slot, start) = winner_key?;
        self.remove_raw((slot, start))
    }

    fn insert_raw(&mut self, run: FreeRun) {
        debug_assert!(
            run.length_in_weights > 0,
            "refusing to insert zero-length run"
        );
        debug_assert!(
            !self.by_position.contains_key(&(run.slot, run.start_weight)),
            "duplicate free run at ({}, {})",
            run.slot,
            run.start_weight,
        );
        self.by_fit.insert(run.fit_key());
        self.by_position.insert(
            (run.slot, run.start_weight),
            FreeRunEntry {
                length_in_weights: run.length_in_weights,
                max_ceiling: run.max_ceiling,
                max_salience: run.max_salience,
            },
        );
    }

    fn remove_raw(&mut self, key: (u16, u32)) -> Option<FreeRun> {
        let entry = self.by_position.remove(&key)?;
        let run = FreeRun {
            slot: key.0,
            start_weight: key.1,
            length_in_weights: entry.length_in_weights,
            max_ceiling: entry.max_ceiling,
            max_salience: entry.max_salience,
        };
        self.by_fit.remove(&run.fit_key());
        Some(run)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(slot: u16, start: u32, len: u32, ceiling: f32, salience: f32) -> FreeRun {
        FreeRun {
            slot,
            start_weight: start,
            length_in_weights: len,
            max_ceiling: ceiling,
            max_salience: salience,
        }
    }

    #[test]
    fn fit_key_orders_by_ceiling_then_salience_then_length() {
        // Same max_ceiling → shorter length sorts first (so best-fit
        // prefers tighter-fitting runs when multiple satisfy the
        // min_length threshold).
        let longer = run(0, 0, 100, 0.1, 0.0);
        let shorter = run(0, 200, 50, 0.1, 0.0);
        assert!(
            shorter.fit_key() < longer.fit_key(),
            "tie-broken by shorter-first"
        );

        // Higher max_ceiling sorts after lower regardless of length.
        let low_max = FreeRun {
            max_ceiling: 0.05,
            ..longer
        };
        let high_max = FreeRun {
            max_ceiling: 0.5,
            ..shorter
        };
        assert!(low_max.fit_key() < high_max.fit_key());
    }

    #[test]
    fn fit_key_salience_is_secondary_tiebreaker() {
        // Same ceiling, same length, different salience → lower
        // salience sorts first.
        let low = run(0, 0, 100, 0.2, 0.05);
        let high = run(0, 500, 100, 0.2, 0.5);
        assert!(low.fit_key() < high.fit_key());

        // Ceiling wins over salience: low ceiling + high salience
        // still sorts before high ceiling + zero salience.
        let low_ceiling_high_sal = run(0, 0, 100, 0.1, 1.0);
        let high_ceiling_zero_sal = run(0, 500, 100, 0.9, 0.0);
        assert!(low_ceiling_high_sal.fit_key() < high_ceiling_zero_sal.fit_key());
    }

    #[test]
    fn end_weight_computation() {
        let r = run(0, 100, 50, 0.0, 0.0);
        assert_eq!(r.end_weight(), 150);
    }

    #[test]
    fn internal_insert_and_remove() {
        let mut fl = FreeRunSet::new();
        let r = run(0, 0, 10, 1.0, 0.0);
        fl.insert_raw(r);
        assert_eq!(fl.len(), 1);
        let removed = fl.remove_raw((0, 0)).unwrap();
        assert_eq!(removed, r);
        assert!(fl.is_empty());
    }
}
