//! V2 free-run set — per-cover tracker of unallocated contiguous
//! WeightRef runs, keyed for efficient "smallest-max-ceiling with
//! length ≥ L" queries and merge-on-free.
//!
//! Properties under test:
//! 1. **Round-trip** — insert then pop_best_fit returns the same run
//!    when only one is present.
//! 2. **Best-fit ordering** — among runs meeting the length
//!    requirement, pop_best_fit picks the one with the lowest
//!    max_ceiling.
//! 3. **Too-small rejection** — all candidate runs shorter than the
//!    request returns `None`.
//! 4. **Slot scoping** — runs from different slots don't merge.
//! 5. **Merge-on-free** — freeing a run adjacent to existing free
//!    run(s) in the same slot produces a single coalesced run with
//!    max_ceiling = max(neighbour_max, freed_max).
//! 6. **Three-way merge** — freeing a run with neighbours on both
//!    sides collapses all three.

use llmdb::v2::ceiling::CeilingSummary;
use llmdb::v2::freelist::{FreeRun, FreeRunSet, ReserveError};
use llmdb::v2::salience::SalienceTable;

fn make_run(slot: u16, start: u32, len: u32, max: f32) -> FreeRun {
    FreeRun {
        slot,
        start_weight: start,
        length_in_weights: len,
        max_ceiling: max,
        max_salience: 0.0,
    }
}

#[test]
fn new_is_empty() {
    let fl = FreeRunSet::new();
    assert_eq!(fl.len(), 0);
    assert!(fl.is_empty());
}

#[test]
fn insert_then_pop_round_trip() {
    let mut fl = FreeRunSet::new();
    let run = make_run(0, 0, 100, 0.5);
    fl.insert(run);
    assert_eq!(fl.len(), 1);
    let got = fl.pop_best_fit(50).expect("pop");
    assert_eq!(got, run);
    assert!(fl.is_empty());
}

#[test]
fn pop_best_fit_picks_lowest_max_ceiling() {
    let mut fl = FreeRunSet::new();
    fl.insert(make_run(0, 0, 100, 5.0));
    fl.insert(make_run(0, 200, 100, 1.0)); // smallest
    fl.insert(make_run(0, 400, 100, 3.0));

    let got = fl.pop_best_fit(50).expect("pop");
    assert_eq!(got.max_ceiling, 1.0);
    assert_eq!(got.start_weight, 200);
    assert_eq!(fl.len(), 2);
}

#[test]
fn pop_best_fit_skips_short_runs_even_with_low_ceiling() {
    let mut fl = FreeRunSet::new();
    fl.insert(make_run(0, 0, 10, 0.1)); // low ceiling but too small
    fl.insert(make_run(0, 100, 200, 0.9));
    let got = fl.pop_best_fit(100).expect("pop");
    assert_eq!(got.length_in_weights, 200);
    assert_eq!(got.max_ceiling, 0.9);
}

#[test]
fn pop_best_fit_returns_none_if_nothing_fits() {
    let mut fl = FreeRunSet::new();
    fl.insert(make_run(0, 0, 50, 0.1));
    assert_eq!(fl.pop_best_fit(100), None);
    // Didn't consume the run.
    assert_eq!(fl.len(), 1);
}

#[test]
fn pop_best_fit_handles_empty() {
    let mut fl = FreeRunSet::new();
    assert_eq!(fl.pop_best_fit(1), None);
}

#[test]
fn runs_from_different_slots_do_not_merge() {
    let mut fl = FreeRunSet::new();
    fl.insert_with_merge(make_run(0, 0, 100, 0.5));
    fl.insert_with_merge(make_run(1, 100, 100, 0.5)); // different slot!
    assert_eq!(fl.len(), 2);
}

#[test]
fn insert_with_merge_coalesces_left_neighbour() {
    let mut fl = FreeRunSet::new();
    fl.insert_with_merge(make_run(0, 0, 100, 0.5));
    fl.insert_with_merge(make_run(0, 100, 50, 0.8));

    assert_eq!(fl.len(), 1);
    let got = fl.pop_best_fit(1).expect("pop");
    assert_eq!(got.start_weight, 0);
    assert_eq!(got.length_in_weights, 150);
    assert!(
        (got.max_ceiling - 0.8).abs() < f32::EPSILON,
        "merged max = max(0.5, 0.8) = 0.8; got {}",
        got.max_ceiling,
    );
}

#[test]
fn insert_with_merge_coalesces_right_neighbour() {
    let mut fl = FreeRunSet::new();
    fl.insert_with_merge(make_run(0, 100, 50, 0.3));
    // Free the run just before it.
    fl.insert_with_merge(make_run(0, 0, 100, 0.9));

    assert_eq!(fl.len(), 1);
    let got = fl.pop_best_fit(1).expect("pop");
    assert_eq!(got.start_weight, 0);
    assert_eq!(got.length_in_weights, 150);
    assert!((got.max_ceiling - 0.9).abs() < f32::EPSILON);
}

#[test]
fn insert_with_merge_three_way_coalesce() {
    let mut fl = FreeRunSet::new();
    fl.insert_with_merge(make_run(0, 0, 100, 0.3));
    fl.insert_with_merge(make_run(0, 150, 100, 0.8));
    // Fill the gap: run 100..150, length 50. Should merge all three.
    fl.insert_with_merge(make_run(0, 100, 50, 0.5));

    assert_eq!(fl.len(), 1);
    let got = fl.pop_best_fit(1).expect("pop");
    assert_eq!(got.start_weight, 0);
    assert_eq!(got.length_in_weights, 250);
    assert!(
        (got.max_ceiling - 0.8).abs() < f32::EPSILON,
        "max should be 0.8"
    );
}

#[test]
fn insert_with_merge_no_adjacency_keeps_separate() {
    let mut fl = FreeRunSet::new();
    fl.insert_with_merge(make_run(0, 0, 100, 0.5));
    fl.insert_with_merge(make_run(0, 200, 100, 0.5)); // gap of 100 weights
    assert_eq!(fl.len(), 2);
}

#[test]
fn ties_broken_by_weightref() {
    // Two runs with identical max_ceiling. pop_best_fit should pick
    // deterministically (by weight-ordering) so tests of the free
    // list behaviour are stable.
    let mut fl = FreeRunSet::new();
    fl.insert(make_run(0, 200, 100, 0.5));
    fl.insert(make_run(0, 0, 100, 0.5));
    let got = fl.pop_best_fit(50).expect("pop");
    assert_eq!(got.start_weight, 0, "tie broken by lowest WeightRef");
}

// ------------------------------------------------------------------
// reserve_weight
// ------------------------------------------------------------------

/// Build a one-slot CeilingSummary where every bucket has max=1.0,
/// matching a cover where we don't care about the actual magnitudes
/// for these tests — we just want reserve_weight's split logic to
/// have *some* ceiling to query.
fn flat_summary(bucket_counts: &[u32]) -> CeilingSummary {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"CSUM");
    bytes.push(1);
    bytes.extend_from_slice(&[0, 0, 0]);
    bytes.extend_from_slice(&(bucket_counts.len() as u32).to_le_bytes());
    for &c in bucket_counts {
        bytes.extend_from_slice(&c.to_le_bytes());
    }
    for &c in bucket_counts {
        for _ in 0..c {
            bytes.extend_from_slice(&1.0_f32.to_le_bytes());
        }
    }
    CeilingSummary::deserialize(&bytes).expect("summary")
}

#[test]
fn reserve_weight_in_middle_splits_into_two_halves() {
    let mut fl = FreeRunSet::new();
    fl.insert(make_run(0, 0, 100, 0.5));
    let summary = flat_summary(&[1]); // 1 slot, 1 bucket covering ≥ 100 weights

    fl.reserve_weight(0, 42, &summary, &SalienceTable::empty()).expect("reserve");
    assert_eq!(fl.len(), 2, "run split into left + right");
    // Left half: [0, 42). Right half: [43, 100).
    let left = fl.pop_best_fit(40).expect("left fits");
    let right = fl.pop_best_fit(1).expect("right fits");
    // Determinism: pop_best_fit picks by (max, length, slot, start).
    // Both halves have max=1.0 (from the flat summary), so shorter
    // wins first — length 42 vs 57 means left goes first.
    let (l, r) = if left.length_in_weights < right.length_in_weights {
        (left, right)
    } else {
        (right, left)
    };
    assert_eq!(l.start_weight, 0);
    assert_eq!(l.length_in_weights, 42);
    assert_eq!(r.start_weight, 43);
    assert_eq!(r.length_in_weights, 57);
}

#[test]
fn reserve_weight_at_run_start_shrinks_from_left() {
    let mut fl = FreeRunSet::new();
    fl.insert(make_run(0, 0, 100, 0.5));
    let summary = flat_summary(&[1]);

    fl.reserve_weight(0, 0, &summary, &SalienceTable::empty()).expect("reserve");
    assert_eq!(fl.len(), 1);
    let got = fl.pop_best_fit(1).unwrap();
    assert_eq!(got.start_weight, 1);
    assert_eq!(got.length_in_weights, 99);
}

#[test]
fn reserve_weight_at_run_end_shrinks_from_right() {
    let mut fl = FreeRunSet::new();
    fl.insert(make_run(0, 0, 100, 0.5));
    let summary = flat_summary(&[1]);

    fl.reserve_weight(0, 99, &summary, &SalienceTable::empty()).expect("reserve");
    assert_eq!(fl.len(), 1);
    let got = fl.pop_best_fit(1).unwrap();
    assert_eq!(got.start_weight, 0);
    assert_eq!(got.length_in_weights, 99);
}

#[test]
fn reserve_weight_on_single_weight_run_empties_it() {
    let mut fl = FreeRunSet::new();
    fl.insert(make_run(0, 10, 1, 0.5));
    let summary = flat_summary(&[1]);

    fl.reserve_weight(0, 10, &summary, &SalienceTable::empty()).expect("reserve");
    assert!(fl.is_empty());
}

#[test]
fn reserve_weight_not_in_any_run_errors() {
    let mut fl = FreeRunSet::new();
    fl.insert(make_run(0, 100, 50, 0.5));
    let summary = flat_summary(&[1]);

    // Before the run.
    let err = fl.reserve_weight(0, 50, &summary, &SalienceTable::empty()).expect_err("not free");
    assert_eq!(
        err,
        ReserveError::NotFree {
            slot: 0,
            weight_index: 50
        }
    );
    // After the run.
    let err = fl.reserve_weight(0, 200, &summary, &SalienceTable::empty()).expect_err("not free");
    assert_eq!(
        err,
        ReserveError::NotFree {
            slot: 0,
            weight_index: 200
        }
    );
    // Wrong slot.
    let err = fl.reserve_weight(1, 100, &summary, &SalienceTable::empty()).expect_err("not free");
    assert_eq!(
        err,
        ReserveError::NotFree {
            slot: 1,
            weight_index: 100
        }
    );
}

#[test]
fn reserve_many_weights_produces_many_small_runs() {
    // Start with one big run, reserve 10 scattered positions,
    // end up with 11 gaps (or fewer if any land at a boundary).
    let mut fl = FreeRunSet::new();
    fl.insert(make_run(0, 0, 1000, 0.5));
    let summary = flat_summary(&[4]); // 4 buckets × 256 = 1024 weights covered

    for w in [5_u32, 100, 200, 300, 400, 500, 600, 700, 800, 900] {
        fl.reserve_weight(0, w, &summary, &SalienceTable::empty()).expect("reserve");
    }
    assert_eq!(
        fl.len(),
        11,
        "10 reservations → 11 gaps in the original run"
    );
}

// ------------------------------------------------------------------

#[test]
fn pop_leaves_oversized_run_intact() {
    // pop_best_fit returns the whole picked run even if it's larger
    // than requested — the allocator's split logic handles trimming.
    let mut fl = FreeRunSet::new();
    fl.insert(make_run(0, 0, 500, 0.1));
    let got = fl.pop_best_fit(100).expect("pop");
    assert_eq!(got.length_in_weights, 500);
    assert!(fl.is_empty());
}
