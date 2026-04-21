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

use llmdb::v2::freelist::{FreeRun, FreeRunSet};

fn make_run(slot: u16, start: u32, len: u32, max: f32) -> FreeRun {
    FreeRun {
        slot,
        start_weight: start,
        length_in_weights: len,
        max_ceiling: max,
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
    assert!((got.max_ceiling - 0.8).abs() < f32::EPSILON, "max should be 0.8");
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
