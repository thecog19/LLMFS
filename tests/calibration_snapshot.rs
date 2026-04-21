//! Snapshot test for the magnitude-only ranking. Locks in the exact
//! bottom-N weight ordering produced against a reference cover file,
//! so any change in `lowest_magnitude_weights` (bucket boundaries,
//! tiebreakers, K-quant decoders, selection algorithm) shows up as
//! a concrete diff instead of a silent drift.
//!
//! What this catches:
//! - Bucket boundary arithmetic breaking (rounding in `log2`, etc)
//! - K-quant decoder output drifting (a subtle off-by-one in
//!   `get_scale_min_k4`, a sign flip in the q6 bias)
//! - Platform / compiler changes in f32 arithmetic affecting the
//!   partition order at the cutoff bucket
//! - Tiebreaker behaviour in `select_nth_unstable_by` if the cmp
//!   function loses totality
//!
//! What this does NOT catch: the magnitude estimator going from
//! "correct" to "also-correct-but-different" — the snapshot is a
//! lock, not a spec. Updating the snapshot is cheap (set
//! `LLMDB_UPDATE_SNAPSHOT=1`) and should be a deliberate commit
//! whenever the ranking changes on purpose.
//!
//! Cover fidelity: the fixture includes the pristine file's byte
//! size as a fingerprint. If you have a different-sized file in the
//! expected path the test aborts before computing, so you never get
//! a stale-pristine-induced "diff" that misleads the debugger.

use std::fs::File;
use std::path::Path;

use llmdb::gguf::parser::parse_path;
use llmdb::stego::calibration::magnitude::{lowest_magnitude_weights, read_weight_abs};
use llmdb::stego::planner::{AllocationMode, build_allocation_plan};
use llmdb::stego::tensor_map::TensorMap;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};

const PRISTINE: &str = "models/pristine/smollm2-135m-f16.gguf";
const FIXTURE: &str = "tests/fixtures/calibration/smollm2-135m-bottom-1000.json";
const N: usize = 1000;
// Fingerprint from models/pristine/MANIFEST.sha256 — size of the
// F16 cover, used as a cheap "are we looking at the same file?"
// check. Mismatches here mean the pristine file on disk drifted
// from the provenance in models/pristine/README.md, not that the
// ranking is wrong.
const EXPECTED_COVER_BYTES: u64 = 270_885_952;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Entry {
    slot_index: u32,
    weight_index: u64,
    /// f32 bits, not the value — this avoids "0.12300001 vs
    /// 0.12300000" drift in JSON parse round-trips, and gives an
    /// exact byte-for-byte comparison against the fixture.
    mag_bits: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Snapshot {
    cover_path: String,
    cover_bytes: u64,
    n: usize,
    /// Captured for context only — useful when diagnosing a diff.
    /// Not in the equality check because it's just total weight
    /// count across eligible slots (derived from cover + planner).
    total_weights_ranked: u64,
    entries: Vec<Entry>,
}

fn compute_snapshot() -> Option<Snapshot> {
    if !Path::new(PRISTINE).exists() {
        return None;
    }
    let meta = std::fs::metadata(PRISTINE).expect("stat pristine");
    assert_eq!(
        meta.len(),
        EXPECTED_COVER_BYTES,
        "pristine cover at {PRISTINE} is {} bytes, expected {EXPECTED_COVER_BYTES} — \
         check models/pristine/MANIFEST.sha256 and redownload if needed",
        meta.len(),
    );

    let parsed = parse_path(PRISTINE).expect("parse pristine");
    let plan = build_allocation_plan(&parsed.tensors, AllocationMode::Standard);
    let map = TensorMap::from_allocation_plan_with_base(&plan, parsed.tensor_data_offset as u64);

    let file = File::open(PRISTINE).expect("open pristine");
    let mmap = unsafe { Mmap::map(&file).expect("mmap pristine") };

    let ranked = lowest_magnitude_weights(&mmap[..], &map, N);
    assert_eq!(ranked.len(), N);
    let total_weights: u64 = map.slots.iter().map(|s| s.weight_count).sum();

    let entries: Vec<Entry> = ranked
        .iter()
        .map(|r| {
            let slot = &map.slots[r.slot_index as usize];
            let mag = read_weight_abs(&mmap[..], slot, r.weight_index);
            Entry {
                slot_index: r.slot_index,
                weight_index: r.weight_index,
                mag_bits: mag.to_bits(),
            }
        })
        .collect();

    Some(Snapshot {
        cover_path: PRISTINE.to_owned(),
        cover_bytes: meta.len(),
        n: N,
        total_weights_ranked: total_weights,
        entries,
    })
}

fn update_requested() -> bool {
    matches!(
        std::env::var("LLMDB_UPDATE_SNAPSHOT").ok().as_deref(),
        Some("1" | "true" | "yes"),
    )
}

#[test]
#[ignore = "needs pristine model + committed fixture; run with --ignored"]
fn bottom_1000_matches_committed_snapshot() {
    let Some(current) = compute_snapshot() else {
        eprintln!("skipping: {PRISTINE} not present");
        return;
    };

    if update_requested() {
        let json = serde_json::to_string_pretty(&current).expect("serialize snapshot");
        std::fs::write(FIXTURE, json).expect("write fixture");
        eprintln!(
            "wrote fresh snapshot to {FIXTURE} ({} entries)",
            current.entries.len()
        );
        return;
    }

    let expected_raw = std::fs::read_to_string(FIXTURE).unwrap_or_else(|e| {
        panic!(
            "could not read fixture at {FIXTURE}: {e}. \
             To create it: LLMDB_UPDATE_SNAPSHOT=1 cargo test --test \
             calibration_snapshot -- --ignored --exact \
             bottom_1000_matches_committed_snapshot"
        )
    });
    let expected: Snapshot =
        serde_json::from_str(&expected_raw).expect("fixture JSON malformed — regenerate");

    assert_eq!(
        expected.cover_bytes, current.cover_bytes,
        "snapshot cover_bytes mismatch — fixture is for a different cover"
    );
    assert_eq!(expected.n, current.n, "snapshot N mismatch");
    assert_eq!(
        expected.total_weights_ranked, current.total_weights_ranked,
        "total ranked weights changed ({} vs {}) — planner/eligibility changed; \
         regenerate snapshot if intentional",
        current.total_weights_ranked, expected.total_weights_ranked,
    );
    assert_eq!(
        expected.entries.len(),
        current.entries.len(),
        "entry count changed"
    );

    // Per-entry diff produces the most useful failure output: show
    // the first divergence instead of a wall of serialized JSON.
    for (i, (cur, exp)) in current
        .entries
        .iter()
        .zip(expected.entries.iter())
        .enumerate()
    {
        if cur != exp {
            let cur_mag = f32::from_bits(cur.mag_bits);
            let exp_mag = f32::from_bits(exp.mag_bits);
            panic!(
                "rank {i}: current = (slot={}, weight={}, |w|={cur_mag}) \
                 vs expected = (slot={}, weight={}, |w|={exp_mag}). \
                 If the new ranking is intentionally different, regenerate with \
                 LLMDB_UPDATE_SNAPSHOT=1",
                cur.slot_index, cur.weight_index, exp.slot_index, exp.weight_index,
            );
        }
    }
}
