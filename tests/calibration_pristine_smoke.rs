//! Smoke test: run the magnitude-only salience estimator against a
//! real pristine model. Confirms the dispatch path works end-to-end
//! (parse GGUF → build TensorMap → mmap → rank weights) on actual
//! data, not just synthetic fixtures.
//!
//! Skipped gracefully if the pristine file isn't present, so the
//! test suite still passes on machines without the cache.

use std::fs::File;
use std::path::Path;

use llmdb::gguf::parser::parse_path;
use llmdb::stego::calibration::magnitude::{lowest_magnitude_weights, read_weight_abs};
use llmdb::stego::planner::{AllocationMode, build_allocation_plan};
use llmdb::stego::tensor_map::TensorMap;
use memmap2::Mmap;

const PRISTINE: &str = "models/pristine/smollm2-135m-f16.gguf";

#[test]
#[ignore = "scans 100M+ weights against a real GGUF; run with --ignored when verifying"]
fn ranks_pristine_smollm2_f16_lowest_thousand() {
    if !Path::new(PRISTINE).exists() {
        eprintln!("skipping: {PRISTINE} not present");
        return;
    }

    let parsed = parse_path(PRISTINE).expect("parse pristine GGUF");
    let plan = build_allocation_plan(&parsed.tensors, AllocationMode::Standard);
    let map = TensorMap::from_allocation_plan(&plan);

    let file = File::open(PRISTINE).expect("open pristine");
    let mmap = unsafe { Mmap::map(&file).expect("mmap") };

    let total_weights: u64 = map.slots.iter().map(|s| s.weight_count).sum();
    println!(
        "TensorMap: {} eligible slots, {total_weights} total weights, {} stego bytes",
        map.slots.len(),
        map.total_capacity_bytes
    );

    let n = 1000;
    let ranked = lowest_magnitude_weights(&mmap[..], &map, n);
    assert_eq!(ranked.len(), n);

    // Sanity: ranking is monotone non-decreasing in |w|.
    let mut last = 0.0_f32;
    for r in &ranked {
        let slot = &map.slots[r.slot_index as usize];
        let mag = read_weight_abs(&mmap[..], slot, r.weight_index);
        assert!(
            mag + 1e-9 >= last,
            "ranking not monotone: {mag} < {last} at slot {} weight {}",
            r.slot_index,
            r.weight_index
        );
        last = mag;
    }

    let smallest = {
        let r = &ranked[0];
        let slot = &map.slots[r.slot_index as usize];
        read_weight_abs(&mmap[..], slot, r.weight_index)
    };
    let largest_in_top_n = last;
    println!("lowest 1000 |w|: smallest={smallest:.6}, 1000th={largest_in_top_n:.6}",);

    // For a trained F16 model on wikitext-scale tasks, we expect lots of
    // near-zero weights — the smallest of the bottom 1000 should be
    // genuinely tiny (pretty much zero-ish) and the 1000th still small.
    assert!(
        smallest < 1e-3,
        "expected near-zero smallest weight, got {smallest}"
    );
    assert!(
        largest_in_top_n < 1.0,
        "expected the bottom 1000 to all be < 1.0, got {largest_in_top_n}"
    );
}
