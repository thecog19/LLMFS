//! One-off timing breakdown for the mount path. Times each phase so
//! we can see where the wall clock goes on a real cover.
//!
//! Usage: cargo run --release --example mount_timing -- <model.gguf>

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use llmdb::gguf::parser::parse_path as parse_gguf;
use llmdb::stego::calibration::magnitude::read_weight_ceiling_abs;
use llmdb::stego::planner::{AllocationMode, build_allocation_plan};
use llmdb::stego::tensor_map::TensorMap;
use llmdb::v2::ceiling::CeilingSummary;
use llmdb::v2::fs::Filesystem;

fn main() {
    let path: PathBuf = env::args()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: mount_timing <model.gguf>");

    let t0 = Instant::now();
    let parsed = parse_gguf(&path).expect("parse");
    eprintln!(
        "parse_gguf:              {:>8.3}s",
        t0.elapsed().as_secs_f64()
    );

    let t1 = Instant::now();
    let plan = build_allocation_plan(&parsed.tensors, AllocationMode::Standard);
    let map = TensorMap::from_allocation_plan_with_base(&plan, parsed.tensor_data_offset as u64);
    eprintln!(
        "build map ({:>3} slots):   {:>8.3}s",
        map.slots.len(),
        t1.elapsed().as_secs_f64()
    );

    let t2 = Instant::now();
    let cover = std::fs::read(&path).expect("read cover");
    eprintln!(
        "std::fs::read ({} MB): {:>8.3}s",
        cover.len() / 1_048_576,
        t2.elapsed().as_secs_f64()
    );

    // CeilingSummary::build cost in isolation (also done inside mount).
    let t3 = Instant::now();
    let total_weights: u64 = map.slots.iter().map(|s| s.weight_count).sum();
    let ceiling = CeilingSummary::build(&cover, &map);
    eprintln!(
        "CeilingSummary::build ({} weights): {:>8.3}s",
        total_weights,
        t3.elapsed().as_secs_f64()
    );
    drop(ceiling);

    // Re-do read_weight_ceiling_abs in a tight loop to confirm the
    // per-weight cost dominates.
    let t3b = Instant::now();
    let mut sample_max = 0.0_f32;
    let sample_slot = &map.slots[0];
    let take = sample_slot.weight_count.min(1_000_000);
    for w in 0..take {
        let c = read_weight_ceiling_abs(&cover, sample_slot, w);
        if c > sample_max {
            sample_max = c;
        }
    }
    eprintln!(
        "read_weight_ceiling_abs ×{:>7}:    {:>8.3}s ({:.0} ns/call, sample_max={})",
        take,
        t3b.elapsed().as_secs_f64(),
        t3b.elapsed().as_nanos() as f64 / take as f64,
        sample_max
    );

    let t4 = Instant::now();
    let fs = Filesystem::mount(cover, map.clone()).expect("mount");
    eprintln!(
        "Filesystem::mount (full): {:>8.3}s",
        t4.elapsed().as_secs_f64()
    );

    eprintln!(
        "  generation={} dedup_entries={} dirty_bits_set={}",
        fs.generation(),
        fs.dedup_index().len(),
        fs.dirty_bitmap().set_count(),
    );

    eprintln!(
        "total wall:             {:>8.3}s",
        t0.elapsed().as_secs_f64()
    );
}
