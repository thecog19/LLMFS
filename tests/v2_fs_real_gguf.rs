//! V2 Filesystem round-trip against a real pristine GGUF cover.
//!
//! Bytes-on-disk end-to-end: parse the GGUF, build a TensorMap
//! rooted at the real tensor-data offset, init the V2 filesystem
//! into a copy of the cover, write a payload, unmount, remount,
//! read back, assert the payload survives.
//!
//! Gated with `#[ignore]` because:
//! 1. The pristine fixture isn't on every dev machine.
//! 2. A 270 MB cover is slow compared to synthetic tests.
//!
//! Run with `cargo test --test v2_fs_real_gguf -- --ignored`.
//!
//! This is the smallest real-world sanity check that V2 works
//! end-to-end: parser → allocation plan → tensor map → anchor
//! placement (ceiling-magnitude over millions of weights) → alloc
//! → chunk I/O → commit → read. If this passes, V2's synthetic
//! tests aren't lying about the stack.

use std::path::Path;

use llmdb::gguf::parser::parse_path;
use llmdb::stego::planner::{AllocationMode, build_allocation_plan};
use llmdb::stego::tensor_map::TensorMap;
use llmdb::v2::fs::Filesystem;

const PRISTINE_F16: &str = "models/pristine/smollm2-135m-f16.gguf";

#[test]
#[ignore = "requires 270 MB pristine fixture; run with --ignored"]
fn smollm2_135m_f16_round_trips_through_v2_filesystem() {
    if !Path::new(PRISTINE_F16).exists() {
        eprintln!("skipping: {PRISTINE_F16} not present");
        return;
    }

    let parsed = parse_path(PRISTINE_F16).expect("parse pristine GGUF");
    let plan = build_allocation_plan(&parsed.tensors, AllocationMode::Standard);
    let map =
        TensorMap::from_allocation_plan_with_base(&plan, parsed.tensor_data_offset as u64);
    let cover = std::fs::read(PRISTINE_F16).expect("read pristine bytes");

    let total_weights: u64 = map.slots.iter().map(|s| s.weight_count).sum();
    eprintln!(
        "cover: {} bytes, {} eligible slots, {total_weights} eligible weights",
        cover.len(),
        map.slots.len(),
    );

    // Init a fresh V2 filesystem. Default 4 KB data chunks; at 12
    // direct pointers that's a 48 KB file cap — plenty for this test.
    let init_t = std::time::Instant::now();
    let mut fs = Filesystem::init(cover, map.clone()).expect("V2 init on real GGUF");
    eprintln!("init: {:?}", init_t.elapsed());

    // Write a mid-sized payload — multi-chunk under a 4 KB chunk size.
    let payload: Vec<u8> = (0..10_000_u32)
        .map(|i| (i.wrapping_mul(2654435761) & 0xFF) as u8)
        .collect();
    let write_t = std::time::Instant::now();
    fs.create_file("/data", &payload).expect("V2 write payload");
    eprintln!("write ({} bytes): {:?}", payload.len(), write_t.elapsed());
    assert_eq!(fs.read_file("/data").expect("read after write"), payload);

    // Round-trip across unmount + remount. This is the thing that
    // would have broken if ceiling-magnitude anchor placement didn't
    // survive on a real cover's magnitude distribution.
    let cover_after = fs.unmount().expect("unmount");
    let mount_t = std::time::Instant::now();
    let fs2 = Filesystem::mount(cover_after, map).expect("V2 mount on written cover");
    eprintln!("mount: {:?}", mount_t.elapsed());
    let readback = fs2.read_file("/data").expect("V2 read after remount");
    assert_eq!(
        readback, payload,
        "payload must survive unmount+remount on a real GGUF cover"
    );
    assert_eq!(fs2.generation(), 2, "init = gen 1, one write = gen 2");
}
