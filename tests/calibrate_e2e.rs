//! End-to-end calibration test: run the bundled corpus through the
//! forward pass, commit salience to a real SmolLM2 cover, and
//! verify `status` reflects the calibration.
//!
//! Skipped when `models/smollm2-135m-f16.gguf` isn't present (same
//! pattern as other forward-pass tests). `#[ignore]`'d because a
//! full 512-token forward pass through 30 layers on CPU is
//! multi-minute.

use std::path::{Path, PathBuf};

use llmdb::calibrate::{DEFAULT_CALIBRATION_CORPUS, run_calibration};
use llmdb::diagnostics::gather;
use llmdb::gguf::parser::parse_path;
use llmdb::stego::planner::{AllocationMode, build_allocation_plan};
use llmdb::stego::tensor_map::TensorMap;
use llmdb::v2::fs::Filesystem;

const SMOLLM2: &str = "models/smollm2-135m-f16.gguf";

fn stage_model() -> Option<PathBuf> {
    if !Path::new(SMOLLM2).exists() {
        eprintln!("skipping: {SMOLLM2} not present");
        return None;
    }
    // Copy to a tmp file so init + commit can mutate without
    // touching the source fixture. SmolLM2-135M-F16 is ~270 MB so
    // the copy is small for WSL/Linux tmp budgets.
    let tmp = std::env::temp_dir().join("llmdb_calibrate_e2e.gguf");
    std::fs::copy(SMOLLM2, &tmp).ok()?;
    Some(tmp)
}

fn tensor_map_for(path: &Path) -> TensorMap {
    let gguf = parse_path(path).expect("parse gguf");
    let plan = build_allocation_plan(&gguf.tensors, AllocationMode::Standard);
    TensorMap::from_allocation_plan_with_base(&plan, gguf.tensor_data_offset as u64)
}

#[test]
#[ignore = "slow: 512-token forward through 30 layers (multi-minute on CPU). \
            Run with --ignored when validating B4b changes."]
fn calibrate_populates_salience_inode_and_status_reflects_it() {
    let Some(path) = stage_model() else {
        return;
    };
    let map = tensor_map_for(&path);

    // Init on a fresh copy.
    {
        let cover = unsafe {
            memmap2::MmapOptions::new().map_mut(
                &std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&path)
                    .expect("open"),
            )
        }
        .expect("mmap");
        let fs = Filesystem::init(cover, map.clone()).expect("init");
        drop(fs); // flushes on drop
    }

    // Mount + calibrate + check summary.
    let cover = unsafe {
        memmap2::MmapOptions::new().map_mut(
            &std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .expect("open"),
        )
    }
    .expect("mmap");
    let mut fs = Filesystem::mount(cover, map.clone()).expect("mount");

    // Pre-calibration: diagnostics report calibrated=false.
    let pre = gather(&fs, &map).expect("gather pre");
    assert!(
        !pre.calibrated,
        "cover is calibrated before we ran calibrate"
    );
    assert_eq!(pre.salience_slot_count, 0);

    let summary =
        run_calibration(&mut fs, &path, &map, DEFAULT_CALIBRATION_CORPUS).expect("calibrate");
    eprintln!(
        "calibration: {} tokens, {}/{} slots",
        summary.token_count, summary.populated_slot_count, summary.total_slot_count,
    );
    assert!(summary.token_count > 0);
    assert!(summary.new_salience_inode_nonzero);
    // At least the linear tensors in the 30 blocks should populate —
    // 30 blocks × 7 linear tensors each = 210 populated slots. The
    // total slot count includes the tensor map's full set (which
    // may be a superset if the planner exposes more than just the
    // linear weights).
    assert!(
        summary.populated_slot_count >= 200,
        "expected ≥ 200 populated slots, got {}",
        summary.populated_slot_count,
    );

    // Post-calibration: diagnostics report calibrated=true.
    let post = gather(&fs, &map).expect("gather post");
    assert!(post.calibrated);
    assert_eq!(
        post.salience_slot_count as usize,
        summary.populated_slot_count
    );
    let persisted = fs
        .load_salience()
        .expect("load salience")
        .expect("some salience");
    assert!(
        persisted.encode().len() < (1 << 20),
        "smollm2 salience should stay under 1 MiB, got {} bytes",
        persisted.encode().len(),
    );

    // Cleanup: drop + remove the staged copy.
    drop(fs);
    let _ = std::fs::remove_file(&path);
}
