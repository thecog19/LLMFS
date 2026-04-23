//! End-to-end calibration test: run the bundled corpus through the
//! forward pass, commit salience to a real SmolLM2 cover, and
//! verify `status` reflects the calibration.
//!
//! Skipped when the requested source model isn't present (same
//! pattern as other forward-pass tests). `#[ignore]`'d because a
//! full 512-token forward pass through 30 layers on CPU is
//! multi-minute.
//!
//! One test per quant type we care about:
//!   - F16: baseline (B4b gate, always preferred for the reference
//!     calibration run).
//!   - Q8_0: C1 gate — confirms the K-quant/Q8_0 dequant path
//!     drives AWQ + commit end-to-end on a quantized cover.

use std::path::{Path, PathBuf};

use llmdb::calibrate::{DEFAULT_CALIBRATION_CORPUS, run_calibration};
use llmdb::diagnostics::gather;
use llmdb::gguf::parser::parse_path;
use llmdb::stego::planner::{AllocationMode, build_allocation_plan};
use llmdb::stego::tensor_map::TensorMap;
use llmdb::v2::fs::Filesystem;

const SMOLLM2_F16: &str = "models/smollm2-135m-f16.gguf";
const SMOLLM2_Q8_0: &str = "models/smollm2-135m-q8_0.gguf";

/// Copy `src` to a unique tmp file so init + commit can mutate
/// without touching the source fixture. Returns the tmp path,
/// or `None` if the source is missing.
fn stage_model(src: &str, label: &str) -> Option<PathBuf> {
    if !Path::new(src).exists() {
        eprintln!("skipping: {src} not present");
        return None;
    }
    let tmp = std::env::temp_dir().join(format!("llmdb_calibrate_e2e_{label}.gguf"));
    std::fs::copy(src, &tmp).ok()?;
    Some(tmp)
}

fn tensor_map_for(path: &Path) -> TensorMap {
    let gguf = parse_path(path).expect("parse gguf");
    let plan = build_allocation_plan(&gguf.tensors, AllocationMode::Standard);
    TensorMap::from_allocation_plan_with_base(&plan, gguf.tensor_data_offset as u64)
}

/// Run the full calibrate-then-status cycle on a staged cover and
/// assert the structural gates. Shared between F16 and Q8_0 tests.
fn calibrate_and_assert(path: &Path, min_populated_slots: usize, max_salience_bytes: usize) {
    let map = tensor_map_for(path);

    // Init on a fresh copy.
    {
        let cover = unsafe {
            memmap2::MmapOptions::new().map_mut(
                &std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(path)
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
                .open(path)
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
        run_calibration(&mut fs, path, &map, DEFAULT_CALIBRATION_CORPUS).expect("calibrate");
    eprintln!(
        "calibration: {} tokens, {}/{} slots",
        summary.token_count, summary.populated_slot_count, summary.total_slot_count,
    );
    assert!(summary.token_count > 0);
    assert!(summary.new_salience_inode_nonzero);
    assert!(
        summary.populated_slot_count >= min_populated_slots,
        "expected ≥ {min_populated_slots} populated slots, got {}",
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
    let encoded_len = persisted.encode().len();
    assert!(
        encoded_len < max_salience_bytes,
        "smollm2 salience should stay under {max_salience_bytes} bytes, got {encoded_len}",
    );

    // Cleanup: drop + remove the staged copy.
    drop(fs);
}

#[test]
#[ignore = "slow: 512-token forward through 30 layers (multi-minute on CPU). \
            Run with --ignored when validating B4b / C1 changes."]
fn calibrate_f16_populates_salience_inode_and_status_reflects_it() {
    let Some(path) = stage_model(SMOLLM2_F16, "f16") else {
        return;
    };
    // 30 blocks × 7 linear tensors each = 210 populated slots.
    calibrate_and_assert(&path, 200, 1 << 20);
    let _ = std::fs::remove_file(&path);
}

#[test]
#[ignore = "slow: 512-token Q8_0 forward through 30 layers (multi-minute on CPU). \
            Run with --ignored when validating C1 changes."]
fn calibrate_q8_0_populates_salience_inode_and_status_reflects_it() {
    // C2 gate: calibration pipeline works unchanged on a Q8_0
    // cover. Dequant runs per-tensor at load, forward + AWQ are
    // identical, commit writes the same salience inode shape.
    // The populated-slot count should match F16's (210 linear
    // tensors in 30 blocks) since the allocation plan is quant-
    // agnostic.
    let Some(path) = stage_model(SMOLLM2_Q8_0, "q8_0") else {
        return;
    };
    calibrate_and_assert(&path, 200, 1 << 20);
    let _ = std::fs::remove_file(&path);
}
