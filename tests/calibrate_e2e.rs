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

use llmdb::calibrate::{
    CalibrationMode, DEFAULT_CALIBRATION_CORPUS, MAX_HESSIAN_TOKENS, run_calibration,
    run_full_forward,
};
use llmdb::diagnostics::gather;
use llmdb::forward::{ActivationSite, ForwardModel};
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
/// assert the structural gates. Shared between F16 / Q8_0 and
/// Fast / Full tests.
fn calibrate_and_assert(
    path: &Path,
    min_populated_slots: usize,
    max_salience_bytes: usize,
    mode: CalibrationMode,
) {
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

    let summary = run_calibration(
        &mut fs,
        path,
        &map,
        DEFAULT_CALIBRATION_CORPUS,
        mode,
    )
    .expect("calibrate");
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
    calibrate_and_assert(&path, 200, 1 << 20, CalibrationMode::Fast);
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
    calibrate_and_assert(&path, 200, 1 << 20, CalibrationMode::Fast);
    let _ = std::fs::remove_file(&path);
}

#[test]
#[ignore = "slow: D1-a gate — full-Hessian calibration on SmolLM2-Q8_0. \
            Runs a 2048-token forward pass, accumulates per-site full H \
            in RAM, Cholesky-factorizes, extracts OBS saliency \
            (1/diag(H⁻¹)) per channel, commits to salience inode. \
            ~7 min on CPU plus Cholesky+OBS extraction work. Run with \
            --ignored when validating D1-a."]
fn calibrate_full_q8_0_populates_salience_inode() {
    // D1-a gate: the Full calibration path runs end-to-end on a
    // real Q8_0 cover. Same structural expectations as the Fast
    // path (200+ populated slots for the 210 linears, <1 MiB
    // encoded salience — the inode's PeriodicSlotSalience encoder
    // is value-agnostic), but the values inside are OBS saliencies
    // per `compensation-design.md §1.2` rather than AWQ means.
    let Some(path) = stage_model(SMOLLM2_Q8_0, "q8_0_full") else {
        return;
    };
    calibrate_and_assert(&path, 200, 1 << 20, CalibrationMode::Full);
    let _ = std::fs::remove_file(&path);
}

#[test]
#[ignore = "slow: D1-b gate — verifies run_full_forward exposes the \
            per-(site, layer) Cholesky factor cache (compensation-design.md \
            §3.2) alongside OBS saliency. Same ~7-min forward pass as the \
            D1-a gate; no filesystem mutation. Run with --ignored when \
            validating D1-b."]
fn full_forward_returns_populated_factor_cache_on_smollm2_q8_0() {
    // D1-b gate: confirms that Phase E consumers can read the L2
    // tier factors straight off a Full calibration run — no
    // re-computation, no re-factorization. Read-only against the
    // source fixture (no init/mount), so staging a copy is
    // unnecessary.
    if !Path::new(SMOLLM2_Q8_0).exists() {
        eprintln!("skipping: {SMOLLM2_Q8_0} not present");
        return;
    }
    let model = ForwardModel::load(SMOLLM2_Q8_0).expect("load q8_0 model");
    let hidden = model.config.hidden_dim;
    let ffn_dim = model.config.ffn_dim;
    let n_layers = model.config.n_layers;

    let mut tokens = model
        .encode(DEFAULT_CALIBRATION_CORPUS)
        .expect("encode default corpus");
    tokens.truncate(MAX_HESSIAN_TOKENS);
    assert!(
        tokens.len() >= ffn_dim,
        "corpus produced {} tokens; need ≥ ffn_dim ({ffn_dim}) for full-rank H",
        tokens.len(),
    );
    let token_count = tokens.len();

    let result = run_full_forward(&model, &tokens, token_count).expect("full forward");
    let factors = result.factors;

    // Four observation sites × n_layers blocks = expected cache size.
    let expected_entries = 4 * n_layers;
    assert_eq!(
        factors.len(),
        expected_entries,
        "expected {expected_entries} cached (site, layer) factors, got {}",
        factors.len(),
    );
    assert!(!factors.is_empty());
    assert!(factors.bytes_resident() > 0);

    // Each factor's N matches the site's input dim; each has a
    // valid lower-triangle length; the Cholesky invariant (positive
    // diagonal) must hold on every row.
    for (site, layer, factor) in factors.iter() {
        let expected_n = match site {
            ActivationSite::FfnDownInput => ffn_dim,
            _ => hidden,
        };
        assert_eq!(
            factor.n, expected_n,
            "factor for ({site:?}, layer {layer}): n = {}, expected {expected_n}",
            factor.n,
        );
        assert_eq!(
            factor.l.len(),
            expected_n * (expected_n + 1) / 2,
            "factor for ({site:?}, layer {layer}): wrong packed length",
        );
        for i in 0..factor.n {
            let diag = factor.l[i * (i + 1) / 2 + i];
            assert!(
                diag > 0.0 && diag.is_finite(),
                "factor for ({site:?}, layer {layer}): diag[{i}] = {diag} \
                 violates Cholesky invariant (expected > 0, finite)",
            );
        }
    }

    // And OBS came along too, keyed per tensor (7 linears per block).
    let expected_tensor_entries = 7 * n_layers;
    assert_eq!(
        result.per_tensor_obs.len(),
        expected_tensor_entries,
        "expected {expected_tensor_entries} per-tensor OBS entries, got {}",
        result.per_tensor_obs.len(),
    );
}
