//! Does the math layer actually work on a real SmolLM2 Q8_0 Hessian,
//! not just hand-picked 2×2 / 3×3 test matrices?
//!
//! Reads one of the dumped H files from `target/hessian-dump/` (produced
//! by the D0 measurement harness in `tests/hessian_measure.rs`) and
//! runs the full Cholesky → h_inv_column → compensation_operator
//! pipeline on it at realistic scale (N = 576 and N = 1536). Checks
//! numerical quality against the optimality condition: for any forced
//! `Δ_R`, applying `Δ_C = M_R · Δ_R` should make `H · Δ` vanish at every
//! free channel, to some tolerance.
//!
//! Skipped when the dump directory isn't present. `#[ignore]`'d because
//! the 1536-dim ffn_down_input case spends ~10–30 seconds on the
//! compensation_operator-on-a-100-weight-region path (O(r · n²) for
//! h_inv_column × r + O(|C| · r²) for the solve, with r=100 and
//! |C|=1436 on N=1536 this lands around a billion operations).

use std::fs;
use std::path::Path;

use llmdb::forward::compensation::{CompensationOperator, compensation_operator};
use llmdb::forward::hessian::upper_tri_offset;
use llmdb::forward::linalg::{cholesky, obs_saliency, pivoted_cholesky};

const DUMP_ROOT: &str = "target/hessian-dump/smollm2-135m-q8_0";

/// Load an upper-triangle-packed f32 H from disk. `n*(n+1)/2` F32 LE.
fn load_h_upper(file: &str, n: usize) -> Option<Vec<f32>> {
    let path = Path::new(DUMP_ROOT).join(file);
    if !path.exists() {
        return None;
    }
    let expected = n * (n + 1) / 2 * 4;
    let bytes = fs::read(&path).ok()?;
    assert_eq!(
        bytes.len(),
        expected,
        "{}: expected {expected} bytes, got {}",
        path.display(),
        bytes.len()
    );
    let mut h = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
        h.push(f32::from_le_bytes(arr));
    }
    Some(h)
}

fn h_entry(h: &[f32], n: usize, i: usize, j: usize) -> f64 {
    let (a, b) = if i <= j { (i, j) } else { (j, i) };
    h[upper_tri_offset(n, a, b)] as f64
}

/// Sum |x_i| — used as a scale for relative error bounds.
fn l1_norm(v: &[f32]) -> f64 {
    v.iter().map(|x| x.abs() as f64).sum()
}

/// Run the full math-layer stack on a single (site, layer) H and print
/// quality numbers. Checked assertions are loose on purpose — the point
/// is "does it work at the scale and conditioning we see on real
/// activations," not "perfect numerical identity."
fn exercise_full_pipeline(label: &str, h: &[f32], n: usize, region_size: usize) {
    // ── 1. Full Cholesky ───────────────────────────────────────────
    let t0 = std::time::Instant::now();
    let l = cholesky(h, n).expect("cholesky on real H should succeed");
    eprintln!("[{label}] cholesky(n={n}) in {:.1?}", t0.elapsed());
    assert_eq!(l.len(), h.len(), "L packed length should match H");

    // ── 2. OBS saliency, cross-check diag is positive and non-trivial ──
    let t0 = std::time::Instant::now();
    let obs = obs_saliency(&l, n);
    eprintln!("[{label}] obs_saliency(n={n}) in {:.1?}", t0.elapsed());
    let obs_min = obs.iter().copied().fold(f32::INFINITY, f32::min);
    let obs_max = obs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        obs_min > 0.0 && obs_min.is_finite(),
        "[{label}] obs_saliency min = {obs_min} should be positive and finite"
    );
    assert!(
        obs_max.is_finite() && obs_max < 1e12,
        "[{label}] obs_saliency max = {obs_max} looks pathological"
    );
    assert!(
        (obs_max / obs_min) > 1.1,
        "[{label}] obs_saliency collapsed to a single value — spread {obs_max}/{obs_min}"
    );
    eprintln!(
        "[{label}] obs_saliency spread: min={obs_min:.3e}, max={obs_max:.3e}, ratio={:.2e}",
        obs_max / obs_min
    );

    // ── 3. Pivoted Cholesky at 5% Frobenius tolerance ──────────────
    let t0 = std::time::Instant::now();
    let pc = pivoted_cholesky(h, n, 0.05, n).expect("pivoted_cholesky on real H");
    eprintln!(
        "[{label}] pivoted_cholesky(n={n}, tol=0.05): rank={} in {:.1?}",
        pc.rank(),
        t0.elapsed()
    );
    assert!(
        pc.rank() >= 1,
        "[{label}] pivoted cholesky extracted 0 columns — PSD failure?"
    );
    assert!(
        pc.rank() <= n,
        "[{label}] pivoted cholesky rank {} exceeds n = {n}",
        pc.rank()
    );
    // Cross-check: V V^T diagonal entries should be close to H diagonal.
    // Compute residual diag trace as a sanity check.
    let mut residual_diag_sum = 0.0_f64;
    for i in 0..n {
        let mut vvt_ii = 0.0_f64;
        for k in 0..pc.rank() {
            let v_ik = pc.v[k * n + i] as f64;
            vvt_ii += v_ik * v_ik;
        }
        let h_ii = h_entry(h, n, i, i);
        residual_diag_sum += (h_ii - vvt_ii).max(0.0);
    }
    let initial_trace: f64 = (0..n).map(|i| h_entry(h, n, i, i)).sum();
    eprintln!(
        "[{label}] pivoted cholesky residual diag sum = {:.3e}, initial trace = {:.3e}, \
         ratio = {:.4}",
        residual_diag_sum,
        initial_trace,
        residual_diag_sum / initial_trace
    );

    // ── 4. Compensation operator on a picked region ────────────────
    // Pick region_size distinct channels — spread across N to avoid
    // hitting only adjacent indices.
    let region: Vec<usize> = (0..region_size)
        .map(|k| k * (n / region_size).max(1))
        .collect();
    let t0 = std::time::Instant::now();
    let op: CompensationOperator =
        compensation_operator(&l, n, &region).expect("compensation_operator on real H");
    eprintln!(
        "[{label}] compensation_operator(n={n}, r={region_size}) in {:.1?}",
        t0.elapsed()
    );
    assert_eq!(op.region_size(), region_size);
    assert_eq!(op.free_size(), n - region_size);

    // ── 5. Apply a randomish Δ_R, verify H · Δ ≈ 0 at free channels ──
    // Pseudo-randomness: a sinusoid over region_size, avoids all-zero
    // or hand-picked degenerate inputs.
    let delta_r: Vec<f32> = (0..region_size)
        .map(|k| ((k as f32) * 1.7).sin() * 1.0 + 0.3)
        .collect();
    let delta_c = op.apply(&delta_r);

    // Assemble full Δ.
    let mut delta = vec![0.0_f32; n];
    for (i, &rj) in op.region.iter().enumerate() {
        delta[rj] = delta_r[i];
    }
    for (i, &cj) in op.c_indices.iter().enumerate() {
        delta[cj] = delta_c[i];
    }

    // H · Δ residual: should be ≈ 0 at every c ∈ C.
    let mut max_residual_c: f64 = 0.0;
    let mut max_h_delta_j: f64 = 0.0;
    for c in &op.c_indices {
        let mut sum = 0.0_f64;
        for (k, &delta_k) in delta.iter().enumerate() {
            sum += h_entry(h, n, *c, k) * delta_k as f64;
        }
        max_residual_c = max_residual_c.max(sum.abs());
    }
    for j in &op.region {
        let mut sum = 0.0_f64;
        for (k, &delta_k) in delta.iter().enumerate() {
            sum += h_entry(h, n, *j, k) * delta_k as f64;
        }
        max_h_delta_j = max_h_delta_j.max(sum.abs());
    }
    // Scale: the "size" of the forced perturbation in H-norm terms.
    let h_delta_scale = max_h_delta_j.max(l1_norm(&delta) * 1e-3); // floor
    let relative_error = max_residual_c / h_delta_scale;
    eprintln!(
        "[{label}] |H·Δ|∞ at free channels: {:.3e}, at forced: {:.3e}, relative: {:.3e}",
        max_residual_c, max_h_delta_j, relative_error
    );
    assert!(
        relative_error < 1e-2,
        "[{label}] compensation residual at free channels too large: {:.3e} / {:.3e} \
         (relative = {:.3e}). Math layer is not landing at the expected precision.",
        max_residual_c,
        h_delta_scale,
        relative_error
    );
}

#[test]
#[ignore = "validation: loads a real dumped H (SmolLM2 Q8_0 layer 0 \
            qkv_input, N=576) and exercises the full math-layer stack \
            (cholesky → obs_saliency → pivoted_cholesky → \
            compensation_operator). Skipped gracefully if \
            target/hessian-dump/ is absent. Run with --ignored when \
            validating the linalg / compensation primitives on real data."]
fn math_layer_on_real_qkv_input_layer_0() {
    let n = 576;
    let Some(h) = load_h_upper("00_qkv_input.f32", n) else {
        eprintln!("skipping: {DUMP_ROOT}/00_qkv_input.f32 not present");
        return;
    };
    exercise_full_pipeline("qkv_input/layer=0", &h, n, 32);
}

#[test]
#[ignore = "validation: same as math_layer_on_real_qkv_input_layer_0 \
            but at the largest-N site (ffn_down_input, N=1536). Runs \
            tens of seconds on CPU due to O(r · n²) compensation ops."]
fn math_layer_on_real_ffn_down_input_layer_0() {
    let n = 1536;
    let Some(h) = load_h_upper("00_ffn_down_input.f32", n) else {
        eprintln!("skipping: {DUMP_ROOT}/00_ffn_down_input.f32 not present");
        return;
    };
    exercise_full_pipeline("ffn_down_input/layer=0", &h, n, 64);
}

#[test]
#[ignore = "validation: sweeps layer 0 of all four sites to catch \
            site-specific numerical surprises."]
fn math_layer_on_real_all_sites_layer_0() {
    let cases: &[(&str, &str, usize, usize)] = &[
        ("qkv_input/layer=0", "00_qkv_input.f32", 576, 16),
        (
            "attn_output_input/layer=0",
            "00_attn_output_input.f32",
            576,
            16,
        ),
        (
            "ffn_gate_up_input/layer=0",
            "00_ffn_gate_up_input.f32",
            576,
            16,
        ),
        ("ffn_down_input/layer=0", "00_ffn_down_input.f32", 1536, 48),
    ];
    let mut any_ran = false;
    for (label, file, n, r) in cases {
        if let Some(h) = load_h_upper(file, *n) {
            exercise_full_pipeline(label, &h, *n, *r);
            any_ran = true;
        } else {
            eprintln!("[{label}] skipping: {file} not present");
        }
    }
    if !any_ran {
        eprintln!("no dumps found at {DUMP_ROOT} — run tests/hessian_measure.rs first");
    }
}
