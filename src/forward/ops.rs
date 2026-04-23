//! Core math primitives for the hand-rolled transformer forward pass.
//!
//! Every op takes plain `&[f32]` / `&mut [f32]` slices — no tensor
//! wrapper, no autograd, no allocation. Naive correctness-first
//! implementations; SIMD / rayon / fusion are Milestone-post
//! (see the plan's "Not in this plan" section).
//!
//! ## Shape conventions
//!
//! - Matrix storage is **row-major**.
//! - Weight tensors coming out of GGUF are stored as `[out_dim,
//!   in_dim]` — each row is one output's incoming weights. Our
//!   [`matmul`] expects this layout directly; no extra transpose.
//! - For a batch of `m` tokens through a `(in_dim, out_dim)` linear
//!   layer: `x` is `[m, in_dim]`, `w` is `[out_dim, in_dim]`, the
//!   output `y` is `[m, out_dim]`. `m = 1` for single-token
//!   decoding is the common case.
//!
//! ## RoPE convention
//!
//! Llama-family GGUFs ship Q/K projection weights permuted for
//! the `GGML_ROPE_TYPE_NORMAL` ("interleaved") rotation: each
//! pair `(x[2i], x[2i+1])` rotates by a per-band angle. `convert_
//! hf_to_gguf.py` does the permutation at conversion time, so we
//! use interleaved here — matching it mirror-for-mirror is the
//! only way A8's PPL gate lands. See llama.cpp
//! `ggml_compute_forward_rope_f32` for the reference.

/// `y[m, n] = x[m, k] @ w[n, k].T` — i.e. `y[i][j] = Σₗ x[i][l] ·
/// w[j][l]`.
///
/// Weight layout is `[out_dim, in_dim]` row-major (the GGUF
/// convention). F32 accumulator; naive triple loop.
///
/// # Panics
/// If the slice lengths don't match the declared shapes.
pub fn matmul(x: &[f32], w: &[f32], y: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(x.len(), m * k, "matmul: x len {} ≠ m*k {}", x.len(), m * k);
    assert_eq!(w.len(), n * k, "matmul: w len {} ≠ n*k {}", w.len(), n * k);
    assert_eq!(y.len(), m * n, "matmul: y len {} ≠ m*n {}", y.len(), m * n);
    for i in 0..m {
        let xi = &x[i * k..(i + 1) * k];
        let yi = &mut y[i * n..(i + 1) * n];
        for j in 0..n {
            let wj = &w[j * k..(j + 1) * k];
            let mut acc = 0.0_f32;
            for l in 0..k {
                acc += xi[l] * wj[l];
            }
            yi[j] = acc;
        }
    }
}

/// Root-mean-square norm with per-channel gain `w`, in place.
///
/// `x[i] ← (x[i] / rms) · w[i]` where `rms = √(mean(x²) + eps)`.
/// The mean is taken over all `x.len()` entries (per-row norm;
/// caller tiles it by row for batched inputs).
pub fn rmsnorm(x: &mut [f32], w: &[f32], eps: f32) {
    assert_eq!(
        x.len(),
        w.len(),
        "rmsnorm: x len {} ≠ w len {}",
        x.len(),
        w.len()
    );
    let n = x.len();
    if n == 0 {
        return;
    }
    let mut sumsq = 0.0_f32;
    for v in x.iter() {
        sumsq += v * v;
    }
    let rms = (sumsq / n as f32 + eps).sqrt();
    let inv = 1.0_f32 / rms;
    for (xi, wi) in x.iter_mut().zip(w.iter()) {
        *xi = *xi * inv * *wi;
    }
}

/// Rotary position embedding, in place. Rotates the first
/// `rope_dim` channels of every `head_dim`-sized head in `x` by
/// the angle `pos · θᵢ`, where `θᵢ = freq_base^(-2i / rope_dim)`
/// for `i ∈ 0..rope_dim/2`.
///
/// Pairs are interleaved: `(x[2i], x[2i+1])` rotate together.
///
/// `x` layout: `[n_heads, head_dim]` row-major (one head per
/// contiguous slice). `rope_dim` must be ≤ `head_dim` and even —
/// channels `rope_dim..head_dim` are left untouched (matches
/// `llama.rope.dimension_count < head_dim` configs, though the
/// two target models use `rope_dim == head_dim`).
pub fn rope(
    x: &mut [f32],
    pos: usize,
    freq_base: f32,
    head_dim: usize,
    rope_dim: usize,
    n_heads: usize,
) {
    assert_eq!(
        x.len(),
        n_heads * head_dim,
        "rope: x len {} ≠ n_heads*head_dim {}",
        x.len(),
        n_heads * head_dim,
    );
    assert!(
        rope_dim <= head_dim,
        "rope: rope_dim {rope_dim} > head_dim {head_dim}"
    );
    assert!(
        rope_dim.is_multiple_of(2),
        "rope: rope_dim {rope_dim} must be even"
    );
    let p = pos as f32;
    let half = rope_dim / 2;
    for h in 0..n_heads {
        let head = &mut x[h * head_dim..(h + 1) * head_dim];
        for i in 0..half {
            let exp = -2.0 * i as f32 / rope_dim as f32;
            let theta = p * freq_base.powf(exp);
            let (s, c) = theta.sin_cos();
            let a = head[2 * i];
            let b = head[2 * i + 1];
            head[2 * i] = a * c - b * s;
            head[2 * i + 1] = a * s + b * c;
        }
    }
}

/// Numerically stable softmax, in place. Subtracts the max before
/// exponentiation so inputs with large magnitudes don't overflow.
/// Result sums to 1.0 within float rounding.
///
/// An empty slice is a no-op (equivalent to softmax over an
/// empty axis — no sensible output).
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for v in x.iter() {
        if *v > max {
            max = *v;
        }
    }
    // All-`-inf` inputs shouldn't appear in practice (attention
    // never produces them); if they do, `max = -inf` and the loop
    // below would emit NaNs. Short-circuit to a uniform output.
    if !max.is_finite() {
        let uniform = 1.0_f32 / x.len() as f32;
        for v in x.iter_mut() {
            *v = uniform;
        }
        return;
    }
    let mut sum = 0.0_f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0_f32 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// SiLU (Swish-1) in place: `x ← x · σ(x) = x / (1 + e^-x)`.
pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0_f32 + (-*v).exp());
    }
}

/// SwiGLU: `out[i] = silu(gate[i]) · up[i]`. Used as the FFN
/// non-linearity in llama-family models (gate/up come from two
/// parallel linear projections of the block input).
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    assert_eq!(gate.len(), up.len());
    assert_eq!(gate.len(), out.len());
    for ((g, u), o) in gate.iter().zip(up.iter()).zip(out.iter_mut()) {
        let sig = 1.0_f32 / (1.0_f32 + (-*g).exp());
        *o = *g * sig * *u;
    }
}

/// Copy the embedding row for `token_id` from `table` into `out`.
/// `table` is `[vocab_size, dim]` row-major; `out.len()` must equal
/// `dim`.
pub fn embed(token_id: u32, table: &[f32], dim: usize, out: &mut [f32]) {
    assert_eq!(out.len(), dim, "embed: out len {} ≠ dim {}", out.len(), dim);
    let id = token_id as usize;
    let start = id * dim;
    let end = start + dim;
    assert!(
        end <= table.len(),
        "embed: token {token_id} out of table (size {})",
        table.len() / dim.max(1),
    );
    out.copy_from_slice(&table[start..end]);
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── matmul ─────────────────────────────────────────────────────

    #[test]
    fn matmul_1x2_2x3_matches_hand_compute() {
        // x = [[1, 2]], shape [1, 2]
        // w = [[3, 4], [5, 6], [7, 8]], shape [3, 2] (= out=3, in=2)
        // y should be x @ w.T = [[1*3+2*4, 1*5+2*6, 1*7+2*8]]
        //                     = [[11, 17, 23]]
        let x = [1.0_f32, 2.0];
        let w = [3.0_f32, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut y = [0.0_f32; 3];
        matmul(&x, &w, &mut y, 1, 2, 3);
        assert_eq!(y, [11.0, 17.0, 23.0]);
    }

    #[test]
    fn matmul_identity_weight_is_copy() {
        // Identity: w = I (n=k), y = x.
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let w = [
            1.0_f32, 0.0, 0.0, 0.0, 0.0_f32, 1.0, 0.0, 0.0, 0.0_f32, 0.0, 1.0, 0.0, 0.0_f32, 0.0,
            0.0, 1.0,
        ];
        let mut y = [0.0_f32; 4];
        matmul(&x, &w, &mut y, 1, 4, 4);
        assert_eq!(y, x);
    }

    #[test]
    fn matmul_batched_m_2() {
        // m=2, k=2, n=2 — two independent rows.
        let x = [1.0_f32, 1.0, 2.0, 3.0];
        let w = [1.0_f32, 0.0, 0.0, 1.0]; // identity
        let mut y = [0.0_f32; 4];
        matmul(&x, &w, &mut y, 2, 2, 2);
        assert_eq!(y, [1.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "matmul: x len")]
    fn matmul_panics_on_bad_x_shape() {
        let x = [1.0_f32];
        let w = [1.0_f32; 4];
        let mut y = [0.0_f32; 2];
        matmul(&x, &w, &mut y, 1, 2, 2);
    }

    // ─── rmsnorm ────────────────────────────────────────────────────

    #[test]
    fn rmsnorm_unit_gain_normalizes_to_unit_rms() {
        let mut x = [3.0_f32, 4.0];
        let w = [1.0_f32, 1.0];
        rmsnorm(&mut x, &w, 0.0);
        // mean(x²) = (9+16)/2 = 12.5; rms = √12.5
        let rms = 12.5_f32.sqrt();
        assert!((x[0] - 3.0 / rms).abs() < 1e-6);
        assert!((x[1] - 4.0 / rms).abs() < 1e-6);
        // Verify the result itself has unit RMS (gain = ones).
        let out_mean_sq = (x[0] * x[0] + x[1] * x[1]) / 2.0;
        assert!((out_mean_sq - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rmsnorm_gain_scales_output_per_channel() {
        let mut x = [1.0_f32, 1.0];
        let w = [2.0_f32, 5.0];
        rmsnorm(&mut x, &w, 0.0);
        // mean(x²) = 1, rms = 1, so output = x * w = [2, 5].
        assert!((x[0] - 2.0).abs() < 1e-6);
        assert!((x[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn rmsnorm_eps_prevents_division_by_zero() {
        let mut x = [0.0_f32, 0.0];
        let w = [1.0_f32, 1.0];
        rmsnorm(&mut x, &w, 1e-6);
        // rms = √eps ≈ 1e-3; x stays 0.
        assert_eq!(x, [0.0, 0.0]);
    }

    // ─── rope ───────────────────────────────────────────────────────

    #[test]
    fn rope_at_position_zero_is_identity() {
        // cos(0)=1, sin(0)=0 → no rotation.
        let mut x = [1.0_f32, 2.0, 3.0, 4.0];
        let snapshot = x;
        rope(&mut x, 0, 10_000.0, 4, 4, 1);
        assert_eq!(x, snapshot);
    }

    #[test]
    fn rope_first_pair_rotates_by_pos_times_base_pow_0() {
        // For i=0, θ = freq_base^0 = 1, so angle = pos * 1 = pos.
        // With pos=1, (1, 0) → (cos 1, sin 1).
        let mut x = [1.0_f32, 0.0, 0.0, 0.0];
        rope(&mut x, 1, 10_000.0, 4, 4, 1);
        assert!((x[0] - 1.0_f32.cos()).abs() < 1e-6);
        assert!((x[1] - 1.0_f32.sin()).abs() < 1e-6);
        // Second pair (i=1): angle = pos * 10000^(-2/4) = 1/100,
        // applied to (0, 0) → (0, 0).
        assert!(x[2].abs() < 1e-6);
        assert!(x[3].abs() < 1e-6);
    }

    #[test]
    fn rope_second_pair_gets_lower_frequency_band() {
        // head_dim=4, freq_base=10000. Pair i=1 uses
        // θ = 10000^(-2/4) = 10000^(-1/2) = 0.01 → angle = pos*0.01.
        // With pos=100, angle = 1.0. (1, 0) → (cos 1, sin 1).
        let mut x = [0.0_f32, 0.0, 1.0, 0.0];
        rope(&mut x, 100, 10_000.0, 4, 4, 1);
        assert!(x[0].abs() < 1e-6);
        assert!(x[1].abs() < 1e-6);
        assert!((x[2] - 1.0_f32.cos()).abs() < 1e-5);
        assert!((x[3] - 1.0_f32.sin()).abs() < 1e-5);
    }

    #[test]
    fn rope_partial_rotation_leaves_tail_unchanged() {
        // head_dim=6, rope_dim=4 → channels 4..6 untouched.
        let mut x = [1.0_f32, 0.0, 1.0, 0.0, 7.0, 8.0];
        rope(&mut x, 1, 10_000.0, 6, 4, 1);
        assert_eq!(x[4], 7.0);
        assert_eq!(x[5], 8.0);
    }

    #[test]
    fn rope_multi_head_processes_each_head_independently() {
        // Two heads, head_dim=2. Both start (1, 0).
        // With pos=1, angle = 1, both rotate to (cos 1, sin 1).
        let mut x = [1.0_f32, 0.0, 1.0, 0.0];
        rope(&mut x, 1, 10_000.0, 2, 2, 2);
        for pair in x.chunks_exact(2) {
            assert!((pair[0] - 1.0_f32.cos()).abs() < 1e-6);
            assert!((pair[1] - 1.0_f32.sin()).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_preserves_magnitude() {
        // Rotation is orthogonal: ‖x‖² invariant under RoPE.
        let mut x = [0.7_f32, -1.3, 2.1, 0.4, -0.9, 1.5, 0.2, -0.5];
        let mag0: f32 = x.iter().map(|v| v * v).sum();
        rope(&mut x, 42, 10_000.0, 4, 4, 2);
        let mag1: f32 = x.iter().map(|v| v * v).sum();
        assert!(
            (mag0 - mag1).abs() < 1e-4,
            "magnitude not preserved: {mag0} vs {mag1}",
        );
    }

    // ─── softmax ────────────────────────────────────────────────────

    #[test]
    fn softmax_sums_to_one() {
        let mut x = [1.0_f32, 2.0, 3.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_on_uniform_input_is_uniform() {
        let mut x = [5.0_f32; 4];
        softmax(&mut x);
        for v in x {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_is_numerically_stable_on_large_inputs() {
        // Naive softmax would exp(1000) → inf → NaN.
        let mut x = [1000.0_f32, 1000.0, 1000.0];
        softmax(&mut x);
        for v in x {
            assert!(v.is_finite(), "softmax exploded to {v}");
            assert!((v - 1.0_f32 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_matches_hand_compute_small_case() {
        // Inputs [0, ln 2, ln 3] → exp [1, 2, 3] / 6 = [1/6, 1/3, 1/2].
        let mut x = [0.0_f32, 2.0_f32.ln(), 3.0_f32.ln()];
        softmax(&mut x);
        assert!((x[0] - 1.0 / 6.0).abs() < 1e-6);
        assert!((x[1] - 1.0 / 3.0).abs() < 1e-6);
        assert!((x[2] - 1.0 / 2.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_empty_is_noop() {
        let mut x: [f32; 0] = [];
        softmax(&mut x);
    }

    // ─── silu / swiglu ──────────────────────────────────────────────

    #[test]
    fn silu_matches_hand_compute() {
        let mut x = [0.0_f32, 1.0, -1.0];
        silu(&mut x);
        assert!((x[0] - 0.0).abs() < 1e-6); // 0 * σ(0) = 0
        // 1 * σ(1) = 1 / (1+e^-1); -1 * σ(-1) = -1 / (1+e).
        let expect_pos = 1.0_f32 / (1.0 + (-1.0_f32).exp());
        let expect_neg = -1.0_f32 / (1.0 + 1.0_f32.exp());
        assert!((x[1] - expect_pos).abs() < 1e-6);
        assert!((x[2] - expect_neg).abs() < 1e-6);
    }

    #[test]
    fn swiglu_matches_hand_compute() {
        let gate = [1.0_f32, 2.0];
        let up = [3.0_f32, 4.0];
        let mut out = [0.0_f32; 2];
        swiglu(&gate, &up, &mut out);
        let s1 = 1.0_f32 / (1.0 + (-1.0_f32).exp());
        let s2 = 1.0_f32 / (1.0 + (-2.0_f32).exp());
        assert!((out[0] - 1.0 * s1 * 3.0).abs() < 1e-6);
        assert!((out[1] - 2.0 * s2 * 4.0).abs() < 1e-6);
    }

    // ─── embed ──────────────────────────────────────────────────────

    #[test]
    fn embed_copies_the_requested_row() {
        let table = [0.0_f32, 0.1, 0.2, 1.0_f32, 1.1, 1.2, 2.0_f32, 2.1, 2.2];
        let mut out = [0.0_f32; 3];
        embed(1, &table, 3, &mut out);
        assert_eq!(out, [1.0, 1.1, 1.2]);
        embed(2, &table, 3, &mut out);
        assert_eq!(out, [2.0, 2.1, 2.2]);
    }
}
