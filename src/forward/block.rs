//! One transformer block: `x ← x + attn(rmsnorm(x))`, then
//! `x ← x + ffn(rmsnorm(x))`.
//!
//! Prefill-only for now: this path takes an `[seq_len, hidden]`
//! activation tensor, computes per-head Q/K/V inline, runs causal
//! self-attention, and rewrites `x` in place. There's no persistent
//! KV cache yet — adding one is Phase A7's job (it'll wrap this
//! path with a `&mut KvCache` parameter and turn the K/V storage
//! into "append + attend over the full window").
//!
//! ## GQA
//!
//! Llama-3-family models (and SmolLM2) use grouped-query attention:
//! `n_kv_heads < n_heads`, with each KV head shared across a
//! contiguous group of Q heads. llama.cpp's convention (which we
//! follow to stay wire-compatible with GGUF weights) is
//! `kv_head = q_head / (n_heads / n_kv_heads)` — Q heads 0..g-1
//! share KV head 0, Q heads g..2g-1 share KV head 1, etc.
//!
//! ## Weight layout
//!
//! All linear-projection weights come straight off GGUF in
//! `[out_dim, in_dim]` row-major. Matmuls use
//! [`crate::forward::ops::matmul`] directly — no pre-transpose.

use crate::forward::kv_cache::LayerKvCache;
use crate::forward::ops::{matmul, rmsnorm, rope, softmax, swiglu};

/// Weights for one transformer block. All slices are `&[f32]`
/// because Milestone A runs on dequantized F32 weights; Milestone C
/// will extend this with per-quant views.
#[derive(Debug, Clone, Copy)]
pub struct BlockWeights<'a> {
    /// RMSNorm gain before attention. Shape `[hidden]`.
    pub attn_norm: &'a [f32],
    /// Q projection, `[n_heads*head_dim, hidden]`.
    pub wq: &'a [f32],
    /// K projection, `[n_kv_heads*head_dim, hidden]`.
    pub wk: &'a [f32],
    /// V projection, `[n_kv_heads*head_dim, hidden]`.
    pub wv: &'a [f32],
    /// Output projection, `[hidden, n_heads*head_dim]`.
    pub wo: &'a [f32],
    /// RMSNorm gain before FFN. Shape `[hidden]`.
    pub ffn_norm: &'a [f32],
    /// FFN gate projection, `[ffn_dim, hidden]`.
    pub w_gate: &'a [f32],
    /// FFN up projection, `[ffn_dim, hidden]`.
    pub w_up: &'a [f32],
    /// FFN down projection, `[hidden, ffn_dim]`.
    pub w_down: &'a [f32],
}

/// Static shape parameters shared across every block.
#[derive(Debug, Clone, Copy)]
pub struct BlockConfig {
    pub hidden: usize,
    pub ffn_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_freq_base: f32,
    pub rope_dim: usize,
    pub norm_eps: f32,
}

impl BlockConfig {
    /// Q heads per KV head for GQA. Always ≥ 1.
    pub fn q_per_kv(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
    /// Width of the Q projection output.
    pub fn q_width(&self) -> usize {
        self.n_heads * self.head_dim
    }
    /// Width of the K/V projection output (may be < `q_width` under GQA).
    pub fn kv_width(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

/// Reusable scratch for one block call. Caller owns the buffers so
/// the whole forward pass can allocate once per token shape.
pub struct BlockScratch {
    pub norm_out: Vec<f32>,  // [seq, hidden]
    pub q: Vec<f32>,         // [seq, n_heads * head_dim]
    pub k: Vec<f32>,         // [seq, n_kv_heads * head_dim]
    pub v: Vec<f32>,         // [seq, n_kv_heads * head_dim]
    pub attn_out: Vec<f32>,  // [seq, n_heads * head_dim]
    pub proj_out: Vec<f32>,  // [seq, hidden]
    pub gate: Vec<f32>,      // [seq, ffn_dim]
    pub up: Vec<f32>,        // [seq, ffn_dim]
    pub ffn_down_in: Vec<f32>, // [seq, ffn_dim]
    pub scores: Vec<f32>,    // [seq] reused per (head, query)
}

impl BlockScratch {
    /// Allocate scratch sized for `batch` new tokens per call and
    /// an attention window up to `max_ctx` positions (= the KV
    /// cache's `max_ctx`). The scores buffer needs `max_ctx`
    /// entries because one cached call's attention ranges over the
    /// full window, not just the batch.
    pub fn new(cfg: &BlockConfig, batch: usize, max_ctx: usize) -> Self {
        Self {
            norm_out: vec![0.0; batch * cfg.hidden],
            q: vec![0.0; batch * cfg.q_width()],
            k: vec![0.0; batch * cfg.kv_width()],
            v: vec![0.0; batch * cfg.kv_width()],
            attn_out: vec![0.0; batch * cfg.q_width()],
            proj_out: vec![0.0; batch * cfg.hidden],
            gate: vec![0.0; batch * cfg.ffn_dim],
            up: vec![0.0; batch * cfg.ffn_dim],
            ffn_down_in: vec![0.0; batch * cfg.ffn_dim],
            scores: vec![0.0; max_ctx],
        }
    }
}

/// Run one transformer block on an `[seq_len, hidden]` activation
/// tensor in place. Appends the current batch's K/V into `cache`
/// and attends over the full accumulated window (rows 0 ..
/// `cache.current_len + seq_len`), so repeated calls implement
/// KV-cached decoding without recomputation.
///
/// The RoPE starting position is taken from `cache.current_len`;
/// callers only need to reset the cache (`LayerKvCache::clear()`)
/// between independent sequences. Scratch's `.scores` slice must
/// hold at least `cache.current_len + seq_len` entries.
pub fn forward_block(
    x: &mut [f32],
    cfg: &BlockConfig,
    weights: &BlockWeights,
    seq_len: usize,
    cache: &mut LayerKvCache,
    scratch: &mut BlockScratch,
) {
    assert_eq!(x.len(), seq_len * cfg.hidden);
    assert_eq!(cache.kv_width, cfg.kv_width());
    let start_pos = cache.current_len;
    let total = start_pos + seq_len;
    assert!(
        total <= cache.max_ctx,
        "KV cache would overflow: start_pos {} + seq_len {} > max_ctx {}",
        start_pos,
        seq_len,
        cache.max_ctx,
    );
    assert!(
        scratch.scores.len() >= total,
        "scratch.scores too small: need {} entries, got {}",
        total,
        scratch.scores.len(),
    );

    // ── Attention sub-block ────────────────────────────────────────
    // 1. RMSNorm(x) → norm_out, row by row.
    for i in 0..seq_len {
        let src = &x[i * cfg.hidden..(i + 1) * cfg.hidden];
        let dst = &mut scratch.norm_out[i * cfg.hidden..(i + 1) * cfg.hidden];
        dst.copy_from_slice(src);
        rmsnorm(dst, weights.attn_norm, cfg.norm_eps);
    }

    // 2. Project to Q, K, V (K and V for the new tokens only).
    //    Matmul asserts exact slice lengths, so we slice scratch
    //    down to seq_len rows — it may be sized for a larger max
    //    batch, but only these rows are live this call.
    matmul(
        &scratch.norm_out[..seq_len * cfg.hidden],
        weights.wq,
        &mut scratch.q[..seq_len * cfg.q_width()],
        seq_len,
        cfg.hidden,
        cfg.q_width(),
    );
    matmul(
        &scratch.norm_out[..seq_len * cfg.hidden],
        weights.wk,
        &mut scratch.k[..seq_len * cfg.kv_width()],
        seq_len,
        cfg.hidden,
        cfg.kv_width(),
    );
    matmul(
        &scratch.norm_out[..seq_len * cfg.hidden],
        weights.wv,
        &mut scratch.v[..seq_len * cfg.kv_width()],
        seq_len,
        cfg.hidden,
        cfg.kv_width(),
    );

    // 3. Apply RoPE to Q and K (V is not rotated). K rotations are
    //    done before appending so the cache stores post-rotation K.
    for i in 0..seq_len {
        let pos = start_pos + i;
        rope(
            &mut scratch.q[i * cfg.q_width()..(i + 1) * cfg.q_width()],
            pos,
            cfg.rope_freq_base,
            cfg.head_dim,
            cfg.rope_dim,
            cfg.n_heads,
        );
        rope(
            &mut scratch.k[i * cfg.kv_width()..(i + 1) * cfg.kv_width()],
            pos,
            cfg.rope_freq_base,
            cfg.head_dim,
            cfg.rope_dim,
            cfg.n_kv_heads,
        );
    }

    // 3b. Append the rotated K and the raw V into the cache.
    cache.append(
        &scratch.k[..seq_len * cfg.kv_width()],
        &scratch.v[..seq_len * cfg.kv_width()],
        seq_len,
    );

    // 4. Causal self-attention: queries from the current batch,
    //    keys/values drawn entirely from the cache (which now
    //    holds positions 0..total).
    let scale = 1.0 / (cfg.head_dim as f32).sqrt();
    let q_per_kv = cfg.q_per_kv();
    for h in 0..cfg.n_heads {
        let kv_h = h / q_per_kv;
        for q_i in 0..seq_len {
            let q_pos = start_pos + q_i;
            let q_vec = &scratch.q
                [q_i * cfg.q_width() + h * cfg.head_dim
                    ..q_i * cfg.q_width() + (h + 1) * cfg.head_dim];
            for k_i in 0..total {
                if k_i > q_pos {
                    scratch.scores[k_i] = f32::NEG_INFINITY;
                    continue;
                }
                let k_vec = &cache.k
                    [k_i * cfg.kv_width() + kv_h * cfg.head_dim
                        ..k_i * cfg.kv_width() + (kv_h + 1) * cfg.head_dim];
                let mut dot = 0.0_f32;
                for d in 0..cfg.head_dim {
                    dot += q_vec[d] * k_vec[d];
                }
                scratch.scores[k_i] = dot * scale;
            }
            softmax(&mut scratch.scores[..total]);

            let out_slot = &mut scratch.attn_out
                [q_i * cfg.q_width() + h * cfg.head_dim
                    ..q_i * cfg.q_width() + (h + 1) * cfg.head_dim];
            for v in out_slot.iter_mut() {
                *v = 0.0;
            }
            for k_i in 0..total {
                let w = scratch.scores[k_i];
                if w == 0.0 {
                    continue;
                }
                let v_vec = &cache.v
                    [k_i * cfg.kv_width() + kv_h * cfg.head_dim
                        ..k_i * cfg.kv_width() + (kv_h + 1) * cfg.head_dim];
                for d in 0..cfg.head_dim {
                    out_slot[d] += w * v_vec[d];
                }
            }
        }
    }

    // 5. Output projection into proj_out.
    matmul(
        &scratch.attn_out[..seq_len * cfg.q_width()],
        weights.wo,
        &mut scratch.proj_out[..seq_len * cfg.hidden],
        seq_len,
        cfg.q_width(),
        cfg.hidden,
    );

    // 6. Residual add: x ← x + proj_out.
    for (xi, yi) in x.iter_mut().zip(scratch.proj_out[..seq_len * cfg.hidden].iter()) {
        *xi += *yi;
    }

    // ── FFN sub-block ──────────────────────────────────────────────
    // 7. RMSNorm(x) → norm_out.
    for i in 0..seq_len {
        let src = &x[i * cfg.hidden..(i + 1) * cfg.hidden];
        let dst = &mut scratch.norm_out[i * cfg.hidden..(i + 1) * cfg.hidden];
        dst.copy_from_slice(src);
        rmsnorm(dst, weights.ffn_norm, cfg.norm_eps);
    }

    // 8. Gate / up projections.
    matmul(
        &scratch.norm_out[..seq_len * cfg.hidden],
        weights.w_gate,
        &mut scratch.gate[..seq_len * cfg.ffn_dim],
        seq_len,
        cfg.hidden,
        cfg.ffn_dim,
    );
    matmul(
        &scratch.norm_out[..seq_len * cfg.hidden],
        weights.w_up,
        &mut scratch.up[..seq_len * cfg.ffn_dim],
        seq_len,
        cfg.hidden,
        cfg.ffn_dim,
    );

    // 9. SwiGLU → ffn_down_in.
    swiglu(
        &scratch.gate[..seq_len * cfg.ffn_dim],
        &scratch.up[..seq_len * cfg.ffn_dim],
        &mut scratch.ffn_down_in[..seq_len * cfg.ffn_dim],
    );

    // 10. Down projection.
    matmul(
        &scratch.ffn_down_in[..seq_len * cfg.ffn_dim],
        weights.w_down,
        &mut scratch.proj_out[..seq_len * cfg.hidden],
        seq_len,
        cfg.ffn_dim,
        cfg.hidden,
    );

    // 11. Residual add.
    for (xi, yi) in x.iter_mut().zip(scratch.proj_out[..seq_len * cfg.hidden].iter()) {
        *xi += *yi;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple config: 4-dim hidden, 1 head of 4 dims, tiny FFN.
    fn tiny_cfg() -> BlockConfig {
        BlockConfig {
            hidden: 4,
            ffn_dim: 4,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 4,
            rope_freq_base: 10_000.0,
            rope_dim: 4,
            norm_eps: 1e-5,
        }
    }

    /// GQA config: 4 Q heads sharing 2 KV heads.
    fn gqa_cfg() -> BlockConfig {
        BlockConfig {
            hidden: 8,
            ffn_dim: 8,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 2,
            rope_freq_base: 10_000.0,
            rope_dim: 2,
            norm_eps: 1e-5,
        }
    }

    /// Owned-storage counterpart to `BlockWeights`. Test code
    /// allocates one of these, then calls `.view()` to get a
    /// `BlockWeights<'_>` pointing at its slices.
    struct OwnedBlockWeights {
        attn_norm: Vec<f32>,
        wq: Vec<f32>,
        wk: Vec<f32>,
        wv: Vec<f32>,
        wo: Vec<f32>,
        ffn_norm: Vec<f32>,
        w_gate: Vec<f32>,
        w_up: Vec<f32>,
        w_down: Vec<f32>,
    }

    impl OwnedBlockWeights {
        fn view(&self) -> BlockWeights<'_> {
            BlockWeights {
                attn_norm: &self.attn_norm,
                wq: &self.wq,
                wk: &self.wk,
                wv: &self.wv,
                wo: &self.wo,
                ffn_norm: &self.ffn_norm,
                w_gate: &self.w_gate,
                w_up: &self.w_up,
                w_down: &self.w_down,
            }
        }

        /// Zero projections + unit norms: every sub-block contributes
        /// zero, residuals pass `x` through unchanged.
        fn zeros(cfg: &BlockConfig) -> Self {
            Self {
                attn_norm: vec![1.0_f32; cfg.hidden],
                wq: vec![0.0_f32; cfg.q_width() * cfg.hidden],
                wk: vec![0.0_f32; cfg.kv_width() * cfg.hidden],
                wv: vec![0.0_f32; cfg.kv_width() * cfg.hidden],
                wo: vec![0.0_f32; cfg.hidden * cfg.q_width()],
                ffn_norm: vec![1.0_f32; cfg.hidden],
                w_gate: vec![0.0_f32; cfg.ffn_dim * cfg.hidden],
                w_up: vec![0.0_f32; cfg.ffn_dim * cfg.hidden],
                w_down: vec![0.0_f32; cfg.hidden * cfg.ffn_dim],
            }
        }

        /// Deterministic pseudo-random `[-0.5, 0.5]` fills. LCG
        /// seeded from `seed` — no rand-crate dep, bit-identical
        /// across platforms.
        fn random(cfg: &BlockConfig, seed: u32) -> Self {
            let mut state = seed;
            let mut next = |n: usize| -> Vec<f32> {
                (0..n)
                    .map(|_| {
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        ((state >> 16) & 0xFFFF) as f32 / 65_536.0 - 0.5
                    })
                    .collect()
            };
            Self {
                attn_norm: next(cfg.hidden),
                wq: next(cfg.q_width() * cfg.hidden),
                wk: next(cfg.kv_width() * cfg.hidden),
                wv: next(cfg.kv_width() * cfg.hidden),
                wo: next(cfg.hidden * cfg.q_width()),
                ffn_norm: next(cfg.hidden),
                w_gate: next(cfg.ffn_dim * cfg.hidden),
                w_up: next(cfg.ffn_dim * cfg.hidden),
                w_down: next(cfg.hidden * cfg.ffn_dim),
            }
        }
    }

    fn fresh_cache(cfg: &BlockConfig, max_ctx: usize) -> LayerKvCache {
        LayerKvCache::new(max_ctx, cfg.kv_width())
    }

    #[test]
    fn zero_projections_pass_input_through_residuals() {
        // Attention + FFN both output zero → residuals keep x.
        let cfg = tiny_cfg();
        let w = OwnedBlockWeights::zeros(&cfg);
        let mut x = vec![0.5_f32, -1.0, 2.0, 0.25];
        let snapshot = x.clone();
        let mut scratch = BlockScratch::new(&cfg, 1, 8);
        let mut cache = fresh_cache(&cfg, 8);
        forward_block(&mut x, &cfg, &w.view(), 1, &mut cache, &mut scratch);
        assert_eq!(x, snapshot, "zero projections should leave x unchanged");
    }

    #[test]
    fn zero_input_stays_zero_with_any_weights() {
        // Zero input → rmsnorm produces 0 (finite eps keeps it
        // defined), matmuls produce 0, softmax(0) is uniform, but
        // V is also 0 so attention output is 0, FFN output 0.
        // Residual adds 0 → x stays all zeros.
        let cfg = tiny_cfg();
        let w = OwnedBlockWeights::random(&cfg, 1);
        let mut x = vec![0.0_f32; cfg.hidden];
        let mut scratch = BlockScratch::new(&cfg, 1, 8);
        let mut cache = fresh_cache(&cfg, 8);
        forward_block(&mut x, &cfg, &w.view(), 1, &mut cache, &mut scratch);
        for v in x {
            assert!(v.abs() < 1e-6, "zero input drifted to {v}");
        }
    }

    #[test]
    fn block_is_deterministic() {
        // Same inputs + weights → bit-identical outputs across calls.
        let cfg = tiny_cfg();
        let w = OwnedBlockWeights::random(&cfg, 42);
        let x0: Vec<f32> = (0..cfg.hidden * 3).map(|i| (i as f32).sin()).collect();

        let mut x1 = x0.clone();
        let mut s1 = BlockScratch::new(&cfg, 3, 8);
        let mut c1 = fresh_cache(&cfg, 8);
        forward_block(&mut x1, &cfg, &w.view(), 3, &mut c1, &mut s1);

        let mut x2 = x0.clone();
        let mut s2 = BlockScratch::new(&cfg, 3, 8);
        let mut c2 = fresh_cache(&cfg, 8);
        forward_block(&mut x2, &cfg, &w.view(), 3, &mut c2, &mut s2);

        assert_eq!(x1, x2, "forward_block is nondeterministic");
    }

    #[test]
    fn causal_mask_blocks_future_from_influencing_past() {
        // Run forward on x = [t0, t1, t2], capture the output row at
        // position 0. Now swap t1 and t2 (or any later-position row)
        // and rerun: the row-0 output must be identical, because
        // causal attention can't see positions > 0 from position 0.
        let cfg = tiny_cfg();
        let w = OwnedBlockWeights::random(&cfg, 7);

        let t0: Vec<f32> = (0..cfg.hidden).map(|i| (i as f32 * 0.17).sin()).collect();
        let t1_a: Vec<f32> = (0..cfg.hidden).map(|i| (i as f32 * 0.31).cos()).collect();
        let t2_a: Vec<f32> = (0..cfg.hidden).map(|i| (i as f32 * 0.47).sin()).collect();
        let t1_b: Vec<f32> = t2_a.clone();
        let t2_b: Vec<f32> = t1_a.clone();

        let mut x_a = [t0.clone(), t1_a, t2_a].concat();
        let mut x_b = [t0.clone(), t1_b, t2_b].concat();

        let mut sa = BlockScratch::new(&cfg, 3, 8);
        let mut sb = BlockScratch::new(&cfg, 3, 8);
        let mut ca = fresh_cache(&cfg, 8);
        let mut cb = fresh_cache(&cfg, 8);
        forward_block(&mut x_a, &cfg, &w.view(), 3, &mut ca, &mut sa);
        forward_block(&mut x_b, &cfg, &w.view(), 3, &mut cb, &mut sb);

        for d in 0..cfg.hidden {
            assert!(
                (x_a[d] - x_b[d]).abs() < 1e-5,
                "causal mask broken: pos-0 differs after swapping later tokens ({} vs {})",
                x_a[d], x_b[d],
            );
        }
    }

    #[test]
    fn gqa_block_runs_without_panic_and_preserves_shape() {
        // Shape conservation and no NaNs under GQA.
        let cfg = gqa_cfg();
        let w = OwnedBlockWeights::random(&cfg, 99);
        let mut x: Vec<f32> =
            (0..4 * cfg.hidden).map(|i| (i as f32 * 0.1).sin()).collect();
        let start_len = x.len();
        let mut s = BlockScratch::new(&cfg, 4, 8);
        let mut cache = fresh_cache(&cfg, 8);
        forward_block(&mut x, &cfg, &w.view(), 4, &mut cache, &mut s);
        assert_eq!(x.len(), start_len);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn kv_cached_split_matches_single_shot_forward() {
        // Prefill [t0, t1] then decode [t2] against the grown cache
        // should give the same pos-2 activation as a fresh
        // forward over [t0, t1, t2]. This is the A7 coherence gate.
        let cfg = tiny_cfg();
        let w = OwnedBlockWeights::random(&cfg, 31);

        let t0: Vec<f32> = (0..cfg.hidden).map(|i| (i as f32 * 0.11).sin()).collect();
        let t1: Vec<f32> = (0..cfg.hidden).map(|i| (i as f32 * 0.23).cos()).collect();
        let t2: Vec<f32> = (0..cfg.hidden).map(|i| (i as f32 * 0.37).sin()).collect();

        // (a) Single-shot prefill of all three tokens.
        let mut x_single = [t0.clone(), t1.clone(), t2.clone()].concat();
        let mut s_single = BlockScratch::new(&cfg, 3, 8);
        let mut c_single = fresh_cache(&cfg, 8);
        forward_block(&mut x_single, &cfg, &w.view(), 3, &mut c_single, &mut s_single);

        // (b) Two-step: prefill [t0, t1], then decode [t2].
        let mut x_step = [t0.clone(), t1.clone()].concat();
        let mut s_step = BlockScratch::new(&cfg, 2, 8);
        let mut c_step = fresh_cache(&cfg, 8);
        forward_block(&mut x_step, &cfg, &w.view(), 2, &mut c_step, &mut s_step);
        let mut x_step2 = t2.clone();
        // Scratch for the second call with batch=1.
        let mut s_step2 = BlockScratch::new(&cfg, 1, 8);
        forward_block(&mut x_step2, &cfg, &w.view(), 1, &mut c_step, &mut s_step2);

        // The pos-2 row of the single-shot run must equal the
        // decoded output from step 2.
        let pos2_single = &x_single[2 * cfg.hidden..3 * cfg.hidden];
        for d in 0..cfg.hidden {
            assert!(
                (pos2_single[d] - x_step2[d]).abs() < 1e-5,
                "KV cache incoherent at dim {d}: {} vs {}",
                pos2_single[d],
                x_step2[d],
            );
        }
    }

    // Note: `start_pos` is deliberately *not* observable inside a
    // single prefill — RoPE encodes only *relative* positions, so
    // shifting every (Q,K) pair by the same offset leaves every
    // dot product Q·K invariant. The offset becomes observable when
    // A7 introduces a KV cache: prior tokens carry the rotation
    // from *their* positions, and new tokens see the relative-
    // position relationship to them. The A5 gate is just "does it
    // thread through without panicking or polluting the output" —
    // we don't have a meaningful `start_pos`-varies-output case
    // until A7.
}
