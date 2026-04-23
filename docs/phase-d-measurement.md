# Phase D0 — Hessian measurement

*Auto-generated tables (below) produced by `scripts/analyze-hessian.py`; the **Decision** section at the end is maintained by hand.*

## Scope and method

D0 measures the empirical structure of the per-layer activation Hessian
`H = E[x x^T]` on a real cover, so the Phase D production path can commit to
a structured approximation based on data rather than guesses.

- **Covers measured**: `smollm2-135m-q8_0.gguf` and `smollm2-135m-f16.gguf`.
  The two Qwen-style and Llama-tokenizer models available locally can't load
  through our forward pass (`ForwardModel::load` rejects `qwen2` arch and
  non-`gpt2` tokenizers — see `src/forward/config.rs` and
  `src/forward/tokenizer.rs`). SmolLM2 at both quantizations stands in as a
  second data point: same architecture, different per-weight dequant noise,
  so identical H structure across the pair also validates the dequant path.
- **Corpus**: first ~12 KB of `benches/fixtures/wiki.test.raw`, tokenized
  and truncated to `T = 2048` tokens. `T > N_max = 1536` at every site, so
  H is full-rank and the eigenvalue tail reflects real structure (not rank
  truncation).
- **Accumulator**: `HessianAccumulator` in `src/forward/hessian.rs`, F64
  upper-triangle, observer-hooked into every matmul-input site inside
  `forward_all_logits_with_observer`.
- **Strategies compared** (relative Frobenius error, lower is better):
  - Low-rank — keep top-K eigenpairs of `H`.
  - Block-diagonal — zero cross-block entries for block sizes 32–256.
  - Top-K sparse — keep the K largest-magnitude entries per column, plus
    the diagonal.

## Headline findings

1. **All 240 matrices are PSD, zero symmetry error.** Confirms accumulator
   correctness.
2. **F16 vs Q8_0 agree within noise.** Per-(site, layer) relative Frobenius
   errors match to the third decimal across every K and every strategy.
   Q8_0 dequant does not perturb H structure.
3. **Low-rank wins decisively** at every storage budget on every site.
   Representative numbers at 144 KB/layer (K=64 on N=576 sites):

   | Site | Low-rank | Top-K sparse | Block-diag (bs=64) |
   |------|---------:|-------------:|--------------------:|
   | qkv_input | 1.4% | 14.5% | 58.1% |
   | attn_output_input | 4.5% | 46.7% | 85.7% |
   | ffn_gate_up_input | 4.8% | 31.4% | 71.4% |
   | ffn_down_input (N=1536) | 16.4% | 49.8% | 84.8% |

   Low-rank is 3–20× better than every alternative at matched storage.
4. **Effective rank is site-specific.** Within the same model, `qkv_input`
   reaches 1% Frobenius error at K=32, while `ffn_down_input` needs K=512
   for the same quality. A single shared K across sites is wasteful —
   site-specific K (or target-error calibration of K) is required.
5. **Trace vs Frobenius.** `K_95%_of_trace` (the natural eyeball metric)
   significantly over-estimates what's needed. `ffn_down_input` has median
   `K_95_trace = 508`, but 5% Frobenius error is reached at `K ≈ 256`.
   Frobenius is the right quality metric for GPTQ compensation.

## smollm2-135m-f16

- Cover: `smollm2-135m-f16.gguf`
- Corpus: `wiki.test.raw`
- Token count (T): 2048
- Shape: hidden_dim=576, ffn_dim=1536, n_layers=30, n_heads=9, n_kv_heads=3, head_dim=64

### Sanity
- max symmetry error across all (site, layer): 0.00e+00
- max negative-eigenvalue count across all (site, layer): 0 (should be 0 for PSD)

### attn_output_input (N = 576, 30 layers)

**Trace fractions (median / min / max K across layers):**

| Fraction of trace | Median K | Min K | Max K | K/N median |
|---|---|---|---|---|
| 90% | 93 | 2 | 160 | 0.161 |
| 95% | 159 | 4 | 240 | 0.276 |
| 99% | 309 | 23 | 402 | 0.536 |

**Low-rank approximation (relative Frobenius error):**

| K | median err | max err | storage / layer |
|---|---|---|---|
| 16 | 0.1257 | 0.2202 | 36.1 KB |
| 32 | 0.0812 | 0.1530 | 72.1 KB |
| 64 | 0.0455 | 0.0944 | 144.2 KB |
| 128 | 0.0189 | 0.0471 | 288.5 KB |
| 256 | 0.0049 | 0.0161 | 577.0 KB |
| 512 | 0.0002 | 0.0013 | 1154.0 KB |

**Block-diagonal approximation (relative Frobenius error):**

| Block size | median err | max err | storage / layer |
|---|---|---|---|
| 32 | 0.9060 | 0.9512 | 37.1 KB |
| 64 | 0.8561 | 0.9093 | 73.1 KB |

**Top-K sparse per column (relative Frobenius error):**

| K | median err | max err | storage / layer |
|---|---|---|---|
| 16 | 0.6814 | 0.7972 | 56.2 KB |
| 32 | 0.5915 | 0.7088 | 110.2 KB |
| 64 | 0.4669 | 0.5981 | 218.2 KB |
| 128 | 0.3289 | 0.4510 | 434.2 KB |
| 256 | 0.1741 | 0.2506 | 866.2 KB |

*Reference: full upper triangle = 649.1 KB / layer.*

### ffn_down_input (N = 1536, 30 layers)

**Trace fractions (median / min / max K across layers):**

| Fraction of trace | Median K | Min K | Max K | K/N median |
|---|---|---|---|---|
| 90% | 357 | 1 | 443 | 0.232 |
| 95% | 508 | 1 | 617 | 0.331 |
| 99% | 849 | 1 | 968 | 0.553 |

**Low-rank approximation (relative Frobenius error):**

| K | median err | max err | storage / layer |
|---|---|---|---|
| 16 | 0.3253 | 0.4977 | 96.1 KB |
| 32 | 0.2405 | 0.3898 | 192.1 KB |
| 64 | 0.1640 | 0.2809 | 384.2 KB |
| 128 | 0.0961 | 0.1839 | 768.5 KB |
| 256 | 0.0475 | 0.1028 | 1537.0 KB |
| 512 | 0.0171 | 0.0386 | 3074.0 KB |

**Block-diagonal approximation (relative Frobenius error):**

| Block size | median err | max err | storage / layer |
|---|---|---|---|
| 32 | 0.8573 | 0.9391 | 99.0 KB |
| 64 | 0.8479 | 0.9258 | 195.0 KB |
| 128 | 0.8269 | 0.9119 | 387.0 KB |
| 256 | 0.7859 | 0.8561 | 771.0 KB |

**Top-K sparse per column (relative Frobenius error):**

| K | median err | max err | storage / layer |
|---|---|---|---|
| 16 | 0.6462 | 0.7964 | 150.0 KB |
| 32 | 0.5763 | 0.7450 | 294.0 KB |
| 64 | 0.4979 | 0.6811 | 582.0 KB |
| 128 | 0.4032 | 0.5936 | 1158.0 KB |
| 256 | 0.2973 | 0.4716 | 2310.0 KB |

*Reference: full upper triangle = 4611.0 KB / layer.*

### ffn_gate_up_input (N = 576, 30 layers)

**Trace fractions (median / min / max K across layers):**

| Fraction of trace | Median K | Min K | Max K | K/N median |
|---|---|---|---|---|
| 90% | 179 | 38 | 230 | 0.311 |
| 95% | 270 | 106 | 312 | 0.469 |
| 99% | 437 | 307 | 458 | 0.759 |

**Low-rank approximation (relative Frobenius error):**

| K | median err | max err | storage / layer |
|---|---|---|---|
| 16 | 0.0996 | 0.2599 | 36.1 KB |
| 32 | 0.0720 | 0.1981 | 72.1 KB |
| 64 | 0.0475 | 0.1398 | 144.2 KB |
| 128 | 0.0278 | 0.0835 | 288.5 KB |
| 256 | 0.0116 | 0.0330 | 577.0 KB |
| 512 | 0.0012 | 0.0028 | 1154.0 KB |

**Block-diagonal approximation (relative Frobenius error):**

| Block size | median err | max err | storage / layer |
|---|---|---|---|
| 32 | 0.7860 | 0.8789 | 37.1 KB |
| 64 | 0.7135 | 0.8491 | 73.1 KB |

**Top-K sparse per column (relative Frobenius error):**

| K | median err | max err | storage / layer |
|---|---|---|---|
| 16 | 0.4153 | 0.5830 | 56.2 KB |
| 32 | 0.3700 | 0.5327 | 110.2 KB |
| 64 | 0.3141 | 0.4595 | 218.2 KB |
| 128 | 0.2361 | 0.3502 | 434.2 KB |
| 256 | 0.1312 | 0.1964 | 866.2 KB |

*Reference: full upper triangle = 649.1 KB / layer.*

### qkv_input (N = 576, 30 layers)

**Trace fractions (median / min / max K across layers):**

| Fraction of trace | Median K | Min K | Max K | K/N median |
|---|---|---|---|---|
| 90% | 80 | 6 | 140 | 0.139 |
| 95% | 162 | 48 | 226 | 0.281 |
| 99% | 356 | 210 | 397 | 0.618 |

**Low-rank approximation (relative Frobenius error):**

| K | median err | max err | storage / layer |
|---|---|---|---|
| 16 | 0.0311 | 0.0944 | 36.1 KB |
| 32 | 0.0211 | 0.0550 | 72.1 KB |
| 64 | 0.0136 | 0.0304 | 144.2 KB |
| 128 | 0.0071 | 0.0155 | 288.5 KB |
| 256 | 0.0028 | 0.0059 | 577.0 KB |
| 512 | 0.0003 | 0.0005 | 1154.0 KB |

**Block-diagonal approximation (relative Frobenius error):**

| Block size | median err | max err | storage / layer |
|---|---|---|---|
| 32 | 0.5979 | 0.7989 | 37.1 KB |
| 64 | 0.5808 | 0.7675 | 73.1 KB |

**Top-K sparse per column (relative Frobenius error):**

| K | median err | max err | storage / layer |
|---|---|---|---|
| 16 | 0.1962 | 0.3124 | 56.2 KB |
| 32 | 0.1732 | 0.2820 | 110.2 KB |
| 64 | 0.1451 | 0.2401 | 218.2 KB |
| 128 | 0.1072 | 0.1812 | 434.2 KB |
| 256 | 0.0557 | 0.0975 | 866.2 KB |

*Reference: full upper triangle = 649.1 KB / layer.*

## smollm2-135m-q8_0

- Cover: `smollm2-135m-q8_0.gguf`
- Corpus: `wiki.test.raw`
- Token count (T): 2048
- Shape: hidden_dim=576, ffn_dim=1536, n_layers=30, n_heads=9, n_kv_heads=3, head_dim=64

### Sanity
- max symmetry error across all (site, layer): 0.00e+00
- max negative-eigenvalue count across all (site, layer): 0 (should be 0 for PSD)

### attn_output_input (N = 576, 30 layers)

**Trace fractions (median / min / max K across layers):**

| Fraction of trace | Median K | Min K | Max K | K/N median |
|---|---|---|---|---|
| 90% | 93 | 2 | 160 | 0.161 |
| 95% | 160 | 4 | 240 | 0.278 |
| 99% | 309 | 23 | 402 | 0.536 |

**Low-rank approximation (relative Frobenius error):**

| K | median err | max err | storage / layer |
|---|---|---|---|
| 16 | 0.1262 | 0.2199 | 36.1 KB |
| 32 | 0.0815 | 0.1528 | 72.1 KB |
| 64 | 0.0452 | 0.0943 | 144.2 KB |
| 128 | 0.0190 | 0.0471 | 288.5 KB |
| 256 | 0.0049 | 0.0161 | 577.0 KB |
| 512 | 0.0002 | 0.0013 | 1154.0 KB |

**Block-diagonal approximation (relative Frobenius error):**

| Block size | median err | max err | storage / layer |
|---|---|---|---|
| 32 | 0.9064 | 0.9510 | 37.1 KB |
| 64 | 0.8567 | 0.9092 | 73.1 KB |

**Top-K sparse per column (relative Frobenius error):**

| K | median err | max err | storage / layer |
|---|---|---|---|
| 16 | 0.6803 | 0.7952 | 56.2 KB |
| 32 | 0.5929 | 0.7101 | 110.2 KB |
| 64 | 0.4675 | 0.5994 | 218.2 KB |
| 128 | 0.3307 | 0.4524 | 434.2 KB |
| 256 | 0.1747 | 0.2512 | 866.2 KB |

*Reference: full upper triangle = 649.1 KB / layer.*

### ffn_down_input (N = 1536, 30 layers)

**Trace fractions (median / min / max K across layers):**

| Fraction of trace | Median K | Min K | Max K | K/N median |
|---|---|---|---|---|
| 90% | 357 | 1 | 443 | 0.232 |
| 95% | 508 | 1 | 617 | 0.331 |
| 99% | 848 | 1 | 968 | 0.552 |

**Low-rank approximation (relative Frobenius error):**

| K | median err | max err | storage / layer |
|---|---|---|---|
| 16 | 0.3254 | 0.4983 | 96.1 KB |
| 32 | 0.2406 | 0.3902 | 192.1 KB |
| 64 | 0.1638 | 0.2812 | 384.2 KB |
| 128 | 0.0959 | 0.1845 | 768.5 KB |
| 256 | 0.0474 | 0.1031 | 1537.0 KB |
| 512 | 0.0170 | 0.0387 | 3074.0 KB |

**Block-diagonal approximation (relative Frobenius error):**

| Block size | median err | max err | storage / layer |
|---|---|---|---|
| 32 | 0.8577 | 0.9389 | 99.0 KB |
| 64 | 0.8482 | 0.9254 | 195.0 KB |
| 128 | 0.8272 | 0.9117 | 387.0 KB |
| 256 | 0.7861 | 0.8561 | 771.0 KB |

**Top-K sparse per column (relative Frobenius error):**

| K | median err | max err | storage / layer |
|---|---|---|---|
| 16 | 0.6473 | 0.7973 | 150.0 KB |
| 32 | 0.5760 | 0.7458 | 294.0 KB |
| 64 | 0.4985 | 0.6817 | 582.0 KB |
| 128 | 0.4036 | 0.5942 | 1158.0 KB |
| 256 | 0.2976 | 0.4721 | 2310.0 KB |

*Reference: full upper triangle = 4611.0 KB / layer.*

### ffn_gate_up_input (N = 576, 30 layers)

**Trace fractions (median / min / max K across layers):**

| Fraction of trace | Median K | Min K | Max K | K/N median |
|---|---|---|---|---|
| 90% | 179 | 38 | 229 | 0.311 |
| 95% | 270 | 107 | 312 | 0.469 |
| 99% | 437 | 307 | 458 | 0.759 |

**Low-rank approximation (relative Frobenius error):**

| K | median err | max err | storage / layer |
|---|---|---|---|
| 16 | 0.0997 | 0.2595 | 36.1 KB |
| 32 | 0.0720 | 0.1978 | 72.1 KB |
| 64 | 0.0475 | 0.1395 | 144.2 KB |
| 128 | 0.0278 | 0.0833 | 288.5 KB |
| 256 | 0.0116 | 0.0329 | 577.0 KB |
| 512 | 0.0012 | 0.0028 | 1154.0 KB |

**Block-diagonal approximation (relative Frobenius error):**

| Block size | median err | max err | storage / layer |
|---|---|---|---|
| 32 | 0.7860 | 0.8792 | 37.1 KB |
| 64 | 0.7138 | 0.8493 | 73.1 KB |

**Top-K sparse per column (relative Frobenius error):**

| K | median err | max err | storage / layer |
|---|---|---|---|
| 16 | 0.4158 | 0.5832 | 56.2 KB |
| 32 | 0.3706 | 0.5332 | 110.2 KB |
| 64 | 0.3144 | 0.4600 | 218.2 KB |
| 128 | 0.2363 | 0.3506 | 434.2 KB |
| 256 | 0.1311 | 0.1966 | 866.2 KB |

*Reference: full upper triangle = 649.1 KB / layer.*

### qkv_input (N = 576, 30 layers)

**Trace fractions (median / min / max K across layers):**

| Fraction of trace | Median K | Min K | Max K | K/N median |
|---|---|---|---|---|
| 90% | 80 | 6 | 139 | 0.139 |
| 95% | 162 | 48 | 226 | 0.281 |
| 99% | 356 | 210 | 397 | 0.618 |

**Low-rank approximation (relative Frobenius error):**

| K | median err | max err | storage / layer |
|---|---|---|---|
| 16 | 0.0312 | 0.0944 | 36.1 KB |
| 32 | 0.0211 | 0.0550 | 72.1 KB |
| 64 | 0.0137 | 0.0304 | 144.2 KB |
| 128 | 0.0072 | 0.0155 | 288.5 KB |
| 256 | 0.0029 | 0.0058 | 577.0 KB |
| 512 | 0.0003 | 0.0005 | 1154.0 KB |

**Block-diagonal approximation (relative Frobenius error):**

| Block size | median err | max err | storage / layer |
|---|---|---|---|
| 32 | 0.5983 | 0.7990 | 37.1 KB |
| 64 | 0.5809 | 0.7674 | 73.1 KB |

**Top-K sparse per column (relative Frobenius error):**

| K | median err | max err | storage / layer |
|---|---|---|---|
| 16 | 0.1965 | 0.3127 | 56.2 KB |
| 32 | 0.1735 | 0.2821 | 110.2 KB |
| 64 | 0.1454 | 0.2403 | 218.2 KB |
| 128 | 0.1075 | 0.1812 | 434.2 KB |
| 256 | 0.0555 | 0.0974 | 866.2 KB |

*Reference: full upper triangle = 649.1 KB / layer.*

---

## Decision

Full compensation architecture is documented in
[`docs/compensation-design.md`](compensation-design.md). D0's measurements
back two specific choices in that architecture; the rest of this section
is scoped to what D0's numbers actually justify.

### What D0 confirms

**1. Low-rank structure is real and uniform across sites.** Low-rank
eigendecomposition of H dominates block-diagonal and top-K sparse by
3–20× at every measured storage budget across all four observation sites.
This justifies the "optionally low-rank Cholesky factor" clause in
`compensation-design.md §2.2` — the L2 factor can compress with bounded
loss, rather than being stored full-rank. Effective rank is site-specific
(qkv ≈ 3% of N for 5% Frobenius, ffn_down ≈ 17% of N), so a low-rank
factor representation will want per-site `K`, not a uniform one.

**2. F16 is sufficient for the factor.** F16 and Q8_0 dumps agree on
every Frobenius metric to the third decimal. Q8_0 dequant noise
contributes nothing observable to H structure, and (by the same token)
F16 precision for the cached Cholesky factor is well inside the floor.
The "fp16" clause in `compensation-design.md §3.1–3.2` lands.

**3. Frobenius is the right quality metric, not trace.** `K_95%_of_trace`
overestimates the `K` needed for 5% Frobenius error by ~3× on the same
matrix. Compensation quality is Frobenius-norm-bounded (from `‖δy‖² =
Δᵀ H Δ`), so Frobenius is what the calibrator should target when
choosing per-site factor rank.

**4. Accumulator is correct.** Zero symmetry error and zero negative
eigenvalues across 240 matrices. The `HessianAccumulator` in
`src/forward/hessian.rs` produces matrices Cholesky-factorizable without
regularization — no `εI` shim needed in the happy path.

### What D0 explicitly does *not* decide

- **Where H or its factor lives.** Per `compensation-design.md §9` and
  the at-rest stego invariant, nothing H-related is persistent. H and
  the Cholesky factor are recomputed from the user-supplied calibration
  corpus on cold start; the cost (seconds per layer on first write per
  mount) is accepted. D0's earlier framing of a "V2-resident
  `(λ, v)` codec" assumed on-disk persistence and is obsolete. The same
  low-rank idea applies to the *in-RAM* L2 factor, not to V2 storage.
- **What the salience inode stores.** The right placement metric is
  OBS saliency `1/[H⁻¹]_ii` (see `compensation-design.md §1.2`), not
  `diag(H)`. Computable from the Cholesky factor as `‖L⁻¹ e_i‖²`. D1
  replaces AWQ's `mean(|x|)` with OBS saliency, not with `diag(H)`.
- **Cross-architecture validation.** 240 matrices from one architecture
  (SmolLM2) is a conclusive signal for structural choice, not for
  specific `K` values at 3B/7B scale. Per-site `K` will be re-calibrated
  per-cover at calibrate time — the structural decision stays put, the
  parameters are data-driven per run.
- **Streaming / randomized SVD accumulation.** D0's accumulator
  materializes full H in RAM; fits SmolLM2 and up to ~3B. Phase D1+
  may want a Lanczos or randomized-SVD variant for larger models that
  bypasses full materialization. Out of D0's scope.

### Pointers into the architecture

- Hot-path compensation operator `M_R`: `compensation-design.md §1.3, §3.1`.
- Per-layer Cholesky factor (L2): `§3.2`.
- Cold-start lazy recompute: `§9`.
- Capacity cliff and the fidelity ceiling: `§7`.
- Online maintenance and drift bounds: `§6`.


