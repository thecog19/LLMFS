# Compensation math and caching — design

**Scope:** the math framework and caching architecture for error-compensated stego writes on int-quantized covers. Filesystem-layer concerns (inode design, metadata placement, write ordering) are documented separately in `DESIGN-NEW.MD §15` and the V2 module docs. External-reader concurrency (how an mmap'd llama.cpp coexists with an active LLMDB mount) is covered in [`concurrency-design.md`](concurrency-design.md) — this doc bounds the *magnitude* of any given in-flight perturbation; the concurrency doc shows that bound is what makes "write in place, let readers see what they see" a viable story.

**Status:** architecture plus early implementation for Phase E onward. Phase D0 (`docs/phase-d-measurement.md`) measured the per-layer Hessian structure that this design relies on; the low-rank Cholesky option in §2.2 and the F16 factor precision in §3.2 are both backed by that measurement. Phase D1 now has the production `llmdb calibrate --full` path, OBS salience extraction, an in-memory Hessian factor cache, pivoted Cholesky support, and the first Phase E bridge from V2 write coordinates to `(ActivationSite, layer, input_channel)` compensation coordinates. That bridge now groups a written V2 `Pointer` into row-local compensation regions, captures signed per-weight perturbations from chunk writes, folds those perturbations into operator-ready region delta vectors, resolves those regions against cached Hessian factors to produce compensation-channel deltas, has a quantized cover actuator that writes each compensation target to the nearest representable stealable-bit value, exposes a chunk-write helper that performs payload write plus immediate cached compensation, and includes a dirty-target guard for the clean-weight-only path. V2 mounts can now carry the GGUF tensor layout plus Hessian factor cache as runtime-only compensation state, and newly-written file-content chunks route through that runtime when it is attached. The remaining gaps are CLI/mount lifecycle hookup for that runtime, metadata policy, and replacing the guard with the dirty-weight mask-and-correct path.

**Relationship to the stego invariant.** No H, no Cholesky factor, no compensation operator is persistent. Everything is recomputed from the cover plus a calibration corpus that ships with LLMDB itself (bundled in the binary — see `src/calibrate.rs:DEFAULT_CALIBRATION_CORPUS`). The corpus is a property of the tool, not of the user or the cover. This keeps the at-rest cover indistinguishable from a bare GGUF (see `project_stego_invariant_at_rest` memory / `DESIGN-NEW.MD §15`).

---

## 1. Compensation framework

### 1.1 The objective

For each write operation, a forced perturbation `Δ_S` is applied to a steal set `S` (the weights receiving payload). We compensate by adjusting weights in a compensation set `C`, chosen to minimize the resulting loss increase.

At a trained minimum, `∇L ≈ 0`, so the quadratic approximation gives:

```
ΔL ≈ ½ Δᵀ H Δ = ½ Δ_Sᵀ H_SS Δ_S + Δ_Sᵀ H_SC Δ_C + ½ Δ_Cᵀ H_CC Δ_C
```

Minimizing over `Δ_C`:

```
Δ_C* = −H_CC⁻¹ H_CS Δ_S
```

Residual loss after optimal compensation:

```
ΔL* = ½ Δ_Sᵀ (H_SS − H_SC H_CC⁻¹ H_CS) Δ_S
```

The bracketed term is the Schur complement `H/H_CC`. This is the undo-ability metric: it tells us the minimum unavoidable loss for a given steal set after the best possible compensation.

### 1.2 Per-weight saliency

For single-weight steals with everything else in the compensation set, the Schur complement for weight `i` reduces to `1 / [H⁻¹]_ii`. Per-weight cost:

```
cost(i) = ½ · Δ_i² / [H⁻¹]_ii
```

This is the OBS/OBQ saliency. Allocation ranks by this, not by raw `H_ii` or `w_i² · H_ii`. A weight with large `H_ii` can still be cheap to steal from if another weight has strongly opposing coupling; conversely, a weight with small `H_ii` can be uncompensatable if its perturbation direction is isolated. `[H⁻¹]_ii` captures both effects.

### 1.3 The compensation operator

The object we actually cache per region is the linear operator that maps forced perturbations to optimal compensations:

```
M_R = −H_CC⁻¹ H_CR
```

At hot-path query time: `Δ_C = M_R · Δ_S`. One matrix-vector product. Compensation is linear in `Δ_S`, so the operator is independent of the specific perturbation magnitude — the cache doesn't need to know what value is being written.

## 2. The Hessian approximation

### 2.1 Layer-local, not global

We use the per-layer reconstruction Hessian:

```
H_layer = Xᵀ X
```

where `X` is the batch of input activations to that layer, computed from calibration data. This is the Hessian of squared reconstruction error for the layer's output, not the full-model training loss.

Rationale:
- Full-model Hessian is `n × n ≈ 4.9 × 10²¹` entries for a 70B model. Not tractable.
- Layer-local is closed-form, layer-sized (`d × d`), and dense-computable.
- Cross-layer coupling is real but second-order. Empirically (GPTQ, SparseGPT), greedy layer-wise compensation captures most of the available quality.
- If the fidelity tail demands it later, EK-FAC cross-layer compensation is the escalation path. Not in MVP.

### 2.2 Storage for the factorization

We store Cholesky factors of each layer's `H_layer`, not the raw matrix or its explicit inverse. All compensation solves become triangular solves against the factor — `d²` FLOPs per solve, sub-millisecond for `d = 8192`.

Factors are computed once from calibration at filesystem init (or on first write to a layer post-cold-start). Low-rank approximation of the Cholesky factor is on the table as a storage optimization: activation covariances typically have effective rank much lower than `d`, so `k = 512`-rank approximation of a `d = 8192` layer's factor gives ~95% fidelity at 6% of the storage. Worth measuring before committing to full-rank.

*D0 evidence:* `docs/phase-d-measurement.md` confirms the low-rank structure directly on SmolLM2-135M — site-specific effective rank ranges from ~3% of N (qkv) to ~17% of N (ffn_down) for 5% Frobenius error. Low-rank Cholesky is cheap quality-per-byte across every site measured.

## 3. The cache hierarchy

Three tiers, distinguished by latency and residency:

### 3.1 L1: precomputed operators `M_R` for hot regions

Per region `R` of size `r` with compensation neighborhood of size `c`, `M_R` is a `c × r` matrix. For `r = 64`, `c = 512`, fp16: ~64 KB per region. Across all cached regions in all layers, target working-set size: ~1–20 GB for a 70B model depending on how wide we draw the hot set.

Lookup: O(1) by region index. Apply: one matrix-vector product, microseconds.

### 3.2 L2: per-layer Cholesky factors

Full layer factor, resident when the layer is active. ~`d² · 2` bytes per layer at fp16. On cache miss at L1 (uncached region), we fall through to a direct mask-and-correct solve against the L2 factor. Millisecond latency.

*D0 evidence:* F16 and Q8_0 dumps agree on H to the third decimal across every (site, layer) Frobenius metric — F16 precision is sufficient for the factor; dequant noise is below the floor this layer cares about.

### 3.3 L3: raw Hessian / full recompute

Only accessed on drift-triggered recomputation. `H_layer` can be recomputed from the LLMDB-bundled calibration corpus — no external storage required, but seconds-to-minutes of compute. Background thread handles this; never on the hot path.

## 4. The hot-path loop

Three sequential steps per write operation:

```
1. Apply payload → Δ_S
2. Compute Δ_C = M_R · Δ_S, apply to compensation weights
3. Update cache:
   3a. Sherman-Morrison update of affected M entries (numerical)
   3b. If cleanliness thresholds crossed: update active-set metadata (structural)
```

Step 3a runs every write. Step 3b runs only when a weight transitions between cleanliness states (clean → partially-dirty → fully-dirty, or reverse on delete). Both are incremental. No full recompute in the hot path.

Steady-state latency target: microseconds per operation on L1 hit, low milliseconds on L2 fallback. Well under the seconds-scale budget.

## 5. Handling dirty weights in compensation

Dirty weights have filesystem-owned low bits and model-owned high bits. They can still participate in compensation, but with a per-weight quantization grid determined by remaining clean-bit count.

### 5.1 Mask and correct

Approach: keep `M_R` precomputed for the full neighborhood, ignoring cleanliness. At apply time:

1. Compute unconstrained `Δ_C = M_R · Δ_S`.
2. For each dirty weight in `C`, quantize its component of `Δ_C` to its clean-bit grid.
3. Do a small correction solve for the residual, using only fully-clean weights.

Correction solve size = number of dirty weights in the neighborhood. Cheap when dirty fraction `α` in the region is low; dominant cost when `α` approaches 0.5. Above that threshold, fall back to full recompute of `M_R` restricted to the currently-clean subset.

### 5.2 Compensation capacity is continuous

Each weight tracks "clean bits remaining" rather than a binary clean/dirty flag. The allocator and the compensation solve both see per-weight box constraints whose widths shrink as bits are stolen. This is the correct formulation; a binary flag is a coarse approximation of it.

## 6. Online cache maintenance

### 6.1 Sherman-Morrison updates

Each write is a low-rank perturbation to `H_CC`. We maintain the cached operator via Sherman-Morrison (or equivalent Cholesky/LDLᵀ updates). Per write: `O(c · r)` FLOPs per affected neighbor region. Cheap enough to run on a background thread keeping up with typical write rates.

### 6.2 Downdates for deletion

Deletion restores payload bits and re-adds the weight to the compensation active set. This is a Cholesky *downdate* — numerically touchier than updates but well-studied. LDLᵀ representation is preferred for mixed update/downdate workloads.

### 6.3 Drift bounds and full recompute triggers

Two drift sources:

1. **Numerical drift** from accumulated Sherman-Morrison roundoff. Rule of thumb: after `~d` updates to a `d × d` factor, drift becomes non-negligible. For `d = 8192`, trigger full recompute every ~8000 writes to that layer.

2. **Structural drift** from the Hessian itself shifting as activations change due to accumulated perturbation. Not captured by Sherman-Morrison (which assumes `H` is a fixed matrix plus low-rank updates). Detected by periodic fidelity checks; triggers recomputation of `H_layer` from fresh calibration data.

Both recomputations are background-thread work. Hot path never stalls; it may briefly use slightly stale operators during a background refresh.

## 7. Capacity and the fidelity ceiling

### 7.1 The cliff

Compensation efficacy degrades roughly linearly with dirty fraction `α` until a cliff, typically around `α ∈ [0.3, 0.5]` for layer-local compensation with reasonable neighborhood sizes. Past the cliff, compensation collapses and fidelity loss grows superlinearly.

The cliff sets practical capacity at ~30–50% of the information-theoretic ceiling. The info-theoretic ceiling itself is set by the Hessian's flat-subspace fraction and the chosen fidelity budget.

### 7.2 Levers that push the cliff later

1. **Coupling-aware allocation.** Pick steals whose compensation partners will remain available. Moves cliff from `α ≈ 0.3` to `α ≈ 0.4` in quantization literature. Recommended for capacity-critical deployments.

2. **Variable bit-depth per weight.** Allocate more payload bits to weights in flat directions, fewer to weights in sharper directions. Payload capacity per weight is proportional to `[H⁻¹]_ii`, not uniform. GPTQ forgoes this for hardware uniformity; we have no such constraint.

3. **Cross-layer compensation (EK-FAC).** Escalation for the capacity tail. Buys 5–10% more `α` at significant storage/speed cost. Not in MVP; add only if monitoring shows layer-local residuals are unacceptable.

### 7.3 The commitment

Past the cliff, degradation is inevitable. We accept this. The commitment is to delay it as far as possible via the levers above, and to surface a live fidelity metric (cumulative `½ Δᵀ H Δ`, plus periodic validation-set sampling) so that operations past the cliff are explicit rather than silent.

## 8. Pointer perturbation

The fixed root pointer (tens of weights, dirty bits, no compensation applied) contributes:

```
ΔL_pointer ≈ ½ · F · δ² · <H_ii>
```

For `F = 32`, `δ = 10⁻⁵`, `<H_ii> = 10⁻¹`: `ΔL ≈ 1.6 × 10⁻¹⁰`. Negligible. Accepted uncompensated.

Placement recommendation: choose pointer location at init from among the flattest-`H_ii` weights in layer 0. One-time optimization, zero runtime cost, strictly better than an arbitrary fixed location.

**Interaction with V2's anchor invariance.** The `H_ii`-minimal rule as stated is incompatible with V2's self-locating anchor (`DESIGN-NEW.MD §15.2`, `src/v2/anchor.rs`). The anchor must be findable on any reopen without reading any persisted metadata, so its placement rule must be **invariant under any stealable-bit write**. V2 uses ceiling magnitude — `|w|` with stealable bits at their worst-case value — which is invariant by construction because writes can only touch those bits. `H_ii = E[x_i²]` depends on activations, which depend on weights, which change under writes; so a rule that ranks weights by `H_ii` would land at different positions after any write, breaking the anchor's self-location.

The loss we'd save by switching is already `~10⁻¹⁰` (above). We keep the ceiling-magnitude rule, accept the residual perturbation, and treat this section's recommendation as applying only to contexts where persistence lets us remember a one-off placement (none exists in V2 today). If a future design introduces such a context (e.g., a persistent anchor-location table stored via some other invariant-preserving rule), the `H_ii` optimization becomes available then.

## 9. Cold-start behavior

No cache state is persistent. On restart:

1. Read root pointer (few weights, known location).
2. Bootstrap inode table via self-describing first entry.
3. Read metadata files (dirty-bit map, etc.).
4. Filesystem is operational for reads. Direct bit extraction needs no Hessian state.

Writes require the compensation machinery. First write to each layer triggers lazy computation of `H_layer` from the LLMDB-bundled calibration corpus — seconds per layer. Full steady-state reached after every write-targeted layer has been warmed.

**Scratch disk** (allowed by the at-rest-only stego invariant — local scratch is fine as long as it's wiped on unmount) is used for:
- Staging calibration activations during Hessian computation.
- Write-ahead log for crash consistency within multi-step operations.

No persistent state lives outside the filesystem itself.

## 10. Summary table of decisions

| Decision | Choice | Rationale |
|---|---|---|
| Compensation framework | Schur complement / OBS saliency | Optimal given quadratic approximation; well-studied |
| Hessian approximation | Layer-local `Xᵀ X` | Tractable; captures most available quality |
| Factor representation | Cholesky (optionally low-rank) | Fast triangular solves; low-rank exploits activation covariance structure (D0 evidence) |
| Cached object | `M_R = −H_CC⁻¹ H_CR` | Linear in `Δ_S`; independent of payload value |
| Cache tiering | L1 operators / L2 factors / L3 recompute | Matches 90/10 access pattern; bounded resident memory |
| Dirty-weight compensation | Mask and correct | Constant-time lookup; cheap correction for low `α` |
| Online maintenance | Sherman-Morrison + LDLᵀ downdates | Incremental; background thread |
| Full recompute trigger | `~d` updates, or fidelity drift | Bounds numerical and structural drift separately |
| Capacity ceiling | `α ≈ 0.3–0.5` (layer-local) | Empirical cliff from compression literature |
| Pointer compensation | None | `~10⁻¹⁰` effect; below noise floor |
| Cold-start | Lazy per-layer Hessian computation | No persistent cache state; at-rest-stego-invariant-compliant |
