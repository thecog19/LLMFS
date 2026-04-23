#!/usr/bin/env python3
"""Phase D0 analyzer — read dumped Hessians, emit a markdown decision report.

Input: the `target/hessian-dump/` directory produced by
`tests/hessian_measure.rs`, containing one subdirectory per cover, each
holding per-(site, layer) `.f32` upper-triangle files plus a `manifest.json`.

Output: markdown on stdout (pipe to `docs/phase-d-measurement.md`).

For each (cover, site, layer), computes:
  - eigenvalue spectrum (via numpy.linalg.eigvalsh),
  - symmetry error (sanity — should be 0) and negative-eigenvalue count
    (sanity — should be 0 for a PSD second-moment matrix),
  - smallest K capturing 90 / 95 / 99% of trace,
  - low-rank Frobenius error at K in {16, 32, 64, 128, 256, 512},
  - block-diagonal Frobenius error at block sizes
    {head_dim, 32, 64, 128, 256} (whichever divide N),
  - top-K-per-column sparse Frobenius error at K in {16, 32, 64, 128, 256}.

Aggregates per site type (median / min / max across layers per cover) and
writes a summary table. Runs in a few minutes for SmolLM2, tens of minutes
for Qwen-0.5B (dominated by the FFN-down site's 4864x4864 eigh).

Pure numpy + stdlib. No plotting.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np


# ---- tunables ---------------------------------------------------------

LOW_RANK_KS = [16, 32, 64, 128, 256, 512]
SPARSE_KS = [16, 32, 64, 128, 256]
BLOCK_SIZES_EXTRA = [32, 64, 128, 256]
TRACE_FRACTIONS = [0.90, 0.95, 0.99]


# ---- IO ---------------------------------------------------------------


def read_upper_triangle(path: Path, n: int) -> np.ndarray:
    """Load flat little-endian F32 upper triangle, return full N×N F64 matrix."""
    expected = n * (n + 1) // 2 * 4
    raw = path.read_bytes()
    if len(raw) != expected:
        raise RuntimeError(f"{path}: expected {expected} bytes, got {len(raw)}")
    tri = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    H = np.zeros((n, n), dtype=np.float64)
    off = 0
    for i in range(n):
        row_len = n - i
        H[i, i:] = tri[off:off + row_len]
        off += row_len
    # Mirror upper → lower. Leaves diagonal untouched.
    H = H + H.T - np.diag(np.diag(H))
    return H


# ---- per-matrix metrics ----------------------------------------------


@dataclass
class Metrics:
    site: str
    layer: int
    n: int
    trace: float
    frob_sq: float
    symmetry_err: float
    neg_eigenvalue_count: int
    # eigvals clipped to >= 0, descending.
    eigvals: np.ndarray
    # K_for_trace[0.90], [0.95], [0.99]
    k_for_trace: dict[float, int]
    # low_rank_err[K] = relative Frobenius error keeping top-K eigenpairs
    low_rank_err: dict[int, float]
    # block_err[block_size] = relative Frobenius error of block-diag approx
    block_err: dict[int, float]
    # sparse_err[K_per_col] = relative Frobenius error of top-K-per-col
    sparse_err: dict[int, float]


def analyze_matrix(site: str, layer: int, H: np.ndarray) -> Metrics:
    n = H.shape[0]
    sym_err = float(np.abs(H - H.T).max())
    eigvals = np.linalg.eigvalsh(H)[::-1]  # descending
    tol = 1e-6 * max(abs(eigvals).max(), 1e-30)
    neg_count = int((eigvals < -tol).sum())
    eigvals = np.clip(eigvals, 0.0, None)

    trace = float(eigvals.sum())
    frob_sq = float((eigvals ** 2).sum())

    # K for trace fractions
    k_for_trace: dict[float, int] = {}
    cum = np.cumsum(eigvals)
    total = cum[-1] if len(cum) else 0.0
    for frac in TRACE_FRACTIONS:
        if total < 1e-30:
            k_for_trace[frac] = 0
        else:
            idx = int(np.searchsorted(cum, frac * total))
            k_for_trace[frac] = idx + 1

    # Low-rank Frobenius error
    low_rank_err: dict[int, float] = {}
    for K in LOW_RANK_KS:
        if K >= n:
            low_rank_err[K] = 0.0
            continue
        if frob_sq < 1e-30:
            low_rank_err[K] = 0.0
            continue
        drop = float((eigvals[K:] ** 2).sum())
        low_rank_err[K] = float(np.sqrt(drop / frob_sq))

    # Block-diagonal Frobenius error
    block_err: dict[int, float] = {}
    frob_sq_full = float((H ** 2).sum())  # includes off-diag + diag, symmetric
    block_sizes = sorted(set(BLOCK_SIZES_EXTRA))
    for bs in block_sizes:
        if bs >= n or n % bs != 0:
            continue
        mask = _block_mask(n, bs)
        H_block = H * mask
        err_sq = float(((H - H_block) ** 2).sum())
        if frob_sq_full < 1e-30:
            block_err[bs] = 0.0
        else:
            block_err[bs] = float(np.sqrt(err_sq / frob_sq_full))

    # Top-K sparse Frobenius error (K per column, plus diagonal)
    sparse_err: dict[int, float] = {}
    abs_H = np.abs(H)
    for K in SPARSE_KS:
        if K >= n:
            sparse_err[K] = 0.0
            continue
        mask = np.zeros_like(H, dtype=bool)
        # For each column, mark top-K indices by |value|.
        # argpartition on -abs_H gives K largest at front.
        topk_idx = np.argpartition(-abs_H, K, axis=0)[:K, :]
        col_indices = np.broadcast_to(np.arange(n), (K, n))
        mask[topk_idx, col_indices] = True
        np.fill_diagonal(mask, True)
        H_sparse = H * mask
        err_sq = float(((H - H_sparse) ** 2).sum())
        if frob_sq_full < 1e-30:
            sparse_err[K] = 0.0
        else:
            sparse_err[K] = float(np.sqrt(err_sq / frob_sq_full))

    return Metrics(
        site=site,
        layer=layer,
        n=n,
        trace=trace,
        frob_sq=frob_sq,
        symmetry_err=sym_err,
        neg_eigenvalue_count=neg_count,
        eigvals=eigvals,
        k_for_trace=k_for_trace,
        low_rank_err=low_rank_err,
        block_err=block_err,
        sparse_err=sparse_err,
    )


def _block_mask(n: int, block_size: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    for b in range(0, n, block_size):
        mask[b:b + block_size, b:b + block_size] = True
    return mask


# ---- aggregation ------------------------------------------------------


def aggregate_per_site(metrics: list[Metrics]) -> dict[str, dict]:
    """Group metrics by site, compute median / min / max across layers."""
    by_site: dict[str, list[Metrics]] = {}
    for m in metrics:
        by_site.setdefault(m.site, []).append(m)
    out: dict[str, dict] = {}
    for site, ms in sorted(by_site.items()):
        ns = sorted({m.n for m in ms})
        # All layers at a given site should share N; sanity check.
        if len(ns) > 1:
            print(
                f"WARN: site {site} has inconsistent N across layers: {ns}",
                file=sys.stderr,
            )
        out[site] = {
            "n_layers": len(ms),
            "n": ns[0],
            "symmetry_err_max": max(m.symmetry_err for m in ms),
            "neg_eig_max": max(m.neg_eigenvalue_count for m in ms),
            "trace_median": float(np.median([m.trace for m in ms])),
            "k_for_trace": {
                frac: {
                    "median": int(np.median([m.k_for_trace[frac] for m in ms])),
                    "min": min(m.k_for_trace[frac] for m in ms),
                    "max": max(m.k_for_trace[frac] for m in ms),
                }
                for frac in TRACE_FRACTIONS
            },
            "low_rank_err": {
                K: {
                    "median": float(np.median([m.low_rank_err[K] for m in ms])),
                    "max": max(m.low_rank_err[K] for m in ms),
                }
                for K in LOW_RANK_KS
            },
            "block_err": {
                bs: {
                    "median": float(np.median([m.block_err[bs] for m in ms if bs in m.block_err])),
                    "max": max((m.block_err[bs] for m in ms if bs in m.block_err), default=0.0),
                }
                for bs in sorted({bs for m in ms for bs in m.block_err})
            },
            "sparse_err": {
                K: {
                    "median": float(np.median([m.sparse_err[K] for m in ms])),
                    "max": max(m.sparse_err[K] for m in ms),
                }
                for K in SPARSE_KS
            },
        }
    return out


# ---- formatting -------------------------------------------------------


def fmt_float(x: float) -> str:
    if x < 1e-4:
        return f"{x:.2e}"
    return f"{x:.4f}"


def storage_per_layer_bytes(n: int, strategy: str, K: int) -> int:
    if strategy == "full":
        return n * (n + 1) // 2 * 4
    if strategy == "low_rank":
        # K eigenvalues (f32) + K eigenvectors of length n (f32)
        return K * 4 + K * n * 4
    if strategy == "block_diag":
        # n/K blocks of K×K upper triangle each = (n/K) * K*(K+1)/2 * 4
        if n % K != 0:
            return -1
        return (n // K) * K * (K + 1) // 2 * 4
    if strategy == "sparse":
        # K entries per col + diagonal. Each entry: (row_idx: u16, value: f32)
        # = (K * n entries) * (2 + 4) bytes. Plus n diagonal (f32, row implicit).
        # But in practice COO would store (row, col, val) — simplify.
        return K * n * 6 + n * 4
    raise ValueError(strategy)


def write_report(
    manifest: dict,
    per_layer: list[Metrics],
    per_site: dict[str, dict],
    out,
):
    cover = manifest["cover"]
    tag = manifest["cover_tag"]
    T = manifest["token_count"]
    hidden = manifest["hidden_dim"]
    ffn = manifest["ffn_dim"]
    n_layers = manifest["n_layers"]

    print(f"## {tag}", file=out)
    print(file=out)
    print(f"- Cover: `{cover}`", file=out)
    print(f"- Corpus: `{manifest['corpus']}`", file=out)
    print(f"- Token count (T): {T}", file=out)
    print(
        f"- Shape: hidden_dim={hidden}, ffn_dim={ffn}, "
        f"n_layers={n_layers}, n_heads={manifest['n_heads']}, "
        f"n_kv_heads={manifest['n_kv_heads']}, head_dim={manifest['head_dim']}",
        file=out,
    )
    print(file=out)

    # Sanity
    max_sym = max(m.symmetry_err for m in per_layer)
    max_neg = max(m.neg_eigenvalue_count for m in per_layer)
    print("### Sanity", file=out)
    print(f"- max symmetry error across all (site, layer): {fmt_float(max_sym)}", file=out)
    print(
        f"- max negative-eigenvalue count across all (site, layer): {max_neg} "
        f"(should be 0 for PSD)",
        file=out,
    )
    print(file=out)

    # Per-site summary
    for site, agg in per_site.items():
        N = agg["n"]
        print(f"### {site} (N = {N}, {agg['n_layers']} layers)", file=out)
        print(file=out)

        print("**Trace fractions (median / min / max K across layers):**", file=out)
        print(file=out)
        print("| Fraction of trace | Median K | Min K | Max K | K/N median |", file=out)
        print("|---|---|---|---|---|", file=out)
        for frac in TRACE_FRACTIONS:
            kft = agg["k_for_trace"][frac]
            print(
                f"| {int(frac*100)}% | {kft['median']} | {kft['min']} | {kft['max']} "
                f"| {kft['median']/N:.3f} |",
                file=out,
            )
        print(file=out)

        print("**Low-rank approximation (relative Frobenius error):**", file=out)
        print(file=out)
        print("| K | median err | max err | storage / layer |", file=out)
        print("|---|---|---|---|", file=out)
        for K in LOW_RANK_KS:
            lr = agg["low_rank_err"][K]
            bytes_est = storage_per_layer_bytes(N, "low_rank", K)
            print(
                f"| {K} | {fmt_float(lr['median'])} | {fmt_float(lr['max'])} "
                f"| {bytes_est/1024:.1f} KB |",
                file=out,
            )
        print(file=out)

        if agg["block_err"]:
            print("**Block-diagonal approximation (relative Frobenius error):**", file=out)
            print(file=out)
            print("| Block size | median err | max err | storage / layer |", file=out)
            print("|---|---|---|---|", file=out)
            for bs in sorted(agg["block_err"].keys()):
                be = agg["block_err"][bs]
                bytes_est = storage_per_layer_bytes(N, "block_diag", bs)
                print(
                    f"| {bs} | {fmt_float(be['median'])} | {fmt_float(be['max'])} "
                    f"| {bytes_est/1024:.1f} KB |",
                    file=out,
                )
            print(file=out)

        print("**Top-K sparse per column (relative Frobenius error):**", file=out)
        print(file=out)
        print("| K | median err | max err | storage / layer |", file=out)
        print("|---|---|---|---|", file=out)
        for K in SPARSE_KS:
            sp = agg["sparse_err"][K]
            bytes_est = storage_per_layer_bytes(N, "sparse", K)
            print(
                f"| {K} | {fmt_float(sp['median'])} | {fmt_float(sp['max'])} "
                f"| {bytes_est/1024:.1f} KB |",
                file=out,
            )
        print(file=out)

        # Full-H reference size
        full_bytes = storage_per_layer_bytes(N, "full", 0)
        print(f"*Reference: full upper triangle = {full_bytes/1024:.1f} KB / layer.*", file=out)
        print(file=out)


# ---- main -------------------------------------------------------------


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(f"usage: {sys.argv[0]} <hessian-dump-dir> [covers...]", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"{root}: not a directory", file=sys.stderr)
        return 2

    # Each subdirectory is a cover.
    cover_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if not cover_dirs:
        print(f"{root}: no cover subdirectories found", file=sys.stderr)
        return 2

    print("# Phase D0 — Hessian measurement", file=sys.stdout)
    print(file=sys.stdout)
    print(
        "*Generated by `scripts/analyze-hessian.py`; manually append "
        "the decision section.*",
        file=sys.stdout,
    )
    print(file=sys.stdout)

    for cover_dir in cover_dirs:
        manifest_path = cover_dir / "manifest.json"
        if not manifest_path.exists():
            print(f"skip {cover_dir}: no manifest.json", file=sys.stderr)
            continue
        manifest = json.loads(manifest_path.read_text())

        print(f"Analyzing {cover_dir}...", file=sys.stderr)
        per_layer: list[Metrics] = []
        for entry in manifest["entries"]:
            site = entry["site"]
            layer = int(entry["layer"])
            n = int(entry["n"])
            fpath = cover_dir / entry["file"]
            H = read_upper_triangle(fpath, n)
            m = analyze_matrix(site, layer, H)
            per_layer.append(m)
            print(
                f"  layer={layer:3d} site={site:22s} N={n:5d} "
                f"K95={m.k_for_trace[0.95]:5d}  "
                f"symErr={m.symmetry_err:.1e} negEig={m.neg_eigenvalue_count}",
                file=sys.stderr,
            )

        per_site = aggregate_per_site(per_layer)
        write_report(manifest, per_layer, per_site, sys.stdout)

    print("---", file=sys.stdout)
    print(file=sys.stdout)
    print("## Decision", file=sys.stdout)
    print(file=sys.stdout)
    print(
        "_To be filled in manually after reviewing the tables above. "
        "Should specify: (a) structure chosen (low-rank / block-diagonal / "
        "top-K sparse / something else), (b) parameters (K, block size, etc.), "
        "(c) quality guarantee (e.g. Frobenius error ≤ X% across all sites), "
        "(d) storage cost (fraction of V2 capacity at SmolLM2 / Qwen / 3B / 7B), "
        "(e) how this feeds into D1's production collector and codec._",
        file=sys.stdout,
    )
    print(file=sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
