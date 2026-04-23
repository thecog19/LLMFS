//! D0 measurement — full-Hessian accumulator.
//!
//! Parallels [`crate::forward::awq::AwqCollector`] but accumulates
//! the full symmetric second-moment matrix `H = E[x x^T]` per
//! observation site, not just the per-channel diagonal. This is the
//! reference implementation used *once*, during the Phase D0
//! measurement step, to decide which structured approximation Phase
//! D's production path should commit to (sparse, block-diagonal,
//! low-rank, …). It is not on any production calibration path.
//!
//! Storage: per `(site, layer)`, one `Vec<f64>` of length
//! `N·(N+1)/2` holding the upper triangle of `Σ_t x_t x_t^T`, in
//! row-major order within the triangle. Row `i` contributes `N - i`
//! entries, covering columns `i..N` (diagonal first).
//!
//! F64 is deliberate — D0 is supposed to be the numerical reference
//! against which approximation error is measured, so any F32
//! rounding is kept *out* of the reference.
//!
//! Observer wiring: implements [`BlockObserver`] the same way
//! `AwqCollector` does, so it drops into
//! [`crate::forward::model::ForwardModel::forward_all_logits_with_observer`]
//! with no block-level changes.
//!
//! RAM profile: for SmolLM2-135M at F64 upper triangle the whole
//! accumulator sits around 400 MB; for Qwen2.5-0.5B around 2.5 GB.
//! Both comfortably fit on a dev box. The accumulator isn't
//! intended to scale beyond those.
//!
//! Throughput: the inner loop is a rank-1 symmetric update per
//! token (SYR-style), implemented as a plain `f64` scalar loop with
//! no external BLAS dep. Per-call cost is `O(rows · N·(N+1)/2)`;
//! dominated in practice by the FfnDownInput site, whose `N` is
//! `ffn_dim`.
//!
//! This module will likely be deleted or superseded once D0 lands
//! its decision and D1 builds the real (approximated) collector.

use std::collections::HashMap;

use crate::forward::awq::ActivationSite;
use crate::forward::block::BlockObserver;

/// Full-H accumulator keyed by `(site, layer)`.
#[derive(Debug, Default)]
pub struct HessianAccumulator {
    entries: HashMap<(ActivationSite, usize), HessianEntry>,
}

/// Per-(site, layer) accumulated upper triangle plus token count.
#[derive(Debug)]
struct HessianEntry {
    /// Matrix dimension `N` — equal to `cols` at the first observe
    /// call for this key. Subsequent observe calls assert equality.
    n: usize,
    /// Upper triangle of `Σ_t x_t x_t^T`, row-major within the
    /// triangle. Length `n * (n + 1) / 2`.
    tri: Vec<f64>,
    /// Number of token rows accumulated.
    tokens: u64,
}

impl HessianAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop every accumulated entry without deallocating the map.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Matrix dim `N` for `(site, layer)`, or `None` if unseen.
    pub fn dim(&self, site: ActivationSite, layer: usize) -> Option<usize> {
        self.entries.get(&(site, layer)).map(|e| e.n)
    }

    /// Token count at `(site, layer)`, or 0 if unseen.
    pub fn token_count(&self, site: ActivationSite, layer: usize) -> u64 {
        self.entries.get(&(site, layer)).map(|e| e.tokens).unwrap_or(0)
    }

    /// Raw upper-triangle sum `Σ_t x_t x_t^T` at `(site, layer)` —
    /// not normalized by token count. Row-major within the
    /// triangle. `None` if unseen.
    pub fn raw_upper_triangle(&self, site: ActivationSite, layer: usize) -> Option<&[f64]> {
        self.entries.get(&(site, layer)).map(|e| e.tri.as_slice())
    }

    /// Finalize to `H = (1/T) Σ_t x_t x_t^T`, cast to `f32` upper
    /// triangle per `(site, layer)`. Returns `(N, Vec<f32>)` where
    /// the vector has `N*(N+1)/2` entries. Unseen keys are omitted.
    pub fn finalize(&self) -> HashMap<(ActivationSite, usize), (usize, Vec<f32>)> {
        let mut out = HashMap::with_capacity(self.entries.len());
        for (&key, entry) in self.entries.iter() {
            if entry.tokens == 0 {
                continue;
            }
            let inv = 1.0_f64 / entry.tokens as f64;
            let tri: Vec<f32> = entry.tri.iter().map(|&v| (v * inv) as f32).collect();
            out.insert(key, (entry.n, tri));
        }
        out
    }

    /// Number of `(site, layer)` keys observed so far.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl BlockObserver for HessianAccumulator {
    fn observe(
        &mut self,
        site: ActivationSite,
        layer: usize,
        x: &[f32],
        rows: usize,
        cols: usize,
    ) {
        assert_eq!(
            x.len(),
            rows * cols,
            "x length {} does not match rows*cols = {}*{}",
            x.len(),
            rows,
            cols,
        );
        let entry = self.entries.entry((site, layer)).or_insert_with(|| HessianEntry {
            n: cols,
            tri: vec![0.0_f64; cols * (cols + 1) / 2],
            tokens: 0,
        });
        assert_eq!(
            entry.n, cols,
            "observe({site:?}, layer={layer}): cols changed from {} to {cols}",
            entry.n,
        );

        // Rank-1 symmetric update per token, into the upper triangle.
        // Upper-triangle row-major layout: row i covers columns i..N,
        // contributing N-i entries.
        let n = cols;
        for r in 0..rows {
            let row = &x[r * n..(r + 1) * n];
            let mut offset = 0_usize;
            for i in 0..n {
                let xi = row[i] as f64;
                let dst = &mut entry.tri[offset..offset + (n - i)];
                for (slot, &xj) in dst.iter_mut().zip(row[i..].iter()) {
                    *slot += xi * xj as f64;
                }
                offset += n - i;
            }
            debug_assert_eq!(offset, n * (n + 1) / 2);
        }
        entry.tokens += rows as u64;
    }
}

/// Compute the flat offset of entry `(i, j)` (with `i <= j`) inside
/// an upper-triangle row-major layout for an `N × N` matrix.
/// Offset formula: `i*N - i*(i-1)/2 + (j - i)`.
#[inline]
pub fn upper_tri_offset(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i <= j, "upper-triangle offset requires i <= j");
    debug_assert!(j < n, "upper-triangle offset: j out of range");
    i * n - i * i.saturating_sub(1) / 2 + (j - i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_token_triangle(x: &[f32]) -> Vec<f64> {
        // Hand-roll the expected upper-triangle for Σ = x x^T on
        // a single token, for test oracle purposes.
        let n = x.len();
        let mut tri = Vec::with_capacity(n * (n + 1) / 2);
        for i in 0..n {
            for j in i..n {
                tri.push(x[i] as f64 * x[j] as f64);
            }
        }
        tri
    }

    #[test]
    fn offset_matches_row_major_layout() {
        // For N=4, upper-triangle flat layout is:
        //   row 0: (0,0) (0,1) (0,2) (0,3)   → offsets 0..=3
        //   row 1:       (1,1) (1,2) (1,3)   → offsets 4..=6
        //   row 2:             (2,2) (2,3)   → offsets 7..=8
        //   row 3:                   (3,3)   → offset 9
        let n = 4;
        assert_eq!(upper_tri_offset(n, 0, 0), 0);
        assert_eq!(upper_tri_offset(n, 0, 3), 3);
        assert_eq!(upper_tri_offset(n, 1, 1), 4);
        assert_eq!(upper_tri_offset(n, 1, 3), 6);
        assert_eq!(upper_tri_offset(n, 2, 2), 7);
        assert_eq!(upper_tri_offset(n, 2, 3), 8);
        assert_eq!(upper_tri_offset(n, 3, 3), 9);
    }

    #[test]
    fn observe_single_token_produces_outer_product_upper_triangle() {
        // x = [1, 2, 3]  → xx^T = [[1,2,3],[2,4,6],[3,6,9]]
        // upper triangle row-major = [1, 2, 3, 4, 6, 9]
        let mut acc = HessianAccumulator::new();
        let x = [1.0_f32, 2.0, 3.0];
        acc.observe(ActivationSite::QkvInput, 0, &x, 1, 3);
        let tri = acc.raw_upper_triangle(ActivationSite::QkvInput, 0).unwrap();
        assert_eq!(tri, &[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]);
        assert_eq!(acc.token_count(ActivationSite::QkvInput, 0), 1);
        assert_eq!(acc.dim(ActivationSite::QkvInput, 0), Some(3));
    }

    #[test]
    fn observe_two_tokens_accumulates_outer_products() {
        // Two tokens: x1 = [1, 0], x2 = [0, 1]
        // x1 x1^T = [[1,0],[0,0]]
        // x2 x2^T = [[0,0],[0,1]]
        // sum     = [[1,0],[0,1]]  → upper triangle [1, 0, 1]
        let mut acc = HessianAccumulator::new();
        let x = [1.0_f32, 0.0, 0.0, 1.0]; // row 0 = [1,0], row 1 = [0,1]
        acc.observe(ActivationSite::QkvInput, 0, &x, 2, 2);
        let tri = acc.raw_upper_triangle(ActivationSite::QkvInput, 0).unwrap();
        assert_eq!(tri, &[1.0, 0.0, 1.0]);
        assert_eq!(acc.token_count(ActivationSite::QkvInput, 0), 2);
    }

    #[test]
    fn observe_sums_across_multiple_calls_at_same_key() {
        // Two separate observe calls for the same (site, layer)
        // should accumulate the triangle and the token count.
        let mut acc = HessianAccumulator::new();
        let x1 = [1.0_f32, 2.0, 3.0];
        let x2 = [4.0_f32, 5.0, 6.0];
        acc.observe(ActivationSite::FfnDownInput, 2, &x1, 1, 3);
        acc.observe(ActivationSite::FfnDownInput, 2, &x2, 1, 3);

        let mut expected = single_token_triangle(&x1);
        for (slot, v) in expected.iter_mut().zip(single_token_triangle(&x2)) {
            *slot += v;
        }
        let got = acc.raw_upper_triangle(ActivationSite::FfnDownInput, 2).unwrap();
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-9, "got {g}, expected {e}");
        }
        assert_eq!(acc.token_count(ActivationSite::FfnDownInput, 2), 2);
    }

    #[test]
    fn observe_keeps_sites_independent() {
        let mut acc = HessianAccumulator::new();
        let x = [1.0_f32, 2.0, 3.0];
        acc.observe(ActivationSite::QkvInput, 0, &x, 1, 3);
        acc.observe(ActivationSite::FfnDownInput, 0, &x, 1, 3);
        // Same layer, different sites — same numerical content but
        // distinct keys.
        assert_eq!(acc.len(), 2);
        assert!(acc.raw_upper_triangle(ActivationSite::QkvInput, 0).is_some());
        assert!(acc.raw_upper_triangle(ActivationSite::FfnDownInput, 0).is_some());
        // And no accidental sharing of the vector.
        assert_eq!(acc.token_count(ActivationSite::QkvInput, 0), 1);
        assert_eq!(acc.token_count(ActivationSite::FfnDownInput, 0), 1);
    }

    #[test]
    fn observe_keeps_layers_independent() {
        let mut acc = HessianAccumulator::new();
        let x = [1.0_f32, 2.0];
        acc.observe(ActivationSite::QkvInput, 0, &x, 1, 2);
        acc.observe(ActivationSite::QkvInput, 1, &x, 1, 2);
        acc.observe(ActivationSite::QkvInput, 1, &x, 1, 2);
        assert_eq!(acc.token_count(ActivationSite::QkvInput, 0), 1);
        assert_eq!(acc.token_count(ActivationSite::QkvInput, 1), 2);
    }

    #[test]
    fn finalize_divides_by_token_count() {
        // Four identical tokens x=[2,0,0] → each outer product
        // contributes [[4,0,0],[0,0,0],[0,0,0]]; sum = [[16,0,0]...];
        // mean with T=4 = [[4,0,0]...]. Upper triangle mean = [4,0,0,0,0,0].
        let mut acc = HessianAccumulator::new();
        let x = [2.0_f32, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        acc.observe(ActivationSite::QkvInput, 0, &x, 4, 3);
        let finalized = acc.finalize();
        let (n, tri) = finalized.get(&(ActivationSite::QkvInput, 0)).unwrap();
        assert_eq!(*n, 3);
        assert_eq!(tri, &vec![4.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn finalize_skips_unseen_keys() {
        let acc = HessianAccumulator::new();
        assert!(acc.finalize().is_empty());
        assert!(acc.is_empty());
    }

    #[test]
    fn clear_resets_state() {
        let mut acc = HessianAccumulator::new();
        let x = [1.0_f32, 1.0];
        acc.observe(ActivationSite::QkvInput, 0, &x, 1, 2);
        assert_eq!(acc.len(), 1);
        acc.clear();
        assert!(acc.is_empty());
        assert_eq!(acc.token_count(ActivationSite::QkvInput, 0), 0);
        assert!(acc.dim(ActivationSite::QkvInput, 0).is_none());
    }

    #[test]
    #[should_panic(expected = "cols changed")]
    fn observe_panics_on_dim_mismatch_for_same_key() {
        let mut acc = HessianAccumulator::new();
        let x3 = [1.0_f32, 2.0, 3.0];
        let x2 = [1.0_f32, 2.0];
        acc.observe(ActivationSite::QkvInput, 0, &x3, 1, 3);
        // Same (site, layer), different cols — must panic.
        acc.observe(ActivationSite::QkvInput, 0, &x2, 1, 2);
    }

    #[test]
    #[should_panic(expected = "x length")]
    fn observe_panics_on_length_mismatch() {
        let mut acc = HessianAccumulator::new();
        let x = [1.0_f32, 2.0, 3.0];
        acc.observe(ActivationSite::QkvInput, 0, &x, 2, 2); // claims 4 elements but only has 3
    }

    #[test]
    fn produced_matrix_is_positive_semi_definite_on_real_input() {
        // H = Σ x_t x_t^T is PSD by construction. Round-trip through
        // the accumulator and confirm on a small random-ish input.
        let mut acc = HessianAccumulator::new();
        let tokens: &[&[f32]] = &[
            &[1.0, 2.0, -1.0, 0.5],
            &[0.0, 1.0, 1.0, -1.0],
            &[2.0, 0.0, 0.5, 0.0],
            &[-1.0, -1.0, 0.0, 2.0],
        ];
        let mut flat = Vec::with_capacity(4 * 4);
        for t in tokens {
            flat.extend_from_slice(t);
        }
        acc.observe(ActivationSite::FfnDownInput, 5, &flat, 4, 4);
        let tri = acc.raw_upper_triangle(ActivationSite::FfnDownInput, 5).unwrap();

        // Reconstruct full H from the upper triangle.
        let n = 4;
        let mut full = vec![0.0_f64; n * n];
        let mut off = 0;
        for i in 0..n {
            for j in i..n {
                full[i * n + j] = tri[off];
                full[j * n + i] = tri[off];
                off += 1;
            }
        }

        // Sanity: diagonal entries must be non-negative sums of
        // squares. Cheap PSD evidence; the full eigenvalue check
        // lives in the Python analyzer.
        for i in 0..n {
            assert!(full[i * n + i] >= 0.0, "diag[{i}] = {} is negative", full[i * n + i]);
        }
        // Symmetry.
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (full[i * n + j] - full[j * n + i]).abs() < 1e-12,
                    "not symmetric at ({i},{j})"
                );
            }
        }
    }
}
