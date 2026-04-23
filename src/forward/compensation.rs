//! Phase E compensation primitives — the math that maps forced
//! stego perturbations onto optimal counter-adjustments.
//!
//! This module implements the single-weight case from
//! [`docs/compensation-design.md §1.1`](../../../docs/compensation-design.md).
//! Multi-weight regions, the `M_R` operator cache, Sherman-Morrison
//! online maintenance, and the mask-and-correct dirty-weight path
//! are later Phase E slices.
//!
//! ## The single-weight case
//!
//! A forced perturbation `Δ[j] = ε` at a single input channel `j`,
//! with every other channel free, optimally minimizes `½Δᵀ H Δ`
//! when:
//!
//! ```text
//! Δ[k] = (H⁻¹[k, j] / H⁻¹[j, j]) · ε    for k ≠ j
//! Δ[j] = ε
//! ```
//!
//! (Derivation: Lagrangian `L = ½ΔᵀHΔ - λ(Δ[j] - ε)` yields
//! `HΔ = λ e_j`, so `Δ = λ H⁻¹ e_j`. Solve for `λ` using the
//! constraint `Δ[j] = ε`.)
//!
//! This module produces the **compensation factor vector** `c`,
//! where `c[k] = H⁻¹[k, j] / H⁻¹[j, j]` for `k ≠ j` and `c[j] = 0`.
//! Scaling by `ε` gives the compensation to apply alongside the
//! forced perturbation: `Δ_compensation[k] = ε · c[k]`.
//!
//! ## Cost
//!
//! `O(N²)` per weight. The two triangular solves against the
//! cached Cholesky factor dominate; neither depends on the
//! perturbation magnitude, so we can precompute `c` per-channel
//! and amortize across all writes targeting that channel.

use thiserror::Error;

use crate::forward::hessian::upper_tri_offset;
use crate::forward::linalg::{
    CholeskyError, cholesky, h_inv_column, solve_lower, solve_lower_transposed,
};

#[derive(Debug, Error)]
pub enum CompensationError {
    #[error("region contains duplicate index {index}")]
    DuplicateRegionIndex { index: usize },

    #[error("region index {index} out of range for n = {n}")]
    RegionIndexOutOfRange { n: usize, index: usize },

    #[error("cholesky on inv(H)[R, R]: {source}")]
    SubmatrixCholesky {
        #[source]
        source: CholeskyError,
    },
}

/// Compensation operator `M_R = -H_CC⁻¹ H_CR` for a region `R` of
/// input channels, per [`compensation-design.md §1.3`](../../../docs/compensation-design.md).
/// For a forced perturbation `Δ[R]` (an `r`-vector), the optimal
/// compensation at the free channels `C = {0..n} \ R` is
/// `Δ[C] = M_R · Δ[R]`, an `|C|`-vector.
///
/// Derived from the block-inverse identity `inv(H)[C,R] =
/// -H_CC⁻¹ H_CR inv(H)[R,R]`, so `M_R = inv(H)[C,R] ·
/// inv(inv(H)[R,R])`. Implementation:
///
/// 1. Extract `inv(H)[:, j]` for each `j ∈ R` via `h_inv_column`
///    (two triangular solves per column, `O(r · n²)`).
/// 2. Slice out the `r × r` submatrix `inv(H)[R, R]`.
/// 3. Cholesky-factor it (small, `O(r³)`).
/// 4. For each `c ∈ C`, solve `inv(H)[R,R] · x = (inv(H)[c, R])ᵀ`
///    via the submatrix factor; `x` becomes the `c`-th row of
///    `M_R`.
#[derive(Debug, Clone)]
pub struct CompensationOperator {
    /// Layer dimension.
    pub n: usize,
    /// Sorted, deduplicated region indices `R ⊆ {0..n}`. Length `r`.
    pub region: Vec<usize>,
    /// Complementary free channels `C = {0..n} \ R`, sorted.
    /// Length `|C| = n − r`.
    pub c_indices: Vec<usize>,
    /// `M_R` in column-major packed storage: `m[j * |C| + i] =
    /// M_R[i, j]` where `i` indexes `c_indices` and `j` indexes
    /// `region`. Length `|C| · r`.
    pub m: Vec<f32>,
}

impl CompensationOperator {
    pub fn region_size(&self) -> usize {
        self.region.len()
    }

    pub fn free_size(&self) -> usize {
        self.c_indices.len()
    }

    /// Apply `M_R` to a forced-perturbation vector `delta_r` of
    /// length `r`. Returns the `|C|`-vector of compensation
    /// values. `delta_c[i]` is the compensation at channel
    /// `c_indices[i]`.
    pub fn apply(&self, delta_r: &[f32]) -> Vec<f32> {
        assert_eq!(
            delta_r.len(),
            self.region_size(),
            "apply: delta_r length {} != region_size {}",
            delta_r.len(),
            self.region_size(),
        );
        let c = self.free_size();
        let r = self.region_size();
        let mut out = vec![0.0_f32; c];
        for i in 0..c {
            let mut sum = 0.0_f64;
            for j in 0..r {
                sum += self.m[j * c + i] as f64 * delta_r[j] as f64;
            }
            out[i] = sum as f32;
        }
        out
    }
}

/// Compute the compensation operator `M_R` for a region of input
/// channels, given the Cholesky factor `l` of the layer's Hessian.
///
/// See [`CompensationOperator`] for the math and the output
/// layout. `region` must contain sorted, unique indices in
/// `0..n`; duplicates and out-of-range entries are rejected with
/// [`CompensationError`].
///
/// Returns an empty operator (zero free size, zero region size)
/// when `region` is empty — the identity case, where nothing is
/// perturbed and nothing needs compensation.
pub fn compensation_operator(
    l: &[f32],
    n: usize,
    region: &[usize],
) -> Result<CompensationOperator, CompensationError> {
    // Validate region: unique, in-range.
    let mut seen = vec![false; n];
    for &j in region {
        if j >= n {
            return Err(CompensationError::RegionIndexOutOfRange { n, index: j });
        }
        if seen[j] {
            return Err(CompensationError::DuplicateRegionIndex { index: j });
        }
        seen[j] = true;
    }

    // Sort region canonically (caller may pass unsorted).
    let mut region_sorted: Vec<usize> = region.to_vec();
    region_sorted.sort_unstable();

    // C = {0..n} \ R, sorted ascending.
    let c_indices: Vec<usize> = (0..n).filter(|i| !seen[*i]).collect();
    let r = region_sorted.len();
    let c_len = c_indices.len();

    if r == 0 {
        return Ok(CompensationOperator {
            n,
            region: Vec::new(),
            c_indices,
            m: Vec::new(),
        });
    }

    // 1. Extract inv(H)[:, R]. Row-major by region index:
    //    h_inv_cols[ri * n + k] = inv(H)[k, region_sorted[ri]].
    let mut h_inv_cols = vec![0.0_f32; r * n];
    for (ri, &j) in region_sorted.iter().enumerate() {
        let col = h_inv_column(l, n, j);
        h_inv_cols[ri * n..(ri + 1) * n].copy_from_slice(&col);
    }

    // 2. Build inv(H)[R, R] as an r×r upper-triangle-packed matrix.
    //    Entry (i, j) for i ≤ j uses the symmetric value
    //    inv(H)[region_sorted[i], region_sorted[j]] =
    //    h_inv_cols[j * n + region_sorted[i]].
    let mut hinv_rr = vec![0.0_f32; r * (r + 1) / 2];
    for i in 0..r {
        for j in i..r {
            hinv_rr[upper_tri_offset(r, i, j)] = h_inv_cols[j * n + region_sorted[i]];
        }
    }

    // 3. Cholesky-factor the r×r submatrix.
    let l_rr = cholesky(&hinv_rr, r)
        .map_err(|e| CompensationError::SubmatrixCholesky { source: e })?;

    // 4. For each free channel c, solve inv(H)[R,R] · x =
    //    (inv(H)[c, R])ᵀ to get the c-th row of M_R (as an
    //    r-vector), then stash it column-wise into m.
    let mut m = vec![0.0_f32; c_len * r];
    let mut rhs = vec![0.0_f32; r];
    for (ci_idx, &c) in c_indices.iter().enumerate() {
        // inv(H)[c, region_sorted[ri]] = h_inv_cols[ri * n + c]
        // (symmetry of inv(H)).
        for ri in 0..r {
            rhs[ri] = h_inv_cols[ri * n + c];
        }
        solve_lower(&l_rr, r, &mut rhs);
        solve_lower_transposed(&l_rr, r, &mut rhs);
        // rhs now holds x = inv(inv(H)[R,R]) · (inv(H)[c, R])ᵀ.
        // Write as column of M_R: m[j * c_len + ci_idx] = x[j].
        for j in 0..r {
            m[j * c_len + ci_idx] = rhs[j];
        }
    }

    Ok(CompensationOperator {
        n,
        region: region_sorted,
        c_indices,
        m,
    })
}

/// Compensation factor vector for a forced perturbation at input
/// channel `j` of a layer whose Hessian is represented by the
/// lower-triangle-packed Cholesky factor `l`.
///
/// Returns an N-vector `c` such that, for any `ε`:
///
/// - `Δ_compensation[k] = ε · c[k]` is the optimal counter-adjustment
///   at channel `k ≠ j`,
/// - `c[j] = 0` (the forced channel contributes `ε` directly, not
///   via compensation),
/// - applying `Δ_compensation` alongside the forced `Δ[j] = ε`
///   zeroes out the residual `H · Δ` in every free channel.
///
/// Cost: one `h_inv_column` call (two triangular solves, `O(n²)`).
pub fn single_weight_compensation_vector(l: &[f32], n: usize, j: usize) -> Vec<f32> {
    let column = h_inv_column(l, n, j);
    // H⁻¹ is symmetric PSD and full-rank (cholesky succeeded), so
    // diag(H⁻¹)[j] = column[j] > 0.
    let denom = column[j] as f64;
    assert!(
        denom > 0.0 && denom.is_finite(),
        "single_weight_compensation_vector: H⁻¹[{j}, {j}] = {denom} must be \
         strictly positive and finite",
    );
    let mut c = vec![0.0_f32; n];
    for k in 0..n {
        if k == j {
            c[k] = 0.0;
        } else {
            c[k] = (column[k] as f64 / denom) as f32;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::hessian::upper_tri_offset;
    use crate::forward::linalg::cholesky;

    #[test]
    fn compensation_on_identity_is_all_zero() {
        // H = I → H⁻¹ = I → column j is e_j → c[k != j] = 0 / 1 = 0,
        // c[j] = 0 by spec. So the whole vector is zero.
        let n = 4;
        let mut l = vec![0.0_f32; n * (n + 1) / 2];
        for i in 0..n {
            // L = I for H = I (diagonal sqrt of 1).
            l[i * (i + 1) / 2 + i] = 1.0;
        }
        for j in 0..n {
            let c = single_weight_compensation_vector(&l, n, j);
            for (k, &v) in c.iter().enumerate() {
                assert!(
                    v.abs() < 1e-6,
                    "c[{k}] = {v} should be 0 for identity H (j = {j})",
                );
            }
        }
    }

    #[test]
    fn compensation_on_2x2_matches_closed_form() {
        // H = [[4, 2], [2, 5]]
        // H⁻¹ = (1/16) [[5, -2], [-2, 4]]
        // For j = 0: c[0] = 0; c[1] = (-2/16) / (5/16) = -2/5 = -0.4
        // For j = 1: c[0] = (-2/16) / (4/16) = -0.5; c[1] = 0
        let h = [4.0_f32, 2.0, 5.0];
        let l = cholesky(&h, 2).unwrap();

        let c0 = single_weight_compensation_vector(&l, 2, 0);
        assert!((c0[0] - 0.0).abs() < 1e-5, "c0[0] = {}", c0[0]);
        assert!((c0[1] - (-0.4)).abs() < 1e-4, "c0[1] = {}", c0[1]);

        let c1 = single_weight_compensation_vector(&l, 2, 1);
        assert!((c1[0] - (-0.5)).abs() < 1e-4, "c1[0] = {}", c1[0]);
        assert!((c1[1] - 0.0).abs() < 1e-5, "c1[1] = {}", c1[1]);
    }

    #[test]
    fn compensation_zeroes_residual_at_non_forced_channels() {
        // Property: for Δ = ε·(e_j + c), H·Δ should be zero at every
        // channel except j. This is the whole point of OBS
        // compensation — it kills the gradient impact on free
        // channels.
        let n = 5;
        let v = [1.0_f32, -0.5, 2.0, 0.3, -1.2];
        let mut h = vec![0.0_f32; n * (n + 1) / 2];
        for i in 0..n {
            for j in i..n {
                let vv = v[i] * v[j];
                let identity = if i == j { 1.0 } else { 0.0 };
                h[upper_tri_offset(n, i, j)] = vv + identity;
            }
        }
        let l = cholesky(&h, n).unwrap();

        for j in 0..n {
            let c = single_weight_compensation_vector(&l, n, j);
            // Δ = e_j + c (ε factors out).
            let mut delta = c.clone();
            delta[j] = 1.0;
            // Compute H · Δ and check non-j entries are ~0, j-th is nonzero.
            for i in 0..n {
                let mut row = 0.0_f64;
                for k in 0..n {
                    let h_ik = if i <= k {
                        h[upper_tri_offset(n, i, k)]
                    } else {
                        h[upper_tri_offset(n, k, i)]
                    } as f64;
                    row += h_ik * delta[k] as f64;
                }
                if i == j {
                    assert!(
                        row.abs() > 1e-6,
                        "(H Δ)[{i}] = {row} should be nonzero at forced channel",
                    );
                } else {
                    assert!(
                        row.abs() < 1e-3,
                        "(H Δ)[{i}] = {row} should be ≈ 0 at free channel (j = {j})",
                    );
                }
            }
        }
    }

    #[test]
    fn compensation_j_component_is_always_zero() {
        // Sanity: c[j] must be identically 0 regardless of input.
        let n = 3;
        let mut h = vec![0.0_f32; n * (n + 1) / 2];
        h[upper_tri_offset(n, 0, 0)] = 9.0;
        h[upper_tri_offset(n, 0, 1)] = 1.0;
        h[upper_tri_offset(n, 1, 1)] = 4.0;
        h[upper_tri_offset(n, 1, 2)] = 2.0;
        h[upper_tri_offset(n, 2, 2)] = 5.0;
        let l = cholesky(&h, n).unwrap();
        for j in 0..n {
            let c = single_weight_compensation_vector(&l, n, j);
            assert_eq!(c[j], 0.0, "c[{j}] must be exactly 0");
        }
    }

    // ── compensation_operator tests ────────────────────────────────

    /// Build an SPD matrix as `v v^T + I` and return its upper triangle.
    fn spd_upper(v: &[f32]) -> Vec<f32> {
        let n = v.len();
        let mut h = vec![0.0_f32; n * (n + 1) / 2];
        for i in 0..n {
            for j in i..n {
                let vv = v[i] * v[j];
                let identity = if i == j { 1.0 } else { 0.0 };
                h[upper_tri_offset(n, i, j)] = vv + identity;
            }
        }
        h
    }

    fn h_entry(h: &[f32], n: usize, i: usize, j: usize) -> f64 {
        let (a, b) = if i <= j { (i, j) } else { (j, i) };
        h[upper_tri_offset(n, a, b)] as f64
    }

    #[test]
    fn compensation_operator_rejects_out_of_range_region() {
        let h = [4.0_f32, 2.0, 5.0];
        let l = cholesky(&h, 2).unwrap();
        let err = compensation_operator(&l, 2, &[5]).unwrap_err();
        assert!(matches!(
            err,
            CompensationError::RegionIndexOutOfRange { n: 2, index: 5 }
        ));
    }

    #[test]
    fn compensation_operator_rejects_duplicate_region_entries() {
        let h = [4.0_f32, 2.0, 5.0];
        let l = cholesky(&h, 2).unwrap();
        let err = compensation_operator(&l, 2, &[0, 0]).unwrap_err();
        assert!(matches!(
            err,
            CompensationError::DuplicateRegionIndex { index: 0 }
        ));
    }

    #[test]
    fn compensation_operator_empty_region_produces_empty_m() {
        let h = [4.0_f32, 2.0, 5.0];
        let l = cholesky(&h, 2).unwrap();
        let op = compensation_operator(&l, 2, &[]).unwrap();
        assert_eq!(op.region_size(), 0);
        assert_eq!(op.free_size(), 2);
        assert_eq!(op.c_indices, vec![0, 1]);
        assert!(op.m.is_empty());
    }

    #[test]
    fn compensation_operator_full_region_produces_empty_c() {
        let h = [4.0_f32, 2.0, 5.0];
        let l = cholesky(&h, 2).unwrap();
        let op = compensation_operator(&l, 2, &[0, 1]).unwrap();
        assert_eq!(op.region_size(), 2);
        assert_eq!(op.free_size(), 0);
        assert!(op.c_indices.is_empty());
        assert!(op.m.is_empty());
    }

    #[test]
    fn compensation_operator_sorts_region() {
        let n = 3;
        let h = spd_upper(&[1.0_f32, -0.5, 2.0]);
        let l = cholesky(&h, n).unwrap();
        // Pass unsorted region.
        let op = compensation_operator(&l, n, &[2, 0]).unwrap();
        assert_eq!(op.region, vec![0, 2]);
        assert_eq!(op.c_indices, vec![1]);
    }

    #[test]
    fn compensation_operator_r1_matches_single_weight_compensation_vector() {
        // For a single-channel region {j}, compensation_operator's
        // M_R column should equal single_weight_compensation_vector
        // restricted to the free-channel indices.
        let n = 4;
        let h = spd_upper(&[1.0_f32, -0.5, 2.0, 0.3]);
        let l = cholesky(&h, n).unwrap();
        for j in 0..n {
            let c_full = single_weight_compensation_vector(&l, n, j);
            let op = compensation_operator(&l, n, &[j]).unwrap();
            assert_eq!(op.region_size(), 1);
            assert_eq!(op.free_size(), n - 1);
            for (ci_idx, &c_chan) in op.c_indices.iter().enumerate() {
                let m_val = op.m[ci_idx]; // column-major, single column
                let expected = c_full[c_chan];
                assert!(
                    (m_val - expected).abs() < 1e-4,
                    "op.m[{ci_idx}] (c_chan={c_chan}) = {m_val}, expected {expected} \
                     from single_weight_compensation_vector[j={j}]",
                );
            }
        }
    }

    #[test]
    fn compensation_operator_r2_zeroes_residual_at_free_channels() {
        // The optimality condition: for any forced Δ_R, applying
        // Δ_C = M_R · Δ_R makes H · Δ = 0 at every c ∈ C.
        // Verify on a 4×4 SPD with |R| = 2, across several Δ_R.
        let n = 4;
        let h = spd_upper(&[1.0_f32, -0.5, 2.0, 0.3]);
        let l = cholesky(&h, n).unwrap();
        let region = vec![0_usize, 2];
        let op = compensation_operator(&l, n, &region).unwrap();
        assert_eq!(op.region_size(), 2);
        assert_eq!(op.free_size(), 2);
        assert_eq!(op.c_indices, vec![1, 3]);

        for delta_r in [
            vec![1.0_f32, 0.0],
            vec![0.0_f32, 1.0],
            vec![0.7_f32, -1.3],
            vec![-2.0_f32, 0.5],
        ] {
            let delta_c = op.apply(&delta_r);
            // Reconstruct full Δ: R indices get delta_r values,
            // C indices get delta_c values.
            let mut delta = vec![0.0_f32; n];
            for (i, &rj) in op.region.iter().enumerate() {
                delta[rj] = delta_r[i];
            }
            for (i, &cj) in op.c_indices.iter().enumerate() {
                delta[cj] = delta_c[i];
            }
            // Check (H Δ) at each free channel ≈ 0.
            for &c in &op.c_indices {
                let mut sum = 0.0_f64;
                for k in 0..n {
                    sum += h_entry(&h, n, c, k) * delta[k] as f64;
                }
                assert!(
                    sum.abs() < 1e-3,
                    "(H Δ)[{c}] = {sum} not ≈ 0 for delta_r = {delta_r:?}",
                );
            }
        }
    }

    #[test]
    fn compensation_operator_apply_panics_on_wrong_length() {
        let h = [4.0_f32, 2.0, 5.0];
        let l = cholesky(&h, 2).unwrap();
        let op = compensation_operator(&l, 2, &[0]).unwrap();
        // region_size == 1, delta_r must be length 1
        let result = std::panic::catch_unwind(|| {
            op.apply(&[1.0_f32, 2.0]);
        });
        assert!(result.is_err(), "apply should panic on wrong length");
    }
}
