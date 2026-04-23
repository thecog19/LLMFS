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

use crate::forward::linalg::h_inv_column;

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
}
