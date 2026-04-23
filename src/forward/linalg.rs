//! Small linear-algebra primitives used by the OBS saliency pipeline
//! (Phase D1-a) and — later — Phase E's compensation math.
//!
//! Hand-rolled to keep LLMDB's external-deps surface small. The needs
//! are narrow: Cholesky factorization of symmetric positive-definite
//! matrices and a lower-triangular forward solve. No general linalg,
//! no matrix-matrix BLAS, no eigenvalue routines — D0's analyzer
//! confirmed our H matrices are well-behaved (zero negative
//! eigenvalues, zero symmetry error on 240 matrices), so the only
//! numerics we need to land here is the Cholesky path.
//!
//! ## Storage conventions
//!
//! Both the upper triangle of H and the lower triangle of L are
//! stored in **row-major packed form** with `n*(n+1)/2` entries.
//! - Upper triangle: row `i` covers columns `i..n`, with `n - i`
//!   entries starting from `(i, i)`. Offset formula lives in
//!   [`crate::forward::hessian::upper_tri_offset`].
//! - Lower triangle: row `i` covers columns `0..=i`, with `i + 1`
//!   entries starting at `i*(i+1)/2`. Offset for `(i, j)` with
//!   `j <= i` is `i*(i+1)/2 + j`.
//!
//! ## Numerics
//!
//! F32 throughout. Cholesky sums accumulate in f64 for the dot-products
//! to keep round-off under control on larger matrices; the factor
//! itself is stored f32 since D0 measured F16 is already sufficient
//! for the compensation math, so f32 is comfortably inside tolerance.

use thiserror::Error;

use crate::forward::hessian::upper_tri_offset;

#[derive(Debug, Error)]
pub enum CholeskyError {
    #[error(
        "input upper-triangle length {got} does not match n*(n+1)/2 = {expected} for n={n}"
    )]
    LengthMismatch { n: usize, expected: usize, got: usize },

    #[error(
        "matrix not positive definite: diagonal at row {row} computed as sqrt({diag_sq}), \
         expected > 0"
    )]
    NotPositiveDefinite { row: usize, diag_sq: f64 },
}

/// Offset of `L[i][j]` (for `j <= i`) in lower-triangle row-major
/// packed storage.
#[inline]
pub fn lower_tri_offset(i: usize, j: usize) -> usize {
    debug_assert!(j <= i, "lower-triangle offset requires j <= i");
    i * (i + 1) / 2 + j
}

/// Fetch `H[i][j]` from upper-triangle packed storage. Since H is
/// symmetric, this handles both `i <= j` (direct lookup) and `i > j`
/// (mirrors to `H[j][i]`).
#[inline]
fn h_at(h_upper: &[f32], n: usize, i: usize, j: usize) -> f32 {
    if i <= j {
        h_upper[upper_tri_offset(n, i, j)]
    } else {
        h_upper[upper_tri_offset(n, j, i)]
    }
}

/// Cholesky factor `L` of a symmetric positive-definite matrix `H`
/// such that `H = L * L^T`. Input: upper triangle of H, row-major
/// packed. Output: lower triangle of L, row-major packed. Both have
/// length `n*(n+1)/2`.
///
/// Cholesky–Banachiewicz, row-by-row:
///
/// ```text
/// for i in 0..n:
///   for j in 0..i:
///     L[i][j] = (H[i][j] - Σ_{k<j} L[i][k]*L[j][k]) / L[j][j]
///   L[i][i] = sqrt(H[i][i] - Σ_{k<i} L[i][k]^2)
/// ```
///
/// Returns `NotPositiveDefinite` if the value under the square root
/// is non-positive at any row.
pub fn cholesky(h_upper: &[f32], n: usize) -> Result<Vec<f32>, CholeskyError> {
    let expected = n * (n + 1) / 2;
    if h_upper.len() != expected {
        return Err(CholeskyError::LengthMismatch {
            n,
            expected,
            got: h_upper.len(),
        });
    }

    let mut l = vec![0.0_f32; expected];

    for i in 0..n {
        // Off-diagonal entries L[i][j] for j < i.
        for j in 0..i {
            let mut sum = 0.0_f64;
            for k in 0..j {
                sum += l[lower_tri_offset(i, k)] as f64 * l[lower_tri_offset(j, k)] as f64;
            }
            let numerator = h_at(h_upper, n, i, j) as f64 - sum;
            let denom = l[lower_tri_offset(j, j)] as f64;
            // denom was set in a previous iteration; it's the
            // square root of something we already validated > 0.
            debug_assert!(denom > 0.0, "denominator L[{j}][{j}] = {denom} must be > 0");
            l[lower_tri_offset(i, j)] = (numerator / denom) as f32;
        }

        // Diagonal entry L[i][i].
        let mut sum = 0.0_f64;
        for k in 0..i {
            let v = l[lower_tri_offset(i, k)] as f64;
            sum += v * v;
        }
        let diag_sq = h_at(h_upper, n, i, i) as f64 - sum;
        if diag_sq <= 0.0 {
            return Err(CholeskyError::NotPositiveDefinite { row: i, diag_sq });
        }
        l[lower_tri_offset(i, i)] = (diag_sq.sqrt()) as f32;
    }

    Ok(l)
}

/// Per-channel OBS saliency `1 / diag(H⁻¹)[j]` for each `j`, given
/// the Cholesky factor `L` of `H`. Orientation matches AWQ's
/// convention: **higher values = more protected / avoid during
/// stego placement** (allocator's compound `FitKey` prefers low
/// salience).
///
/// Math: `diag(H⁻¹)[j] = ‖L⁻¹ e_j‖² = ‖y_j‖²` where `L y_j = e_j`.
/// So for each `j`: forward-solve `L y = e_j`, compute `‖y‖²`,
/// emit `1 / ‖y‖²`.
///
/// Complexity: `O(n³ / 6)` overall — forward-substitute `n` unit
/// vectors, each solve costs `O((n - j)² / 2)` thanks to the
/// leading zeros of `e_j`.
pub fn obs_saliency(l: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n];
    let mut y = vec![0.0_f32; n];
    for j in 0..n {
        for slot in y.iter_mut() {
            *slot = 0.0;
        }
        y[j] = 1.0;
        solve_lower(l, n, &mut y);
        // ‖y‖² with f64 accumulation; y[i] = 0 for i < j by
        // construction, so start the sum at j.
        let mut norm_sq = 0.0_f64;
        for v in y.iter().skip(j) {
            norm_sq += (*v as f64) * (*v as f64);
        }
        // For a valid Cholesky factor, L[j][j] > 0 guarantees
        // y[j] = 1 / L[j][j] > 0, so norm_sq is strictly positive.
        // Any non-finite or non-positive value here indicates a bug
        // (or pathological input); assert rather than silently
        // returning 0 so callers see it.
        assert!(
            norm_sq.is_finite() && norm_sq > 0.0,
            "obs_saliency: ‖L⁻¹ e_{j}‖² = {norm_sq} is not strictly positive \
             (expected L[j][j] > 0 from cholesky invariant)"
        );
        out[j] = (1.0 / norm_sq) as f32;
    }
    out
}

/// Solve `L * x = b` in place, where `L` is the lower-triangle packed
/// factor produced by [`cholesky`]. Replaces `x` (initially `b`) with
/// the solution.
///
/// Forward substitution:
///
/// ```text
/// x[0] = b[0] / L[0][0]
/// x[i] = (b[i] - Σ_{k<i} L[i][k] * x[k]) / L[i][i]   for i >= 1
/// ```
///
/// Panics on length mismatch. f64 accumulator in the inner sum to
/// keep precision on larger matrices.
pub fn solve_lower(l: &[f32], n: usize, x: &mut [f32]) {
    assert_eq!(x.len(), n, "solve_lower: x length {} != n {}", x.len(), n);
    assert_eq!(
        l.len(),
        n * (n + 1) / 2,
        "solve_lower: L length {} != n*(n+1)/2 = {}",
        l.len(),
        n * (n + 1) / 2,
    );

    for i in 0..n {
        let mut sum = 0.0_f64;
        for k in 0..i {
            sum += l[lower_tri_offset(i, k)] as f64 * x[k] as f64;
        }
        let diag = l[lower_tri_offset(i, i)] as f64;
        x[i] = ((x[i] as f64 - sum) / diag) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reconstruct the full N×N matrix from an upper-triangle packed
    /// slice, just so tests can compare against hand-computed dense
    /// references without re-deriving the offset math.
    fn expand_upper(h_upper: &[f32], n: usize) -> Vec<Vec<f32>> {
        let mut out = vec![vec![0.0_f32; n]; n];
        for i in 0..n {
            for j in i..n {
                let v = h_upper[upper_tri_offset(n, i, j)];
                out[i][j] = v;
                out[j][i] = v;
            }
        }
        out
    }

    /// Reconstruct the full N×N L from a lower-triangle packed slice.
    fn expand_lower(l: &[f32], n: usize) -> Vec<Vec<f32>> {
        let mut out = vec![vec![0.0_f32; n]; n];
        for i in 0..n {
            for j in 0..=i {
                out[i][j] = l[lower_tri_offset(i, j)];
            }
        }
        out
    }

    /// Multiply L * L^T and check it matches H to tolerance.
    fn verify_ll_t(l_full: &[Vec<f32>], h_full: &[Vec<f32>], n: usize, tol: f32) {
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0_f64;
                for k in 0..n {
                    sum += l_full[i][k] as f64 * l_full[j][k] as f64;
                }
                let got = sum as f32;
                let want = h_full[i][j];
                assert!(
                    (got - want).abs() < tol,
                    "(L*L^T)[{i}][{j}] = {got}, expected {want}",
                );
            }
        }
    }

    #[test]
    fn lower_offset_matches_packed_layout() {
        // For n=4, lower triangle flat layout is:
        //   row 0: (0,0)                  → offset 0
        //   row 1: (1,0) (1,1)            → offsets 1..=2
        //   row 2: (2,0) (2,1) (2,2)      → offsets 3..=5
        //   row 3: (3,0) (3,1) (3,2) (3,3)→ offsets 6..=9
        assert_eq!(lower_tri_offset(0, 0), 0);
        assert_eq!(lower_tri_offset(1, 0), 1);
        assert_eq!(lower_tri_offset(1, 1), 2);
        assert_eq!(lower_tri_offset(2, 0), 3);
        assert_eq!(lower_tri_offset(2, 2), 5);
        assert_eq!(lower_tri_offset(3, 0), 6);
        assert_eq!(lower_tri_offset(3, 3), 9);
    }

    #[test]
    fn cholesky_1x1_returns_sqrt() {
        // H = [4] → L = [2].
        let h = [4.0_f32];
        let l = cholesky(&h, 1).unwrap();
        assert_eq!(l, vec![2.0_f32]);
    }

    #[test]
    fn cholesky_2x2_known_case() {
        // H = [[4, 2], [2, 5]]
        //   upper triangle packed: [H00, H01, H11] = [4, 2, 5]
        // Expected L = [[2, 0], [1, 2]]
        //   lower triangle packed: [L00, L10, L11] = [2, 1, 2]
        let h = [4.0_f32, 2.0, 5.0];
        let l = cholesky(&h, 2).unwrap();
        assert_eq!(l, vec![2.0_f32, 1.0, 2.0]);
    }

    #[test]
    fn cholesky_identity_round_trip() {
        let n = 4;
        let mut h = vec![0.0_f32; n * (n + 1) / 2];
        for i in 0..n {
            h[upper_tri_offset(n, i, i)] = 1.0;
        }
        let l = cholesky(&h, n).unwrap();
        let l_full = expand_lower(&l, n);
        let h_full = expand_upper(&h, n);
        verify_ll_t(&l_full, &h_full, n, 1e-6);
    }

    #[test]
    fn cholesky_round_trip_on_outer_product_spd_matrix() {
        // Build H = v v^T + I for a known v — guaranteed SPD.
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
        let l_full = expand_lower(&l, n);
        let h_full = expand_upper(&h, n);
        verify_ll_t(&l_full, &h_full, n, 1e-4);
    }

    #[test]
    fn cholesky_rejects_negative_diagonal() {
        // H = [[-1]] — not PSD.
        let h = [-1.0_f32];
        let err = cholesky(&h, 1).unwrap_err();
        assert!(matches!(
            err,
            CholeskyError::NotPositiveDefinite { row: 0, .. }
        ));
    }

    #[test]
    fn cholesky_rejects_indefinite_matrix() {
        // H = [[1, 2], [2, 1]] — symmetric but indefinite (eigenvalues 3 and -1).
        // First row factors fine (L[0][0] = 1); row 1 diagonal would be
        // H[1][1] - L[1][0]^2 = 1 - 4 = -3, which should trip the guard.
        let h = [1.0_f32, 2.0, 1.0];
        let err = cholesky(&h, 2).unwrap_err();
        assert!(matches!(
            err,
            CholeskyError::NotPositiveDefinite { row: 1, .. }
        ));
    }

    #[test]
    fn cholesky_rejects_length_mismatch() {
        // n=3 expects 6 entries; give 5.
        let h = [1.0_f32, 0.0, 0.0, 1.0, 0.0];
        let err = cholesky(&h, 3).unwrap_err();
        assert!(matches!(
            err,
            CholeskyError::LengthMismatch {
                n: 3,
                expected: 6,
                got: 5
            }
        ));
    }

    #[test]
    fn solve_lower_identity_passes_b_through() {
        let n = 4;
        let mut l = vec![0.0_f32; n * (n + 1) / 2];
        for i in 0..n {
            l[lower_tri_offset(i, i)] = 1.0;
        }
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0];
        solve_lower(&l, n, &mut x);
        assert_eq!(x, vec![1.0_f32, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn solve_lower_round_trip_against_cholesky_factor() {
        // Round-trip: given a known SPD matrix H and vector b, solve
        // L*y = b. Then verify L*y == b to tolerance.
        let n = 4;
        let v = [1.0_f32, -0.5, 2.0, 0.3];
        let mut h = vec![0.0_f32; n * (n + 1) / 2];
        for i in 0..n {
            for j in i..n {
                let vv = v[i] * v[j];
                let identity = if i == j { 1.0 } else { 0.0 };
                h[upper_tri_offset(n, i, j)] = vv + identity;
            }
        }
        let l = cholesky(&h, n).unwrap();
        let b = vec![1.0_f32, 2.0, -1.5, 0.7];
        let mut y = b.clone();
        solve_lower(&l, n, &mut y);
        // Reconstruct L*y and compare to b.
        let l_full = expand_lower(&l, n);
        for i in 0..n {
            let mut sum = 0.0_f64;
            for k in 0..=i {
                sum += l_full[i][k] as f64 * y[k] as f64;
            }
            let got = sum as f32;
            assert!(
                (got - b[i]).abs() < 1e-5,
                "(L*y)[{i}] = {got}, expected {}",
                b[i]
            );
        }
    }

    #[test]
    fn solve_lower_solves_unit_basis_vectors() {
        // Solving L*y = e_j is the forward-sub step that OBS uses.
        // For L = [[2, 0], [1, 2]]:
        //   L y = e_0 = [1, 0] → y = [0.5, -0.25]
        //   L y = e_1 = [0, 1] → y = [0, 0.5]
        let l = vec![2.0_f32, 1.0, 2.0];
        let n = 2;
        let mut y0 = vec![1.0_f32, 0.0];
        solve_lower(&l, n, &mut y0);
        assert!((y0[0] - 0.5).abs() < 1e-6);
        assert!((y0[1] - (-0.25)).abs() < 1e-6);
        let mut y1 = vec![0.0_f32, 1.0];
        solve_lower(&l, n, &mut y1);
        assert!((y1[0] - 0.0).abs() < 1e-6);
        assert!((y1[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "solve_lower: x length")]
    fn solve_lower_panics_on_length_mismatch() {
        let l = vec![1.0_f32, 0.0, 1.0];
        let mut x = vec![1.0_f32]; // wrong length for n=2
        solve_lower(&l, 2, &mut x);
    }

    #[test]
    fn obs_saliency_on_identity_returns_all_ones() {
        // H = I → L = I → H⁻¹ = I → diag(H⁻¹) = [1,1,…] → OBS = [1,1,…]
        let n = 5;
        let mut l = vec![0.0_f32; n * (n + 1) / 2];
        for i in 0..n {
            l[lower_tri_offset(i, i)] = 1.0;
        }
        let obs = obs_saliency(&l, n);
        for (i, v) in obs.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "OBS[{i}] = {v}, expected 1.0 for identity H",
            );
        }
    }

    #[test]
    fn obs_saliency_on_diagonal_matches_closed_form() {
        // H = diag(4, 9, 16) → L = diag(2, 3, 4)
        //   → H⁻¹ = diag(1/4, 1/9, 1/16)
        //   → OBS = [4, 9, 16].
        let n = 3;
        let mut l = vec![0.0_f32; n * (n + 1) / 2];
        l[lower_tri_offset(0, 0)] = 2.0;
        l[lower_tri_offset(1, 1)] = 3.0;
        l[lower_tri_offset(2, 2)] = 4.0;
        let obs = obs_saliency(&l, n);
        assert!((obs[0] - 4.0).abs() < 1e-5, "OBS[0] = {}, expected 4.0", obs[0]);
        assert!((obs[1] - 9.0).abs() < 1e-5, "OBS[1] = {}, expected 9.0", obs[1]);
        assert!((obs[2] - 16.0).abs() < 1e-5, "OBS[2] = {}, expected 16.0", obs[2]);
    }

    #[test]
    fn obs_saliency_on_known_2x2() {
        // H = [[4, 2], [2, 5]] → L = [[2, 0], [1, 2]]
        // det(H) = 4*5 - 2*2 = 16
        // H⁻¹ = (1/16) * [[5, -2], [-2, 4]]
        // diag(H⁻¹) = [5/16, 4/16] = [0.3125, 0.25]
        // OBS = [1/0.3125, 1/0.25] = [3.2, 4.0]
        let h = [4.0_f32, 2.0, 5.0];
        let l = cholesky(&h, 2).unwrap();
        let obs = obs_saliency(&l, 2);
        assert!((obs[0] - 3.2).abs() < 1e-4, "OBS[0] = {}, expected 3.2", obs[0]);
        assert!((obs[1] - 4.0).abs() < 1e-4, "OBS[1] = {}, expected 4.0", obs[1]);
    }

    #[test]
    fn obs_saliency_is_strictly_positive_for_real_psd_input() {
        // Same v v^T + I construction as the round-trip tests — SPD.
        // Every OBS entry should be finite and > 0.
        let n = 6;
        let v = [1.0_f32, -0.5, 2.0, 0.3, -1.2, 0.8];
        let mut h = vec![0.0_f32; n * (n + 1) / 2];
        for i in 0..n {
            for j in i..n {
                let vv = v[i] * v[j];
                let identity = if i == j { 1.0 } else { 0.0 };
                h[upper_tri_offset(n, i, j)] = vv + identity;
            }
        }
        let l = cholesky(&h, n).unwrap();
        let obs = obs_saliency(&l, n);
        for (i, v) in obs.iter().enumerate() {
            assert!(v.is_finite(), "OBS[{i}] = {v} not finite");
            assert!(*v > 0.0, "OBS[{i}] = {v} not positive");
        }
    }

    #[test]
    fn obs_saliency_produces_different_values_on_non_trivial_matrix() {
        // Guard against a degenerate implementation that returns the
        // same value for every channel. For a non-diagonal SPD matrix
        // the OBS values should vary across channels.
        let h = [4.0_f32, 2.0, 5.0]; // same 2x2 as obs_saliency_on_known_2x2
        let l = cholesky(&h, 2).unwrap();
        let obs = obs_saliency(&l, 2);
        assert!(
            (obs[0] - obs[1]).abs() > 0.1,
            "OBS should differ across channels; got {obs:?}"
        );
    }
}
