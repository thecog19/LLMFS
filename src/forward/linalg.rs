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

#[derive(Debug, Error)]
pub enum PivotedCholeskyError {
    #[error(
        "input upper-triangle length {got} does not match n*(n+1)/2 = {expected} for n={n}"
    )]
    LengthMismatch { n: usize, expected: usize, got: usize },

    #[error(
        "tolerance must be in [0, 1] (got {tolerance}) — interpreted as a \
         fraction of trace(H) for the residual-diagonal stopping criterion"
    )]
    ToleranceOutOfRange { tolerance: f32 },
}

/// Output of [`pivoted_cholesky`]: a rank-`K` approximation of the
/// input matrix `H` such that `V * V^T ≈ H` with residual bounded
/// by the tolerance passed to the factorization.
#[derive(Debug, Clone)]
pub struct PivotedCholesky {
    /// Input matrix dimension.
    pub n: usize,
    /// Order in which pivots were selected (one per column of `V`,
    /// in extraction order). Length equals the output's rank `K`.
    pub pivots: Vec<usize>,
    /// Column-major flat storage of `V` (N rows × K columns):
    /// `v[k * n + i]` is `V[i][k]`. Length `n * pivots.len()`.
    pub v: Vec<f32>,
}

impl PivotedCholesky {
    /// Rank `K` of this low-rank factor — i.e., number of columns in `V`.
    pub fn rank(&self) -> usize {
        self.pivots.len()
    }

    /// Whether the factorization produced any columns.
    pub fn is_empty(&self) -> bool {
        self.pivots.is_empty()
    }
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

/// Pivoted Cholesky factorization with adaptive rank (Harbrecht,
/// Peters, Schneider 2012). Produces an `N × K` matrix `V` such that
/// `V * V^T ≈ H`, with `K` chosen adaptively by tolerance or capped
/// at `max_rank`, whichever happens first.
///
/// This is the low-rank counterpart to [`cholesky`], motivated by
/// `docs/compensation-design.md §2.2` and backed by the D0
/// measurements in `docs/phase-d-measurement.md` (per-site effective
/// rank 3–17% of N at 5% Frobenius error on SmolLM2). Phase E's
/// compensation math can work against `V` directly via
/// `V (V^T V)⁻¹ V^T` — O(K²) extra work compared to full `L` solves
/// but with O(K·N) storage instead of O(N²).
///
/// **Algorithm outline.** At each step `k`:
///
/// 1. Select pivot `p_k = argmax_i diag(R_k)[i]` where `R_k` is the
///    current residual `H − V_{<k} V_{<k}^T`. (Diagonal residual
///    entries are maintained incrementally — no explicit residual
///    matrix construction.)
/// 2. If `R_k[p_k][p_k] < tolerance · trace(H)`, stop. This bounds
///    the dropped diagonal contribution (and so, bounds the trace
///    of the residual) to at most `tolerance · trace(H)`.
/// 3. Compute column `v_k` of length `N`:
///    - `v_k[p_k] = √R_k[p_k][p_k]`,
///    - `v_k[i] = R_k[i][p_k] / v_k[p_k]` for `i ≠ p_k`,
///    where `R_k[i][p_k] = H[i][p_k] − Σ_{j<k} v_j[i] · v_j[p_k]`.
/// 4. Update residual diagonal: `d_i ← d_i − v_k[i]²`; force
///    `d_{p_k} ← 0` to mark that position exhausted.
///
/// Cost: `O(K · N · K)` total (each iteration's inner sum is `O(K)`
/// and runs for `N` positions). For the regimes we care about
/// (`K ≪ N`) this is substantially cheaper than the `O(N³/3)` of
/// full Cholesky.
///
/// # Arguments
///
/// - `h_upper`: upper triangle of `H`, row-major packed; same layout
///   as [`cholesky`]'s input.
/// - `n`: matrix dimension.
/// - `tolerance`: must be in `[0, 1]`. Fraction of `trace(H)` below
///   which to stop extracting new columns. `0` means "keep going to
///   `max_rank` regardless" (modulo exhausting positive residual).
/// - `max_rank`: hard cap on number of columns. `N` recovers the
///   full factorization (up to pivoting).
pub fn pivoted_cholesky(
    h_upper: &[f32],
    n: usize,
    tolerance: f32,
    max_rank: usize,
) -> Result<PivotedCholesky, PivotedCholeskyError> {
    let expected = n * (n + 1) / 2;
    if h_upper.len() != expected {
        return Err(PivotedCholeskyError::LengthMismatch {
            n,
            expected,
            got: h_upper.len(),
        });
    }
    if !(0.0..=1.0).contains(&tolerance) || tolerance.is_nan() {
        return Err(PivotedCholeskyError::ToleranceOutOfRange { tolerance });
    }
    if n == 0 || max_rank == 0 {
        return Ok(PivotedCholesky {
            n,
            pivots: Vec::new(),
            v: Vec::new(),
        });
    }

    // Residual diagonal. Starts as diag(H); decreases as we peel
    // off rank-1 updates.
    let mut diagonal: Vec<f64> = (0..n)
        .map(|i| h_at(h_upper, n, i, i) as f64)
        .collect();
    let initial_trace: f64 = diagonal.iter().copied().sum();
    let stop_threshold = initial_trace * tolerance as f64;

    let k_max = max_rank.min(n);
    let mut v: Vec<f32> = Vec::with_capacity(k_max * n);
    let mut pivots: Vec<usize> = Vec::with_capacity(k_max);

    for k in 0..k_max {
        // Pivot: index of max residual diagonal.
        let (p, &d_p) = match diagonal
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            Some(x) => x,
            None => break, // unreachable while n > 0, but guard anyway
        };

        // Stop on exhaustion (<= 0) or tolerance.
        if d_p <= 0.0 || d_p <= stop_threshold {
            break;
        }

        let diag_sqrt = d_p.sqrt();

        // Reserve the column. Write pivot entry first (sqrt), then
        // fill i ≠ p from H[i][p] minus prior-column contributions.
        let col_start = v.len();
        v.resize(col_start + n, 0.0);
        v[col_start + p] = diag_sqrt as f32;

        for i in 0..n {
            if i == p {
                continue;
            }
            // r_ip = H[i][p] - Σ_{j<k} v_j[i] · v_j[p].
            let mut r_ip = h_at(h_upper, n, i, p) as f64;
            for j in 0..k {
                let vji = v[j * n + i] as f64;
                let vjp = v[j * n + p] as f64;
                r_ip -= vji * vjp;
            }
            v[col_start + i] = (r_ip / diag_sqrt) as f32;
        }

        // Update the residual diagonal. For every i, subtract v_k[i]²;
        // force diagonal[p] to zero to mark that pivot as exhausted
        // (floating-point subtraction could leave a tiny negative
        // residual otherwise, which would confuse the next argmax).
        for i in 0..n {
            let v_ki = v[col_start + i] as f64;
            diagonal[i] -= v_ki * v_ki;
        }
        diagonal[p] = 0.0;

        pivots.push(p);
    }

    Ok(PivotedCholesky { n, pivots, v })
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

/// Solve `L^T * x = b` in place, where `L` is the lower-triangle
/// packed factor produced by [`cholesky`]. Replaces `x` (initially
/// `b`) with the solution.
///
/// `L^T` is upper-triangular, so this is back-substitution. Walking
/// the packed lower triangle, `L[k][i]` (for `k >= i`) sits at
/// offset `lower_tri_offset(k, i)`:
///
/// ```text
/// x[n-1] = b[n-1] / L[n-1][n-1]
/// x[i] = (b[i] - Σ_{k>i} L[k][i] * x[k]) / L[i][i]   for i < n-1
/// ```
///
/// Paired with [`solve_lower`] to solve `H * x = b` in two steps
/// against the Cholesky factor: forward-solve `L y = b`, then
/// back-solve `L^T x = y`.
pub fn solve_lower_transposed(l: &[f32], n: usize, x: &mut [f32]) {
    assert_eq!(
        x.len(),
        n,
        "solve_lower_transposed: x length {} != n {}",
        x.len(),
        n,
    );
    assert_eq!(
        l.len(),
        n * (n + 1) / 2,
        "solve_lower_transposed: L length {} != n*(n+1)/2 = {}",
        l.len(),
        n * (n + 1) / 2,
    );

    // Iterate i from n-1 down to 0. Using a range with rev() avoids
    // the signed-index gymnastics.
    for i in (0..n).rev() {
        let mut sum = 0.0_f64;
        for k in (i + 1)..n {
            sum += l[lower_tri_offset(k, i)] as f64 * x[k] as f64;
        }
        let diag = l[lower_tri_offset(i, i)] as f64;
        x[i] = ((x[i] as f64 - sum) / diag) as f32;
    }
}

/// Column `j` of `H⁻¹`, given the lower-triangle packed Cholesky
/// factor `L` of `H`. Uses the identity `H⁻¹ e_j = L⁻ᵀ L⁻¹ e_j`:
/// forward-solve `L y = e_j`, then back-solve `L^T x = y`. Result
/// is an N-vector.
///
/// `O(n²)` total. Core primitive under
/// [`obs_saliency`] (which needs only `‖y‖² = (H⁻¹)_{jj}`) and
/// under Phase E's compensation-operator extraction (which needs
/// the full column).
pub fn h_inv_column(l: &[f32], n: usize, j: usize) -> Vec<f32> {
    assert!(j < n, "h_inv_column: j ({j}) out of range for n = {n}");
    assert_eq!(
        l.len(),
        n * (n + 1) / 2,
        "h_inv_column: L length {} != n*(n+1)/2 = {}",
        l.len(),
        n * (n + 1) / 2,
    );
    let mut x = vec![0.0_f32; n];
    x[j] = 1.0;
    solve_lower(l, n, &mut x);
    solve_lower_transposed(l, n, &mut x);
    x
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

    // ── pivoted_cholesky tests ──────────────────────────────────────

    /// Compute `(V * V^T)[i][j]` from column-major packed V (length
    /// `n * rank`). Test helper; zero production use.
    fn vvt_entry(v: &[f32], n: usize, rank: usize, i: usize, j: usize) -> f64 {
        let mut sum = 0.0_f64;
        for k in 0..rank {
            sum += v[k * n + i] as f64 * v[k * n + j] as f64;
        }
        sum
    }

    #[test]
    fn pivoted_cholesky_rejects_length_mismatch() {
        // n=3 expects 6 entries; give 4.
        let h = [1.0_f32, 0.0, 0.0, 1.0];
        let err = pivoted_cholesky(&h, 3, 0.0, 3).unwrap_err();
        assert!(matches!(
            err,
            PivotedCholeskyError::LengthMismatch {
                n: 3,
                expected: 6,
                got: 4
            }
        ));
    }

    #[test]
    fn pivoted_cholesky_rejects_out_of_range_tolerance() {
        let h = [1.0_f32];
        let err = pivoted_cholesky(&h, 1, -0.1, 1).unwrap_err();
        assert!(matches!(
            err,
            PivotedCholeskyError::ToleranceOutOfRange { .. }
        ));
        let err = pivoted_cholesky(&h, 1, 1.5, 1).unwrap_err();
        assert!(matches!(
            err,
            PivotedCholeskyError::ToleranceOutOfRange { .. }
        ));
    }

    #[test]
    fn pivoted_cholesky_n_zero_returns_empty() {
        let h: [f32; 0] = [];
        let result = pivoted_cholesky(&h, 0, 0.0, 5).unwrap();
        assert_eq!(result.n, 0);
        assert!(result.pivots.is_empty());
        assert!(result.v.is_empty());
        assert_eq!(result.rank(), 0);
        assert!(result.is_empty());
    }

    #[test]
    fn pivoted_cholesky_max_rank_zero_returns_empty() {
        let h = [4.0_f32, 2.0, 5.0];
        let result = pivoted_cholesky(&h, 2, 0.0, 0).unwrap();
        assert_eq!(result.rank(), 0);
    }

    #[test]
    fn pivoted_cholesky_identity_reaches_full_rank() {
        // H = I_N → V*V^T must exactly equal I_N given enough rank.
        let n = 5;
        let mut h = vec![0.0_f32; n * (n + 1) / 2];
        for i in 0..n {
            h[upper_tri_offset(n, i, i)] = 1.0;
        }
        let result = pivoted_cholesky(&h, n, 0.0, n).unwrap();
        assert_eq!(result.rank(), n);
        for i in 0..n {
            for j in 0..n {
                let got = vvt_entry(&result.v, n, result.rank(), i, j);
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (got - want).abs() < 1e-6,
                    "(V V^T)[{i}][{j}] = {got}, expected {want}",
                );
            }
        }
    }

    #[test]
    fn pivoted_cholesky_recovers_rank_k_spd_matrix_exactly() {
        // Build H = U U^T where U is N×K, K < N. H has exact rank K,
        // so pivoted Cholesky with max_rank=N should stop at rank K
        // (residual goes to zero once we've captured all K directions)
        // and V V^T should equal H to machine tolerance.
        let n = 6;
        let k_rank = 3;
        // U columns: three non-trivial vectors, selected so their
        // outer products aren't degenerate.
        let u: Vec<Vec<f32>> = vec![
            vec![1.0, 0.5, -1.0, 0.3, 2.0, -0.7],
            vec![0.2, 1.0, 0.5, -0.8, 0.1, 1.3],
            vec![-0.5, 0.3, 1.1, 0.9, -1.2, 0.4],
        ];
        // Build H upper triangle: H[i][j] = Σ_k U[k][i] · U[k][j] for i ≤ j.
        let mut h = vec![0.0_f32; n * (n + 1) / 2];
        for i in 0..n {
            for j in i..n {
                let mut sum = 0.0_f64;
                for col in &u {
                    sum += col[i] as f64 * col[j] as f64;
                }
                h[upper_tri_offset(n, i, j)] = sum as f32;
            }
        }
        let result = pivoted_cholesky(&h, n, 1e-7, n).unwrap();
        assert_eq!(
            result.rank(),
            k_rank,
            "rank-{k_rank} input should produce rank-{k_rank} factor, got {}",
            result.rank(),
        );
        // V V^T must match H to tolerance.
        for i in 0..n {
            for j in 0..n {
                let got = vvt_entry(&result.v, n, result.rank(), i, j);
                let h_ij = if i <= j {
                    h[upper_tri_offset(n, i, j)]
                } else {
                    h[upper_tri_offset(n, j, i)]
                } as f64;
                assert!(
                    (got - h_ij).abs() < 1e-4,
                    "(V V^T)[{i}][{j}] = {got}, H[{i}][{j}] = {h_ij}",
                );
            }
        }
    }

    #[test]
    fn pivoted_cholesky_tolerance_bounds_residual_trace() {
        // For a non-trivial SPD matrix, the residual trace after
        // early-stopping at `tolerance · trace(H)` must satisfy:
        //   Σ_i residual_diag[i] ≤ tolerance · trace(H)
        // which follows from the stopping criterion being applied to
        // the max diagonal entry; the sum is bounded by n × max which
        // is looser, but in practice residual is often near max at stop.
        let n = 5;
        let v_seed: Vec<f32> = vec![1.0, -0.5, 2.0, 0.3, -1.2];
        let mut h = vec![0.0_f32; n * (n + 1) / 2];
        for i in 0..n {
            for j in i..n {
                let vv = v_seed[i] * v_seed[j];
                let identity = if i == j { 1.0 } else { 0.0 };
                h[upper_tri_offset(n, i, j)] = vv + identity;
            }
        }
        let initial_trace: f64 = (0..n)
            .map(|i| h[upper_tri_offset(n, i, i)] as f64)
            .sum();
        let tolerance = 0.5_f32;
        let result = pivoted_cholesky(&h, n, tolerance, n).unwrap();
        // Recompute diagonal of residual H - V V^T.
        let residual_trace: f64 = (0..n)
            .map(|i| {
                h[upper_tri_offset(n, i, i)] as f64
                    - vvt_entry(&result.v, n, result.rank(), i, i)
            })
            .sum();
        assert!(
            residual_trace >= -1e-6,
            "residual trace went negative ({residual_trace}) — PSD violation"
        );
        // Post-stop max diagonal ≤ tolerance · initial_trace by
        // construction; residual trace ≤ n · max ≤ n · tol · init.
        // For this test we just verify the bound isn't wildly off.
        let max_residual_diag = (0..n)
            .map(|i| {
                h[upper_tri_offset(n, i, i)] as f64
                    - vvt_entry(&result.v, n, result.rank(), i, i)
            })
            .fold(0.0_f64, f64::max);
        assert!(
            max_residual_diag <= tolerance as f64 * initial_trace + 1e-5,
            "max residual diag {max_residual_diag} exceeds tolerance × trace \
             ({} × {} = {})",
            tolerance,
            initial_trace,
            tolerance as f64 * initial_trace,
        );
    }

    #[test]
    fn pivoted_cholesky_max_rank_caps_output() {
        // Full-rank SPD input, but request at most 2 columns.
        let n = 5;
        let v_seed: Vec<f32> = vec![1.0, -0.5, 2.0, 0.3, -1.2];
        let mut h = vec![0.0_f32; n * (n + 1) / 2];
        for i in 0..n {
            for j in i..n {
                let vv = v_seed[i] * v_seed[j];
                let identity = if i == j { 1.0 } else { 0.0 };
                h[upper_tri_offset(n, i, j)] = vv + identity;
            }
        }
        let result = pivoted_cholesky(&h, n, 0.0, 2).unwrap();
        assert_eq!(result.rank(), 2);
        assert_eq!(result.pivots.len(), 2);
        assert_eq!(result.v.len(), 2 * n);
    }

    #[test]
    fn pivoted_cholesky_pivot_order_picks_largest_diagonals_first() {
        // For H = diag(9, 1, 4), pivoted Cholesky should pick
        // position 0 (largest diagonal), then position 2 (next), then 1.
        let n = 3;
        let mut h = vec![0.0_f32; n * (n + 1) / 2];
        h[upper_tri_offset(n, 0, 0)] = 9.0;
        h[upper_tri_offset(n, 1, 1)] = 1.0;
        h[upper_tri_offset(n, 2, 2)] = 4.0;
        let result = pivoted_cholesky(&h, n, 0.0, n).unwrap();
        assert_eq!(result.pivots, vec![0, 2, 1]);
    }

    #[test]
    fn pivoted_cholesky_stops_on_zero_diagonal() {
        // H = [[0]] has a zero diagonal; no columns should be extracted.
        let h = [0.0_f32];
        let result = pivoted_cholesky(&h, 1, 0.0, 1).unwrap();
        assert_eq!(result.rank(), 0);
    }

    // ── solve_lower_transposed + h_inv_column tests ────────────────

    #[test]
    fn solve_lower_transposed_identity_passes_b_through() {
        let n = 4;
        let mut l = vec![0.0_f32; n * (n + 1) / 2];
        for i in 0..n {
            l[lower_tri_offset(i, i)] = 1.0;
        }
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0];
        solve_lower_transposed(&l, n, &mut x);
        assert_eq!(x, vec![1.0_f32, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn solve_lower_transposed_known_2x2() {
        // L = [[2, 0], [1, 2]] → L^T = [[2, 1], [0, 2]]
        // L^T x = [1, 0] → x = [0.5, 0]
        // L^T x = [0, 1] → x = [-0.25, 0.5]
        let l = vec![2.0_f32, 1.0, 2.0];
        let n = 2;

        let mut x0 = vec![1.0_f32, 0.0];
        solve_lower_transposed(&l, n, &mut x0);
        assert!((x0[0] - 0.5).abs() < 1e-6, "x0[0] = {}", x0[0]);
        assert!((x0[1] - 0.0).abs() < 1e-6, "x0[1] = {}", x0[1]);

        let mut x1 = vec![0.0_f32, 1.0];
        solve_lower_transposed(&l, n, &mut x1);
        assert!((x1[0] - (-0.25)).abs() < 1e-6, "x1[0] = {}", x1[0]);
        assert!((x1[1] - 0.5).abs() < 1e-6, "x1[1] = {}", x1[1]);
    }

    #[test]
    fn solve_full_system_via_lower_then_transposed() {
        // Solve H x = b where H = L L^T: first L y = b (forward sub),
        // then L^T x = y (back sub). Verify H x ≈ b to tolerance.
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
        let mut x = b.clone();
        solve_lower(&l, n, &mut x);
        solve_lower_transposed(&l, n, &mut x);

        // Reconstruct H x and compare to b.
        for i in 0..n {
            let mut sum = 0.0_f64;
            for j in 0..n {
                let h_ij = if i <= j {
                    h[upper_tri_offset(n, i, j)]
                } else {
                    h[upper_tri_offset(n, j, i)]
                } as f64;
                sum += h_ij * x[j] as f64;
            }
            assert!(
                (sum - b[i] as f64).abs() < 1e-4,
                "H x [{i}] = {sum}, expected {}",
                b[i]
            );
        }
    }

    #[test]
    #[should_panic(expected = "solve_lower_transposed: x length")]
    fn solve_lower_transposed_panics_on_length_mismatch() {
        let l = vec![1.0_f32, 0.0, 1.0];
        let mut x = vec![1.0_f32]; // wrong length for n=2
        solve_lower_transposed(&l, 2, &mut x);
    }

    #[test]
    fn h_inv_column_identity_returns_unit_vector() {
        // H = I → H⁻¹ = I → column j is e_j.
        let n = 4;
        let mut l = vec![0.0_f32; n * (n + 1) / 2];
        for i in 0..n {
            l[lower_tri_offset(i, i)] = 1.0;
        }
        for j in 0..n {
            let col = h_inv_column(&l, n, j);
            for (i, &v) in col.iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (v - want).abs() < 1e-6,
                    "H⁻¹[{i}, {j}] = {v}, expected {want}",
                );
            }
        }
    }

    #[test]
    fn h_inv_column_matches_closed_form_2x2() {
        // H = [[4, 2], [2, 5]] → det = 16 → H⁻¹ = (1/16) [[5, -2], [-2, 4]]
        // column 0 = [5/16, -2/16] = [0.3125, -0.125]
        // column 1 = [-2/16, 4/16] = [-0.125, 0.25]
        let h = [4.0_f32, 2.0, 5.0];
        let l = cholesky(&h, 2).unwrap();
        let c0 = h_inv_column(&l, 2, 0);
        let c1 = h_inv_column(&l, 2, 1);
        assert!((c0[0] - 0.3125).abs() < 1e-4, "c0[0] = {}", c0[0]);
        assert!((c0[1] - (-0.125)).abs() < 1e-4, "c0[1] = {}", c0[1]);
        assert!((c1[0] - (-0.125)).abs() < 1e-4, "c1[0] = {}", c1[0]);
        assert!((c1[1] - 0.25).abs() < 1e-4, "c1[1] = {}", c1[1]);
    }

    #[test]
    fn h_inv_column_satisfies_h_times_col_equals_unit() {
        // H @ (H⁻¹ e_j) must equal e_j for any j. Use a known SPD H.
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
            let col = h_inv_column(&l, n, j);
            for i in 0..n {
                let mut sum = 0.0_f64;
                for k in 0..n {
                    let h_ik = if i <= k {
                        h[upper_tri_offset(n, i, k)]
                    } else {
                        h[upper_tri_offset(n, k, i)]
                    } as f64;
                    sum += h_ik * col[k] as f64;
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - want).abs() < 1e-3,
                    "(H H⁻¹ e_{j})[{i}] = {sum}, expected {want}",
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "j (")]
    fn h_inv_column_panics_on_out_of_range_j() {
        let l = vec![1.0_f32, 0.0, 1.0];
        let _ = h_inv_column(&l, 2, 5);
    }
}
