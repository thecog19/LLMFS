//! In-memory cache of per-(site, layer) Cholesky factors of the
//! activation Hessian `H`. This is the L2 tier from
//! `docs/compensation-design.md §3.2` — resident factors that
//! Phase E's compensation math solves against.
//!
//! D1-b delivers the *storage* type and the API shape (insert /
//! get / len / iterate). It does **not** yet wire the cache into
//! any hot path; Phase E will plumb it through the write path.
//!
//! Cache state is never persisted to disk — per
//! `docs/concurrency-design.md` and the stego invariant, LLMDB's
//! on-disk surface must be exactly the cover and nothing else.
//! Factors are recomputed on each mount.
//!
//! ## Storage format
//!
//! A [`CholeskyFactor`] bundles the matrix dimension `N` with the
//! lower-triangle packed factor produced by
//! [`crate::forward::linalg::cholesky`]. Storage: `N*(N+1)/2` F32
//! entries — same layout as the Cholesky output, no additional
//! encoding. Row `i` of `L` covers columns `0..=i`, with offset
//! `i*(i+1)/2 + j`.
//!
//! At SmolLM2-135M scale the cache peaks at ~200 MB across 120
//! (site, layer) keys. At 3B scale it peaks around ~5 GB. Larger
//! models run into the low-rank-Cholesky optimization described in
//! `compensation-design.md §2.2` — out of scope for D1-b, see
//! `docs/phase-d-measurement.md §2`.

use std::collections::HashMap;

use crate::forward::awq::ActivationSite;
use crate::forward::linalg::PivotedCholesky;

/// Cholesky factor of a single `(site, layer)` Hessian. `l` is the
/// lower triangle of `L` in row-major packed form (length
/// `n*(n+1)/2`), such that `L * L^T = H`.
#[derive(Debug, Clone)]
pub struct CholeskyFactor {
    /// Matrix dimension — equal to `n` from the source Hessian.
    pub n: usize,
    /// Lower triangle of `L`, row-major packed, F32.
    pub l: Vec<f32>,
}

impl CholeskyFactor {
    /// Construct from raw components. Asserts `l.len() == n*(n+1)/2`;
    /// panics otherwise, on the grounds that a mismatch here means
    /// the caller constructed bogus data and we'd rather fail loud.
    pub fn new(n: usize, l: Vec<f32>) -> Self {
        assert_eq!(
            l.len(),
            n * (n + 1) / 2,
            "CholeskyFactor::new: l length {} does not match n*(n+1)/2 = {} for n = {n}",
            l.len(),
            n * (n + 1) / 2,
        );
        Self { n, l }
    }

    /// Number of bytes this factor occupies in RAM. Used by
    /// [`HessianFactorCache::bytes_resident`] to report the cache's
    /// memory footprint without walking the `HashMap` each time.
    #[inline]
    pub fn bytes_resident(&self) -> usize {
        self.l.capacity() * std::mem::size_of::<f32>()
    }
}

/// Low-rank approximation of a `(site, layer)` Hessian such that
/// `V * V^T ≈ H`, produced by
/// [`crate::forward::linalg::pivoted_cholesky`]. Storage complement
/// to [`CholeskyFactor`]: when the full factor is too large to
/// keep resident (big covers, 7B+), callers keep `LowRankFactor`
/// instead — `O(N·K)` entries where K is the per-site effective
/// rank (3–17% of N per the D0 measurements in
/// `docs/phase-d-measurement.md`).
///
/// Phase E's compensation math can solve against this directly via
/// the pseudoinverse identity `H⁺ = V (V^T V)⁻¹ V^T` (see
/// `docs/compensation-design.md §2.2`).
#[derive(Debug, Clone)]
pub struct LowRankFactor {
    /// Matrix dimension of the approximated H.
    pub n: usize,
    /// Pivot indices in selection order. Length = rank `K`.
    pub pivots: Vec<usize>,
    /// Column-major flat storage of V (N rows × K columns):
    /// `v[k * n + i]` is `V[i][k]`. Length `n * rank`.
    pub v: Vec<f32>,
}

impl LowRankFactor {
    /// Construct from raw components. Asserts the relationships
    /// `v.len() == n * pivots.len()`. Panics otherwise — a mismatch
    /// means the caller constructed bogus data.
    pub fn new(n: usize, pivots: Vec<usize>, v: Vec<f32>) -> Self {
        assert_eq!(
            v.len(),
            n * pivots.len(),
            "LowRankFactor::new: v length {} does not match n * rank = {n} * {} = {}",
            v.len(),
            pivots.len(),
            n * pivots.len(),
        );
        Self { n, pivots, v }
    }

    /// Rank `K` of the approximation.
    pub fn rank(&self) -> usize {
        self.pivots.len()
    }

    /// Whether the factor has zero columns — meaningful after a
    /// tolerance-based early-stop of pivoted Cholesky.
    pub fn is_empty(&self) -> bool {
        self.pivots.is_empty()
    }

    /// RAM bytes this factor occupies, including pivots.
    #[inline]
    pub fn bytes_resident(&self) -> usize {
        self.v.capacity() * std::mem::size_of::<f32>()
            + self.pivots.capacity() * std::mem::size_of::<usize>()
    }
}

impl From<PivotedCholesky> for LowRankFactor {
    fn from(pc: PivotedCholesky) -> Self {
        Self {
            n: pc.n,
            pivots: pc.pivots,
            v: pc.v,
        }
    }
}

/// Per-(site, layer) Cholesky-factor storage. Thin wrapper around a
/// `HashMap` today, but named so Phase E has a stable API point
/// even if the backing store changes (e.g., to a `BTreeMap` for
/// deterministic iteration, or a disk-spill tier for very large
/// covers).
#[derive(Debug, Default, Clone)]
pub struct HessianFactorCache {
    entries: HashMap<(ActivationSite, usize), CholeskyFactor>,
}

impl HessianFactorCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of `(site, layer)` keys stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total resident bytes across all stored factors.
    pub fn bytes_resident(&self) -> usize {
        self.entries.values().map(|f| f.bytes_resident()).sum()
    }

    /// Is `(site, layer)` in the cache?
    pub fn contains(&self, site: ActivationSite, layer: usize) -> bool {
        self.entries.contains_key(&(site, layer))
    }

    /// Get the factor for `(site, layer)`, or `None` if unseen.
    pub fn get(&self, site: ActivationSite, layer: usize) -> Option<&CholeskyFactor> {
        self.entries.get(&(site, layer))
    }

    /// Insert a factor. Overwrites any existing entry for the same
    /// key — caller's responsibility to not do this accidentally.
    pub fn insert(
        &mut self,
        site: ActivationSite,
        layer: usize,
        factor: CholeskyFactor,
    ) -> Option<CholeskyFactor> {
        self.entries.insert((site, layer), factor)
    }

    /// Drop every entry without releasing the backing `HashMap`'s
    /// capacity. Useful between calibration runs that want to reuse
    /// the same cache instance.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Iterate over `(site, layer, factor)` triples. Order is
    /// `HashMap`-defined (currently unordered).
    pub fn iter(&self) -> impl Iterator<Item = (ActivationSite, usize, &CholeskyFactor)> {
        self.entries
            .iter()
            .map(|(&(site, layer), factor)| (site, layer, factor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_factor(n: usize) -> CholeskyFactor {
        // Identity factor: L = I, so H = I too.
        let mut l = vec![0.0_f32; n * (n + 1) / 2];
        for i in 0..n {
            l[i * (i + 1) / 2 + i] = 1.0;
        }
        CholeskyFactor::new(n, l)
    }

    #[test]
    fn factor_new_asserts_length_matches_n() {
        let f = synthetic_factor(4);
        assert_eq!(f.n, 4);
        assert_eq!(f.l.len(), 10);
    }

    #[test]
    #[should_panic(expected = "l length")]
    fn factor_new_rejects_length_mismatch() {
        // n=4 expects 10 entries; give 8.
        let _ = CholeskyFactor::new(4, vec![0.0_f32; 8]);
    }

    #[test]
    fn empty_cache_reports_zero_everywhere() {
        let cache = HessianFactorCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.bytes_resident(), 0);
        assert!(!cache.contains(ActivationSite::QkvInput, 0));
        assert!(cache.get(ActivationSite::QkvInput, 0).is_none());
    }

    #[test]
    fn insert_then_get_round_trips() {
        let mut cache = HessianFactorCache::new();
        let f = synthetic_factor(4);
        let replaced = cache.insert(ActivationSite::QkvInput, 3, f.clone());
        assert!(replaced.is_none());
        assert_eq!(cache.len(), 1);
        assert!(cache.contains(ActivationSite::QkvInput, 3));
        let got = cache.get(ActivationSite::QkvInput, 3).unwrap();
        assert_eq!(got.n, 4);
        assert_eq!(got.l, f.l);
    }

    #[test]
    fn insert_returns_previous_value_on_overwrite() {
        let mut cache = HessianFactorCache::new();
        let f1 = synthetic_factor(4);
        cache.insert(ActivationSite::QkvInput, 0, f1);
        let f2 = synthetic_factor(4);
        let replaced = cache.insert(ActivationSite::QkvInput, 0, f2);
        assert!(replaced.is_some(), "overwrite must return the previous factor");
    }

    #[test]
    fn distinct_keys_do_not_collide() {
        let mut cache = HessianFactorCache::new();
        cache.insert(ActivationSite::QkvInput, 0, synthetic_factor(4));
        cache.insert(ActivationSite::QkvInput, 1, synthetic_factor(4));
        cache.insert(ActivationSite::AttnOutputInput, 0, synthetic_factor(4));
        cache.insert(ActivationSite::FfnGateUpInput, 0, synthetic_factor(4));
        cache.insert(ActivationSite::FfnDownInput, 0, synthetic_factor(8));
        assert_eq!(cache.len(), 5);
        assert!(cache.contains(ActivationSite::QkvInput, 0));
        assert!(cache.contains(ActivationSite::QkvInput, 1));
        assert!(cache.contains(ActivationSite::AttnOutputInput, 0));
        assert!(cache.contains(ActivationSite::FfnGateUpInput, 0));
        assert!(cache.contains(ActivationSite::FfnDownInput, 0));
        assert!(!cache.contains(ActivationSite::AttnOutputInput, 1));
    }

    #[test]
    fn bytes_resident_reports_sum_of_factor_sizes() {
        let mut cache = HessianFactorCache::new();
        let n4 = 4;
        let n8 = 8;
        cache.insert(ActivationSite::QkvInput, 0, synthetic_factor(n4));
        cache.insert(ActivationSite::FfnDownInput, 0, synthetic_factor(n8));
        let expected_4 = n4 * (n4 + 1) / 2 * std::mem::size_of::<f32>();
        let expected_8 = n8 * (n8 + 1) / 2 * std::mem::size_of::<f32>();
        assert_eq!(cache.bytes_resident(), expected_4 + expected_8);
    }

    #[test]
    fn clear_empties_without_changing_capacity_semantics() {
        let mut cache = HessianFactorCache::new();
        cache.insert(ActivationSite::QkvInput, 0, synthetic_factor(4));
        cache.insert(ActivationSite::QkvInput, 1, synthetic_factor(4));
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.bytes_resident(), 0);
    }

    #[test]
    fn iter_walks_all_entries() {
        let mut cache = HessianFactorCache::new();
        cache.insert(ActivationSite::QkvInput, 0, synthetic_factor(4));
        cache.insert(ActivationSite::FfnDownInput, 5, synthetic_factor(8));
        let seen: Vec<(ActivationSite, usize)> = cache
            .iter()
            .map(|(site, layer, _)| (site, layer))
            .collect();
        assert_eq!(seen.len(), 2);
        assert!(seen.contains(&(ActivationSite::QkvInput, 0)));
        assert!(seen.contains(&(ActivationSite::FfnDownInput, 5)));
    }

    // ── LowRankFactor tests ─────────────────────────────────────────

    #[test]
    fn low_rank_factor_new_round_trips_fields() {
        let f = LowRankFactor::new(4, vec![1, 3], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        assert_eq!(f.n, 4);
        assert_eq!(f.rank(), 2);
        assert_eq!(f.pivots, vec![1, 3]);
        assert_eq!(f.v.len(), 8);
        assert!(!f.is_empty());
    }

    #[test]
    fn low_rank_factor_empty_reports_zero_rank() {
        let f = LowRankFactor::new(4, Vec::new(), Vec::new());
        assert_eq!(f.rank(), 0);
        assert!(f.is_empty());
        // v.capacity() is 0 for a fresh empty Vec, so bytes_resident
        // is just the pivots' backing (also 0).
        assert_eq!(f.bytes_resident(), 0);
    }

    #[test]
    #[should_panic(expected = "v length")]
    fn low_rank_factor_new_rejects_length_mismatch() {
        // n=4, rank=2 expects v.len() == 8; give 6.
        let _ = LowRankFactor::new(4, vec![0, 1], vec![0.0; 6]);
    }

    #[test]
    fn low_rank_factor_bytes_resident_counts_both_storages() {
        // n=4, rank=2 → 8 f32 entries in v + 2 usize entries in pivots.
        let f = LowRankFactor::new(4, vec![0, 1], vec![0.0_f32; 8]);
        let expected =
            8 * std::mem::size_of::<f32>() + 2 * std::mem::size_of::<usize>();
        assert_eq!(f.bytes_resident(), expected);
    }

    #[test]
    fn low_rank_factor_from_pivoted_cholesky_preserves_fields() {
        // Drive pivoted_cholesky on a small SPD matrix, convert,
        // and verify the LowRankFactor sees the same data.
        use crate::forward::hessian::upper_tri_offset;
        use crate::forward::linalg::pivoted_cholesky;

        let n = 3;
        let mut h = vec![0.0_f32; n * (n + 1) / 2];
        h[upper_tri_offset(n, 0, 0)] = 9.0;
        h[upper_tri_offset(n, 1, 1)] = 1.0;
        h[upper_tri_offset(n, 2, 2)] = 4.0;
        let pc = pivoted_cholesky(&h, n, 0.0, n).unwrap();
        let factor = LowRankFactor::from(pc.clone());

        assert_eq!(factor.n, pc.n);
        assert_eq!(factor.pivots, pc.pivots);
        assert_eq!(factor.v, pc.v);
        assert_eq!(factor.rank(), pc.rank());
    }
}
