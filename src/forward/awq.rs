//! B1 — Tier-1 AWQ salience collector.
//!
//! AWQ's per-channel salience metric is `mean(|x_c|)` across every
//! token in the calibration corpus, computed on the *input* of each
//! linear projection. Since a linear `y = x @ W.T` scales each
//! output by a linear combination of input channels, channels with
//! consistently-large magnitude tokens matter more for downstream
//! fidelity — so they should take precedence during steganographic
//! placement.
//!
//! This module plugs into the block's forward pass via a
//! [`BlockObserver`] trait: at each of the four distinct matmul-
//! input sites per block, the forward pass calls
//! `observer.observe(site, layer, x, rows, cols)`. Multiple
//! projections share an input site (q/k/v all take the post-attn-
//! norm tensor), so the collector dedupes by site rather than by
//! projection name. After the forward pass,
//! [`AwqCollector::finalize`] fans the site-keyed tallies back out
//! to per-tensor salience.
//!
//! Sharing is explicit in the contract:
//!   - `blk.N.attn_q.weight`, `blk.N.attn_k.weight`,
//!     `blk.N.attn_v.weight` → salience from `QkvInput[N]`.
//!   - `blk.N.attn_output.weight` → salience from `AttnOutputInput[N]`.
//!   - `blk.N.ffn_gate.weight`, `blk.N.ffn_up.weight` → salience
//!     from `FfnGateUpInput[N]`.
//!   - `blk.N.ffn_down.weight` → salience from `FfnDownInput[N]`.
//!
//! The embedding table and LM head aren't matmul targets in the
//! same sense (they're lookups / projections to vocab) and AWQ's
//! original paper scopes to the transformer block's linears — so
//! we don't collect on them.

use std::collections::HashMap;

use crate::forward::block::BlockObserver;

/// Which matmul *input* the observer is being given.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationSite {
    /// Input to Q/K/V projections (post-attn-norm x).
    QkvInput,
    /// Input to the attention output projection (concatenated
    /// per-head attention output).
    AttnOutputInput,
    /// Input to the FFN gate + up projections (post-FFN-norm x).
    FfnGateUpInput,
    /// Input to the FFN down projection (SwiGLU output).
    FfnDownInput,
}

/// Running tally of absolute activation magnitudes, keyed by
/// `(site, layer)`. Finalizing divides by the observed token
/// count to produce `mean(|x_c|)`.
#[derive(Debug, Default)]
pub struct AwqCollector {
    /// Σ|x_c| per input channel. Keyed by `(site, layer_idx)`.
    sums: HashMap<(ActivationSite, usize), Vec<f64>>,
    /// Number of rows (tokens) accumulated at each site. Same
    /// across all channels of a given `(site, layer)` key.
    counts: HashMap<(ActivationSite, usize), u64>,
}

impl AwqCollector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop every accumulated slot without deallocating the map —
    /// useful if the same collector runs multiple corpora.
    pub fn clear(&mut self) {
        self.sums.clear();
        self.counts.clear();
    }

    /// Expand the site-keyed tallies out to per-tensor salience.
    /// Each weight tensor gets a `Vec<f32>` of length = its input-
    /// channel count (cols), holding `mean(|x_c|)` across the
    /// collected corpus.
    ///
    /// Layers we never observed are absent from the returned map.
    pub fn finalize(&self) -> HashMap<String, Vec<f32>> {
        let mut out = HashMap::new();
        for (&(site, layer), sum) in self.sums.iter() {
            let count = self.counts.get(&(site, layer)).copied().unwrap_or(0);
            if count == 0 {
                continue;
            }
            let inv = 1.0_f64 / count as f64;
            let salience: Vec<f32> = sum.iter().map(|v| (*v * inv) as f32).collect();
            for name in tensor_names_for(site, layer) {
                out.insert(name, salience.clone());
            }
        }
        out
    }
}

fn tensor_names_for(site: ActivationSite, layer: usize) -> Vec<String> {
    match site {
        ActivationSite::QkvInput => vec![
            format!("blk.{layer}.attn_q.weight"),
            format!("blk.{layer}.attn_k.weight"),
            format!("blk.{layer}.attn_v.weight"),
        ],
        ActivationSite::AttnOutputInput => {
            vec![format!("blk.{layer}.attn_output.weight")]
        }
        ActivationSite::FfnGateUpInput => vec![
            format!("blk.{layer}.ffn_gate.weight"),
            format!("blk.{layer}.ffn_up.weight"),
        ],
        ActivationSite::FfnDownInput => vec![format!("blk.{layer}.ffn_down.weight")],
    }
}

impl BlockObserver for AwqCollector {
    fn observe(
        &mut self,
        site: ActivationSite,
        layer: usize,
        x: &[f32],
        rows: usize,
        cols: usize,
    ) {
        assert_eq!(x.len(), rows * cols);
        let entry = self
            .sums
            .entry((site, layer))
            .or_insert_with(|| vec![0.0_f64; cols]);
        assert_eq!(entry.len(), cols, "channel-count changed between calls");
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            for (dst, v) in entry.iter_mut().zip(row.iter()) {
                *dst += v.abs() as f64;
            }
        }
        *self.counts.entry((site, layer)).or_insert(0) += rows as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observe_accumulates_per_channel_absolute_sum() {
        let mut c = AwqCollector::new();
        // Two rows of 3 columns: row 0 = [1, -2, 3], row 1 = [-4, 5, -6].
        // Expected sum(|x|) per column: [5, 7, 9].
        // Expected mean:                  [2.5, 3.5, 4.5].
        let x = [1.0_f32, -2.0, 3.0, -4.0, 5.0, -6.0];
        c.observe(ActivationSite::QkvInput, 0, &x, 2, 3);
        let finalized = c.finalize();
        let salience = &finalized["blk.0.attn_q.weight"];
        assert!((salience[0] - 2.5).abs() < 1e-6);
        assert!((salience[1] - 3.5).abs() < 1e-6);
        assert!((salience[2] - 4.5).abs() < 1e-6);
    }

    #[test]
    fn qkv_site_emits_three_tensor_names_with_the_same_salience() {
        let mut c = AwqCollector::new();
        let x = [1.0_f32, 1.0];
        c.observe(ActivationSite::QkvInput, 5, &x, 1, 2);
        let f = c.finalize();
        let q = f.get("blk.5.attn_q.weight").unwrap();
        let k = f.get("blk.5.attn_k.weight").unwrap();
        let v = f.get("blk.5.attn_v.weight").unwrap();
        assert_eq!(q, k);
        assert_eq!(q, v);
    }

    #[test]
    fn ffn_gate_up_share_salience_but_down_gets_its_own() {
        let mut c = AwqCollector::new();
        c.observe(ActivationSite::FfnGateUpInput, 2, &[2.0_f32, 4.0], 1, 2);
        c.observe(ActivationSite::FfnDownInput, 2, &[10.0_f32, 20.0, 30.0], 1, 3);
        let f = c.finalize();
        assert_eq!(f["blk.2.ffn_gate.weight"], vec![2.0, 4.0]);
        assert_eq!(f["blk.2.ffn_up.weight"], vec![2.0, 4.0]);
        assert_eq!(f["blk.2.ffn_down.weight"], vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn attn_output_tensor_name_is_blk_n_attn_output_weight() {
        let mut c = AwqCollector::new();
        c.observe(ActivationSite::AttnOutputInput, 9, &[7.0_f32; 4], 1, 4);
        let f = c.finalize();
        assert!(f.contains_key("blk.9.attn_output.weight"));
        assert_eq!(f["blk.9.attn_output.weight"], vec![7.0; 4]);
    }

    #[test]
    fn clear_drops_everything() {
        let mut c = AwqCollector::new();
        c.observe(ActivationSite::QkvInput, 0, &[1.0_f32; 2], 1, 2);
        c.clear();
        assert!(c.finalize().is_empty());
    }
}
