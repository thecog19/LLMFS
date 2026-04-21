//! Calibration produces per-weight salience scores used by V2's
//! sensitivity-ordered allocator (DESIGN-NEW §15). Three estimator
//! tiers are envisioned:
//!
//! - **Tier 0 — magnitude-only.** Calibration-corpus-free: salience
//!   = `|w|`. Used by Layer 0 (implicit metadata addressing) and as
//!   the fallback when a richer estimator isn't available. This
//!   module currently implements only Tier 0.
//! - **Tier 1 — AWQ.** Single forward pass; per-channel salience
//!   `mean(|x|) * |w|`. Activation-aware. Requires the wgpu/dzn
//!   forward-pass infrastructure described in `docs/gpu-dev-env.md`.
//! - **Tier 2 — Hessian (GPTQ).** Accumulated `X^T X` over a
//!   calibration corpus, per-column diagonal as salience. Most
//!   accurate; needed for Layer 5 (error compensation).

pub mod magnitude;
pub mod placement;

use crate::gguf::quant::GgufQuantType;

/// Identifies a single weight within an `AllocationPlan`'s eligible-
/// tensor set. Indexes into `TensorMap::slots` and the weight stream
/// of the corresponding tensor.
///
/// `Ord` derives lexicographically (slot then weight); used as a
/// stable tiebreaker when two weights have identical salience scores
/// — matters for cross-machine reproducibility of the placement
/// ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WeightRef {
    pub slot_index: u32,
    pub weight_index: u64,
}

/// How many stealable bits one weight of this quant type contributes
/// to the stego address space. Mirrors `GgufQuantType::stealable_bits_hint`
/// but is exposed here so calibration callers don't have to pull in the
/// gguf module just to budget bit counts.
pub fn stealable_bits_for(quant_type: GgufQuantType) -> u32 {
    quant_type.stealable_bits_hint() as u32
}
