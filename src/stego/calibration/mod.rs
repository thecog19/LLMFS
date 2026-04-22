//! Calibration produces per-weight salience scores used by V2's
//! sensitivity-ordered allocator (DESIGN-NEW §15). Three estimator
//! tiers are envisioned:
//!
//! - **Tier 0 — magnitude-only** (this module). Calibration-corpus-
//!   free: salience = `|w|`, derivable from the cover alone. Used by
//!   Layer 0 (anchor placement) and by Layer 3's pristine-run
//!   ordering when no richer estimator is present.
//! - **Tier 1 — AWQ.** Single forward pass; per-channel salience
//!   `mean(|x|) * |w|`. Activation-aware. Lives in
//!   [`crate::forward`] — a hand-rolled CPU transformer forward
//!   pass. The earlier design revision expected a wgpu/dzn GPU path;
//!   see `DESIGN-NEW.MD §15.4` for the hand-rolled-Rust decision,
//!   and the V2-progressive-calibration plan for the build-out.
//! - **Tier 2 — Hessian (GPTQ).** Accumulated `X^T X` over a
//!   calibration corpus, per-column diagonal as salience. Most
//!   accurate; needed for Layer 5 (error compensation). Builds on
//!   Tier 1's forward-pass infrastructure.

pub mod bit_io;
pub mod byte_io;
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
