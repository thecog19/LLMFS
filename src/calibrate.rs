//! Orchestration for the `llmdb calibrate` verb.
//!
//! Connects the forward-pass AWQ collector (B1) to the V2 salience
//! inode (B2/B3) via the allocator's compound FitKey (B4a). Flow:
//!
//! 1. Parse the GGUF to get its tensor-name → shape + offset map.
//! 2. Build a [`crate::forward::ForwardModel`] from the same file.
//! 3. Run the calibration corpus through it with an
//!    [`crate::forward::AwqCollector`] observing every matmul input
//!    site.
//! 4. Expand per-channel salience into per-weight salience per
//!    [`TensorMap`] slot (columns of the row-major weight matrix
//!    cycle through input channels).
//! 5. Build a [`SalienceTable`] keyed by slot index.
//! 6. Commit it via [`crate::v2::fs::Filesystem::commit_salience`].
//!
//! The default corpus is an excerpt from the wiki.test.raw benchmark
//! ([`DEFAULT_CALIBRATION_CORPUS`]) — long enough to tokenize past
//! `N_max` at every observation site on SmolLM2 (and comfortably past
//! it on 3B-class models), so [`CalibrationMode::Full`] produces a
//! non-singular Hessian at every site. Callers who want a different
//! corpus can pass their own string.

use std::collections::HashMap;
use std::path::Path;

use thiserror::Error;

use crate::forward::awq::{ActivationSite, tensor_names_for};
use crate::forward::linalg::{self, CholeskyError};
use crate::forward::{
    AwqCollector, ForwardModel, HessianAccumulator, KvCache, ModelLoadError, ModelScratch,
};
use crate::gguf::parser::parse_path;
use crate::stego::tensor_map::TensorMap;
use crate::v2::fs::{Filesystem as V2Filesystem, FsError as V2FsError};
use crate::v2::salience::{PeriodicSlotSalience, SalienceError, SalienceTable};

/// Calibration corpus bundled with the binary. Currently the first
/// 50 KB of `wiki.test.raw`, which tokenizes to roughly 12 k tokens
/// under SmolLM2's BPE — comfortably past the `N_max = ffn_dim =
/// 1536` bound needed for a non-singular Hessian on SmolLM2, and
/// enough headroom for [`CalibrationMode::Full`] on 3B-class
/// architectures (`N_max ~ 8192`) before the corpus becomes the
/// binding constraint.
pub const DEFAULT_CALIBRATION_CORPUS: &str = include_str!("data/calibration-corpus.txt");

/// Token cap for [`CalibrationMode::Fast`] (AWQ). First-moment
/// statistics stabilize quickly; ~500 tokens is plenty and keeps
/// Fast calibration in the seconds regime.
pub const MAX_AWQ_TOKENS: usize = 512;

/// Token cap for [`CalibrationMode::Full`] (Hessian + OBS). Needs
/// `T >= N_max` for the per-site Hessians to be non-singular. 2048
/// covers SmolLM2 (N_max = 1536); for larger covers the Full path
/// will produce rank-deficient Hessians and Cholesky will fail with
/// [`CalibrateError::Cholesky`], which is the correct failure mode
/// — callers should bump this or provide a longer corpus.
pub const MAX_HESSIAN_TOKENS: usize = 2048;

/// Which calibration algorithm to run. [`CalibrationMode::Fast`] is
/// the existing AWQ path (first-moment mean(|x_c|) per channel);
/// [`CalibrationMode::Full`] is the Hessian + OBS path that
/// `docs/compensation-design.md §1.2` specifies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationMode {
    /// AWQ — per-channel `mean(|x_c|)`. Seconds on CPU, low memory,
    /// first-moment approximation to saliency. Default.
    Fast,
    /// Hessian + OBS — per-channel `1 / diag(H^-1)_c`. Minutes on CPU
    /// (dominated by `X^T X` accumulation; quantified in
    /// `docs/phase-d-measurement.md`), ~hundreds of MB for the full
    /// H matrices held in RAM during finalize. Full second-moment
    /// saliency with cross-channel coupling accounted for.
    Full,
}

impl CalibrationMode {
    fn max_tokens(self) -> usize {
        match self {
            Self::Fast => MAX_AWQ_TOKENS,
            Self::Full => MAX_HESSIAN_TOKENS,
        }
    }
}

#[derive(Debug, Error)]
pub enum CalibrateError {
    #[error("load model: {0}")]
    LoadModel(#[from] ModelLoadError),

    #[error("tokenize corpus: {0}")]
    Tokenize(String),

    #[error("corpus tokenizes to 0 tokens — need at least 1")]
    EmptyCorpus,

    #[error(
        "corpus tokenizes to {got} tokens; Full calibration needs at least {need} \
         (N_max = {n_max}) for H to be non-singular. Provide a longer corpus."
    )]
    CorpusTooShortForFull {
        got: usize,
        need: usize,
        n_max: usize,
    },

    #[error("gguf parse: {0}")]
    GgufParse(String),

    #[error("filesystem: {0}")]
    Fs(#[from] V2FsError),

    #[error("salience: {0}")]
    Salience(#[from] SalienceError),

    #[error("cholesky on H at ({site:?}, layer {layer}): {source}")]
    Cholesky {
        site: ActivationSite,
        layer: usize,
        #[source]
        source: CholeskyError,
    },

    #[error(
        "tensor {tensor}: weight_count mismatch between TensorMap ({map_count}) \
         and GGUF ({gguf_count})"
    )]
    WeightCountMismatch {
        tensor: String,
        map_count: u64,
        gguf_count: u64,
    },
}

/// Result summary returned by [`run_calibration`]. The CLI prints
/// this in human-readable form; tests can assert on the counts.
#[derive(Debug, Clone)]
pub struct CalibrationSummary {
    pub token_count: usize,
    pub populated_slot_count: usize,
    pub total_slot_count: usize,
    pub new_salience_inode_nonzero: bool,
}

/// Run a full calibration pass end-to-end. Expects the V2 cover
/// to already be initialized (mounted before this call).
///
/// Steps:
///   1. Load the model from `model_path` (same file the cover lives in).
///   2. Tokenize `corpus`, clip to `mode.max_tokens()`.
///   3. Run one forward pass. In `Fast` mode the observer is
///      [`AwqCollector`]; in `Full` mode it is [`HessianAccumulator`].
///   4. Derive per-tensor salience:
///        - Fast: `mean(|x_c|)` from AWQ, directly per-channel.
///        - Full: factor each site's H → Cholesky → extract
///          `1 / diag(H^-1)` via forward-substitution on unit
///          vectors (see `docs/compensation-design.md §1.2`).
///   5. Map tensor names → TensorMap slot indices; expand
///      per-channel to per-weight via the periodic encoder.
///   6. `commit_salience(&table)` on `fs`.
pub fn run_calibration(
    fs: &mut V2Filesystem,
    model_path: &Path,
    tensor_map: &TensorMap,
    corpus: &str,
    mode: CalibrationMode,
) -> Result<CalibrationSummary, CalibrateError> {
    // 1. Load the model.
    let model = ForwardModel::load(model_path)?;

    // 2. Tokenize and clip.
    let mut tokens = model
        .encode(corpus)
        .map_err(|e| CalibrateError::Tokenize(e.to_string()))?;
    if tokens.is_empty() {
        return Err(CalibrateError::EmptyCorpus);
    }
    tokens.truncate(mode.max_tokens());
    let token_count = tokens.len();

    // 3 + 4. Forward pass + per-tensor salience, mode-dependent.
    let per_tensor_salience = match mode {
        CalibrationMode::Fast => run_fast_forward(&model, &tokens),
        CalibrationMode::Full => run_full_forward(&model, &tokens, token_count)?,
    };

    // 5. Parse GGUF for tensor shapes and build the salience table.
    let gguf = parse_path(model_path).map_err(|e| CalibrateError::GgufParse(e.to_string()))?;
    let table = build_salience_table(tensor_map, &gguf.tensors, &per_tensor_salience)?;
    let populated_slot_count = table.populated_slot_count();
    let total_slot_count = table.slot_count();

    // 6. Commit.
    let ptr = fs.commit_salience(&table)?;
    let new_salience_inode_nonzero = !ptr.is_null();

    Ok(CalibrationSummary {
        token_count,
        populated_slot_count,
        total_slot_count,
        new_salience_inode_nonzero,
    })
}

/// AWQ pass: one forward through the model with an
/// [`AwqCollector`] observer; finalize to per-channel `mean(|x_c|)`.
fn run_fast_forward(model: &ForwardModel, tokens: &[u32]) -> HashMap<String, Vec<f32>> {
    let ctx_len = tokens.len();
    let mut cache = KvCache::new(&model.config, ctx_len);
    let mut scratch = ModelScratch::new(&model.config, ctx_len, ctx_len);
    let mut collector = AwqCollector::new();
    let _ =
        model.forward_all_logits_with_observer(tokens, &mut cache, &mut scratch, &mut collector);
    collector.finalize()
}

/// Full pass: one forward through the model with a
/// [`HessianAccumulator`] observer; for every observed
/// `(site, layer)` Cholesky-factorize H and extract OBS saliency
/// `1 / diag(H^-1)`; fan out per-site-layer values to each weight
/// tensor that shares that input (same convention as AWQ's
/// [`tensor_names_for`]).
fn run_full_forward(
    model: &ForwardModel,
    tokens: &[u32],
    token_count: usize,
) -> Result<HashMap<String, Vec<f32>>, CalibrateError> {
    let ctx_len = tokens.len();
    let mut cache = KvCache::new(&model.config, ctx_len);
    let mut scratch = ModelScratch::new(&model.config, ctx_len, ctx_len);
    let mut acc = HessianAccumulator::new();
    let _ = model.forward_all_logits_with_observer(tokens, &mut cache, &mut scratch, &mut acc);
    let finalized = acc.finalize();

    // Sanity: every (site, layer) Hessian must have N <= token_count
    // for non-singular factorization. If it doesn't, surface a clear
    // error instead of letting Cholesky detect the failure deeper in
    // the pipeline.
    if let Some(((site, layer), (n_max, _))) = finalized
        .iter()
        .max_by_key(|(_, (n, _))| *n)
    {
        if *n_max > token_count {
            return Err(CalibrateError::CorpusTooShortForFull {
                got: token_count,
                need: *n_max,
                n_max: *n_max,
            });
        }
        // Prevent "unused" warnings on the key when no failure.
        let _ = (site, layer);
    }

    let mut per_tensor = HashMap::new();
    for ((site, layer), (n, h_upper)) in finalized {
        let l = linalg::cholesky(&h_upper, n).map_err(|e| CalibrateError::Cholesky {
            site,
            layer,
            source: e,
        })?;
        let obs = linalg::obs_saliency(&l, n);
        for name in tensor_names_for(site, layer) {
            per_tensor.insert(name, obs.clone());
        }
    }
    Ok(per_tensor)
}

/// Convert AWQ's `HashMap<tensor_name, per_channel_salience>` into
/// a `SalienceTable` indexed by the TensorMap's slot order. Each
/// slot's output is dense per-weight: row-major `[out, in]`
/// weights cycle through input channels with period `in_dim`, so
/// `weight[i]` → `channel[i % in_dim]`.
fn build_salience_table(
    tensor_map: &TensorMap,
    gguf_tensors: &[crate::gguf::parser::GgufTensorInfo],
    per_tensor_salience: &std::collections::HashMap<String, Vec<f32>>,
) -> Result<SalienceTable, CalibrateError> {
    let mut per_slot: Vec<Option<PeriodicSlotSalience>> = vec![None; tensor_map.slots.len()];
    for (slot_idx, slot) in tensor_map.slots.iter().enumerate() {
        let Some(channel_salience) = per_tensor_salience.get(&slot.name) else {
            continue; // AWQ didn't observe this tensor — leave slot empty.
        };
        let Some(tensor_info) = gguf_tensors.iter().find(|t| t.name == slot.name) else {
            continue; // name match in awq but not gguf — skip gracefully.
        };
        let gguf_elems = tensor_info.element_count();
        if gguf_elems != slot.weight_count {
            return Err(CalibrateError::WeightCountMismatch {
                tensor: slot.name.clone(),
                map_count: slot.weight_count,
                gguf_count: gguf_elems,
            });
        }
        // GGUF dimensions are innermost-first: `[in_dim, out_dim, ...]`.
        // For a 2-D weight, `dimensions[0]` is the input dim (the
        // axis the matmul reduces over).
        let in_dim = tensor_info
            .dimensions
            .first()
            .copied()
            .unwrap_or(channel_salience.len() as u64) as usize;
        let channels = channel_salience.len();
        if in_dim == 0 || channels == 0 {
            continue;
        }
        let period = in_dim.min(channels);
        per_slot[slot_idx] = Some(PeriodicSlotSalience::new(
            slot.weight_count,
            channel_salience[..period].to_vec(),
        )?);
    }
    Ok(SalienceTable::new(per_slot))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::parser::GgufTensorInfo;
    use crate::gguf::quant::GgufQuantType;
    use crate::stego::planner::TensorTier;
    use crate::stego::tensor_map::TensorSlot;
    use std::collections::HashMap;

    fn tensor_info(name: &str, dimensions: Vec<u64>) -> GgufTensorInfo {
        GgufTensorInfo {
            name: name.to_owned(),
            dimensions,
            raw_type_id: 1, // F16
            data_offset: 0,
        }
    }

    fn slot(name: &str, weight_count: u64) -> TensorSlot {
        TensorSlot {
            name: name.to_owned(),
            quant_type: GgufQuantType::F16,
            tier: TensorTier::Tier1,
            data_offset: 0,
            weight_count,
            stealable_bits_per_weight: 4,
            capacity_bits: weight_count * 4,
            bit_start: 0,
            bit_end: weight_count * 4,
        }
    }

    #[test]
    fn expand_per_channel_salience_cycles_across_weights() {
        // 2 rows × 3 columns = 6 weights. Per-channel salience of
        // [1.0, 2.0, 3.0]. Expansion should be [1, 2, 3, 1, 2, 3].
        let map = TensorMap {
            slots: vec![slot("blk.0.attn_q.weight", 6)],
            total_capacity_bits: 24,
            total_capacity_bytes: 3,
        };
        let tensors = vec![tensor_info("blk.0.attn_q.weight", vec![3, 2])];
        let mut salience = HashMap::new();
        salience.insert("blk.0.attn_q.weight".to_owned(), vec![1.0_f32, 2.0, 3.0]);
        let table = build_salience_table(&map, &tensors, &salience).unwrap();
        assert_eq!(table.slot_count(), 1);
        assert_eq!(table.populated_slot_count(), 1);
        assert_eq!(
            table.encode().len(),
            36,
            "store one 3-channel period, not 6 dense weights"
        );
        // Access via max_over_range for 1-weight windows to verify
        // each position.
        for i in 0..6_u64 {
            let expected = match i % 3 {
                0 => 1.0,
                1 => 2.0,
                _ => 3.0,
            };
            assert!(
                (table.max_over_range(0, i, 1) - expected).abs() < 1e-6,
                "weight {i}: expected {expected}, got {}",
                table.max_over_range(0, i, 1),
            );
        }
    }

    #[test]
    fn slots_with_no_awq_output_stay_unpopulated() {
        let map = TensorMap {
            slots: vec![
                slot("blk.0.attn_q.weight", 6),
                slot("blk.0.token_embd.weight", 12),
            ],
            total_capacity_bits: 0,
            total_capacity_bytes: 0,
        };
        let tensors = vec![
            tensor_info("blk.0.attn_q.weight", vec![3, 2]),
            tensor_info("blk.0.token_embd.weight", vec![4, 3]),
        ];
        let mut salience = HashMap::new();
        salience.insert("blk.0.attn_q.weight".to_owned(), vec![1.0_f32, 2.0, 3.0]);
        let table = build_salience_table(&map, &tensors, &salience).unwrap();
        assert_eq!(table.slot_count(), 2);
        assert_eq!(table.populated_slot_count(), 1);
        assert_eq!(table.max_over_range(1, 0, 12), 0.0);
    }

    #[test]
    fn weight_count_mismatch_errors() {
        let map = TensorMap {
            slots: vec![slot("blk.0.attn_q.weight", 6)],
            total_capacity_bits: 24,
            total_capacity_bytes: 3,
        };
        let tensors = vec![tensor_info("blk.0.attn_q.weight", vec![3, 100])]; // 300 elements
        let mut salience = HashMap::new();
        salience.insert("blk.0.attn_q.weight".to_owned(), vec![1.0_f32, 2.0, 3.0]);
        let err = build_salience_table(&map, &tensors, &salience).unwrap_err();
        assert!(matches!(err, CalibrateError::WeightCountMismatch { .. },));
    }
}
