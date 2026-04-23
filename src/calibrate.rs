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
//! The default corpus is a short English prose paragraph
//! ([`DEFAULT_CALIBRATION_CORPUS`]) — enough for the allocator to
//! pick up a meaningful salience signal on SmolLM2-sized models.
//! Callers who want more can pass their own string.

use std::path::Path;

use thiserror::Error;

use crate::forward::{AwqCollector, ForwardModel, KvCache, ModelLoadError, ModelScratch};
use crate::gguf::parser::parse_path;
use crate::stego::tensor_map::TensorMap;
use crate::v2::fs::{Filesystem as V2Filesystem, FsError as V2FsError};
use crate::v2::salience::{PeriodicSlotSalience, SalienceError, SalienceTable};

/// Calibration corpus bundled with the CLI. Chosen to be short
/// enough to run in seconds on CPU but long enough to produce a
/// non-degenerate salience signal across the standard linear
/// layers.
pub const DEFAULT_CALIBRATION_CORPUS: &str = concat!(
    "The invention of the printing press in the middle of the fifteenth century ",
    "is often described as one of the most consequential technological changes ",
    "in the history of communication. By replacing manuscript copying with a ",
    "mechanical process, the press reduced the cost and raised the reliability ",
    "of making identical copies of a text. This, in turn, helped standardise ",
    "spelling and vocabulary across regions that had previously used local ",
    "variants, because printers tended to draw on a smaller set of exemplars ",
    "when setting type. Scholars, administrators, and merchants all found new ",
    "uses for printed documents, and within a few decades the technology had ",
    "spread across Europe and begun to reshape education, religion, and trade.",
);

/// Max tokens to feed through the forward pass — caps runtime on
/// long corpora and keeps the scratch buffers bounded.
pub const MAX_CALIBRATION_TOKENS: usize = 512;

#[derive(Debug, Error)]
pub enum CalibrateError {
    #[error("load model: {0}")]
    LoadModel(#[from] ModelLoadError),

    #[error("tokenize corpus: {0}")]
    Tokenize(String),

    #[error("corpus tokenizes to 0 tokens — need at least 1")]
    EmptyCorpus,

    #[error("gguf parse: {0}")]
    GgufParse(String),

    #[error("filesystem: {0}")]
    Fs(#[from] V2FsError),

    #[error("salience: {0}")]
    Salience(#[from] SalienceError),

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
///   2. Tokenize `corpus`, clip to `MAX_CALIBRATION_TOKENS`.
///   3. Run one forward pass with an `AwqCollector` observer.
///   4. Finalize salience, map tensor names → TensorMap slot indices,
///      expand per-channel to per-weight.
///   5. `commit_salience(&table)` on `fs`.
pub fn run_calibration(
    fs: &mut V2Filesystem,
    model_path: &Path,
    tensor_map: &TensorMap,
    corpus: &str,
) -> Result<CalibrationSummary, CalibrateError> {
    // 1. Load the model.
    let model = ForwardModel::load(model_path)?;

    // 2. Tokenize.
    let mut tokens = model
        .encode(corpus)
        .map_err(|e| CalibrateError::Tokenize(e.to_string()))?;
    if tokens.is_empty() {
        return Err(CalibrateError::EmptyCorpus);
    }
    tokens.truncate(MAX_CALIBRATION_TOKENS);
    let token_count = tokens.len();

    // 3. Forward pass with observer.
    let ctx_len = tokens.len();
    let mut cache = KvCache::new(&model.config, ctx_len);
    let mut scratch = ModelScratch::new(&model.config, ctx_len, ctx_len);
    let mut collector = AwqCollector::new();
    let _ =
        model.forward_all_logits_with_observer(&tokens, &mut cache, &mut scratch, &mut collector);
    let per_tensor_salience = collector.finalize();

    // 4. Parse GGUF to get tensor shapes (for per-channel→per-weight
    //    expansion we need the input dimension).
    let gguf = parse_path(model_path).map_err(|e| CalibrateError::GgufParse(e.to_string()))?;
    let table = build_salience_table(tensor_map, &gguf.tensors, &per_tensor_salience)?;
    let populated_slot_count = table.populated_slot_count();
    let total_slot_count = table.slot_count();

    // 5. Commit.
    let ptr = fs.commit_salience(&table)?;
    let new_salience_inode_nonzero = !ptr.is_null();

    Ok(CalibrationSummary {
        token_count,
        populated_slot_count,
        total_slot_count,
        new_salience_inode_nonzero,
    })
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
