//! Dequantized F32 weights for a llama-arch forward pass.
//!
//! The whole model is loaded once into owned `Vec<f32>` buffers —
//! no zero-copy aliasing back to mmap. Milestone A optimizes for
//! correctness, not RSS. On SmolLM2-135M-F16 the dequantized
//! weights are ~540 MB in F32, which fits in memory on any box
//! we care about. Larger models will want a streaming / quant-
//! native path; that's Milestone C territory.
//!
//! Weight-layout convention (same as GGUF, same as [`crate::forward::
//! ops::matmul`]): every linear-projection weight is stored row-
//! major as `[out_dim, in_dim]`. Tensor dimensions inside GGUF
//! are reported in **reverse order** (innermost first), so a tensor
//! printed as `dimensions = [hidden, q_width]` is a `[q_width,
//! hidden]` matrix — output first in memory.
//!
//! Some llama GGUFs tie the LM head to the token-embedding table
//! (no separate `output.weight` tensor). SmolLM2-135M is one such
//! model. [`LlamaWeights::load`] handles both: when `output.weight`
//! is absent, `lm_head` aliases `embedding` (same underlying
//! buffer via shared ownership).

use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
use thiserror::Error;

use crate::forward::block::BlockWeights;
use crate::forward::config::LlamaConfig;
use crate::forward::dequant::{self, DequantError};
use crate::gguf::parser::{GgufFile, GgufTensorInfo, parse_path};
use crate::gguf::quant::GgufQuantType;

/// All dequantized weights for one transformer block.
pub struct BlockStorage {
    pub attn_norm: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
}

impl BlockStorage {
    pub fn view(&self) -> BlockWeights<'_> {
        BlockWeights {
            attn_norm: &self.attn_norm,
            wq: &self.wq,
            wk: &self.wk,
            wv: &self.wv,
            wo: &self.wo,
            ffn_norm: &self.ffn_norm,
            w_gate: &self.w_gate,
            w_up: &self.w_up,
            w_down: &self.w_down,
        }
    }
}

/// Dequantized weights for a full llama-arch model.
pub struct LlamaWeights {
    /// `[vocab_size, hidden_dim]` row-major.
    pub embedding: Arc<Vec<f32>>,
    /// Per-block storage, indexed 0..n_layers.
    pub blocks: Vec<BlockStorage>,
    /// `[hidden_dim]` final RMSNorm gain (`output_norm.weight`).
    pub final_norm: Vec<f32>,
    /// `[vocab_size, hidden_dim]` row-major. Aliases `embedding`
    /// when the model ties weights (no `output.weight` tensor).
    pub lm_head: Arc<Vec<f32>>,
}

impl LlamaWeights {
    /// Load + dequantize every tensor a llama-arch forward pass
    /// needs. Returns `WeightLoadError` if any expected tensor is
    /// missing or has a quant type Milestone A can't handle.
    pub fn load<P: AsRef<Path>>(
        path: P,
        cfg: &LlamaConfig,
    ) -> Result<Self, WeightLoadError> {
        let gguf = parse_path(&path).map_err(|e| WeightLoadError::Parse(e.to_string()))?;
        let file =
            std::fs::File::open(&path).map_err(|e| WeightLoadError::Io(e.to_string()))?;
        // SAFETY: the file is opened read-only and the mapping is
        // kept alive only for the dequant loop below.
        let mmap =
            unsafe { Mmap::map(&file).map_err(|e| WeightLoadError::Io(e.to_string()))? };

        let ctx = LoadCtx {
            gguf: &gguf,
            mmap: &mmap,
        };

        // Top-level tensors.
        let embedding = Arc::new(ctx.dequant_tensor(
            "token_embd.weight",
            cfg.vocab_size * cfg.hidden_dim,
        )?);
        let final_norm = ctx.dequant_tensor("output_norm.weight", cfg.hidden_dim)?;

        // LM head: either explicit `output.weight`, or tied to the
        // token embedding.
        let lm_head = if ctx.has("output.weight") {
            Arc::new(ctx.dequant_tensor("output.weight", cfg.vocab_size * cfg.hidden_dim)?)
        } else {
            embedding.clone()
        };

        let q_width = cfg.n_heads * cfg.head_dim;
        let kv_width = cfg.n_kv_heads * cfg.head_dim;

        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for n in 0..cfg.n_layers {
            blocks.push(BlockStorage {
                attn_norm: ctx.dequant_tensor(&format!("blk.{n}.attn_norm.weight"), cfg.hidden_dim)?,
                wq: ctx.dequant_tensor(&format!("blk.{n}.attn_q.weight"), q_width * cfg.hidden_dim)?,
                wk: ctx.dequant_tensor(&format!("blk.{n}.attn_k.weight"), kv_width * cfg.hidden_dim)?,
                wv: ctx.dequant_tensor(&format!("blk.{n}.attn_v.weight"), kv_width * cfg.hidden_dim)?,
                wo: ctx.dequant_tensor(&format!("blk.{n}.attn_output.weight"), cfg.hidden_dim * q_width)?,
                ffn_norm: ctx.dequant_tensor(&format!("blk.{n}.ffn_norm.weight"), cfg.hidden_dim)?,
                w_gate: ctx.dequant_tensor(&format!("blk.{n}.ffn_gate.weight"), cfg.ffn_dim * cfg.hidden_dim)?,
                w_up: ctx.dequant_tensor(&format!("blk.{n}.ffn_up.weight"), cfg.ffn_dim * cfg.hidden_dim)?,
                w_down: ctx.dequant_tensor(&format!("blk.{n}.ffn_down.weight"), cfg.hidden_dim * cfg.ffn_dim)?,
            });
        }

        Ok(Self {
            embedding,
            blocks,
            final_norm,
            lm_head,
        })
    }
}

struct LoadCtx<'a> {
    gguf: &'a GgufFile,
    mmap: &'a [u8],
}

impl LoadCtx<'_> {
    fn find(&self, name: &str) -> Result<&GgufTensorInfo, WeightLoadError> {
        self.gguf
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| WeightLoadError::MissingTensor(name.to_owned()))
    }

    fn has(&self, name: &str) -> bool {
        self.gguf.tensors.iter().any(|t| t.name == name)
    }

    fn dequant_tensor(
        &self,
        name: &str,
        expected_weights: usize,
    ) -> Result<Vec<f32>, WeightLoadError> {
        let tensor = self.find(name)?;
        let quant = tensor
            .quant_type()
            .ok_or_else(|| WeightLoadError::UnknownQuant {
                tensor: name.to_owned(),
                raw: tensor.raw_type_id,
            })?;
        let abs = tensor
            .absolute_offset(self.gguf.tensor_data_offset)
            .ok_or_else(|| WeightLoadError::OffsetOverflow {
                tensor: name.to_owned(),
            })? as usize;
        let src_len = byte_length(tensor, quant)
            .ok_or_else(|| WeightLoadError::SizeOverflow {
                tensor: name.to_owned(),
            })?;
        let end = abs
            .checked_add(src_len)
            .ok_or_else(|| WeightLoadError::OffsetOverflow {
                tensor: name.to_owned(),
            })?;
        if end > self.mmap.len() {
            return Err(WeightLoadError::TensorTruncated {
                tensor: name.to_owned(),
                need: end,
                have: self.mmap.len(),
            });
        }
        let src = &self.mmap[abs..end];

        let got_weights = dequant::weight_count(quant, src_len)
            .ok_or(WeightLoadError::Dequant(DequantError::Unsupported { quant }))?;
        if got_weights != expected_weights {
            return Err(WeightLoadError::WrongWeightCount {
                tensor: name.to_owned(),
                got: got_weights,
                expected: expected_weights,
            });
        }

        let mut dst = vec![0.0_f32; got_weights];
        dequant::dequantize_row_into(quant, src, &mut dst)
            .map_err(WeightLoadError::Dequant)?;
        Ok(dst)
    }
}

/// Byte length for one tensor's packed data.
fn byte_length(tensor: &GgufTensorInfo, quant: GgufQuantType) -> Option<usize> {
    let elems = usize::try_from(tensor.element_count()).ok()?;
    match quant {
        GgufQuantType::F32 => elems.checked_mul(4),
        GgufQuantType::F16 => elems.checked_mul(2),
        GgufQuantType::Q8_0 => {
            // elems must be a multiple of 32.
            if !elems.is_multiple_of(32) {
                return None;
            }
            let blocks = elems / 32;
            blocks.checked_mul(34)
        }
        _ => None,
    }
}

#[derive(Debug, Error)]
pub enum WeightLoadError {
    #[error("gguf parse: {0}")]
    Parse(String),
    #[error("gguf io: {0}")]
    Io(String),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("tensor {tensor}: unknown quant type (raw {raw})")]
    UnknownQuant { tensor: String, raw: u32 },
    #[error("tensor {tensor}: offset overflow computing absolute byte range")]
    OffsetOverflow { tensor: String },
    #[error("tensor {tensor}: size overflow computing packed byte length")]
    SizeOverflow { tensor: String },
    #[error("tensor {tensor}: packed bytes past end of mmap (need {need}, have {have})")]
    TensorTruncated {
        tensor: String,
        need: usize,
        have: usize,
    },
    #[error("tensor {tensor}: weight-count mismatch (got {got}, expected {expected})")]
    WrongWeightCount {
        tensor: String,
        got: usize,
        expected: usize,
    },
    #[error(transparent)]
    Dequant(DequantError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::config::LlamaConfig;
    use crate::gguf::parser::parse_path;
    use std::path::Path;

    const SMOLLM2: &str = "models/smollm2-135m-f16.gguf";

    #[test]
    fn smollm2_config_matches_known_shape() {
        if !Path::new(SMOLLM2).exists() {
            eprintln!("skipping: {SMOLLM2} not present");
            return;
        }
        let gguf = parse_path(SMOLLM2).unwrap();
        let cfg = LlamaConfig::from_gguf(&gguf).unwrap();
        assert_eq!(cfg.vocab_size, 49_152);
        assert_eq!(cfg.hidden_dim, 576);
        assert_eq!(cfg.ffn_dim, 1_536);
        assert_eq!(cfg.n_layers, 30);
        assert_eq!(cfg.n_heads, 9);
        assert_eq!(cfg.n_kv_heads, 3);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.rope_dim, 64);
        assert_eq!(cfg.rope_freq_base, 100_000.0);
    }

    #[test]
    fn smollm2_weights_load_all_tensors() {
        if !Path::new(SMOLLM2).exists() {
            eprintln!("skipping: {SMOLLM2} not present");
            return;
        }
        let gguf = parse_path(SMOLLM2).unwrap();
        let cfg = LlamaConfig::from_gguf(&gguf).unwrap();
        let weights = LlamaWeights::load(SMOLLM2, &cfg).unwrap();

        assert_eq!(weights.embedding.len(), cfg.vocab_size * cfg.hidden_dim);
        assert_eq!(weights.final_norm.len(), cfg.hidden_dim);
        assert_eq!(weights.blocks.len(), cfg.n_layers);

        // Tied LM head: SmolLM2 has no `output.weight`.
        assert!(Arc::ptr_eq(&weights.embedding, &weights.lm_head));

        // Spot-check block 0 shapes.
        let b0 = &weights.blocks[0];
        let q_width = cfg.n_heads * cfg.head_dim;
        let kv_width = cfg.n_kv_heads * cfg.head_dim;
        assert_eq!(b0.attn_norm.len(), cfg.hidden_dim);
        assert_eq!(b0.wq.len(), q_width * cfg.hidden_dim);
        assert_eq!(b0.wk.len(), kv_width * cfg.hidden_dim);
        assert_eq!(b0.wv.len(), kv_width * cfg.hidden_dim);
        assert_eq!(b0.wo.len(), cfg.hidden_dim * q_width);
        assert_eq!(b0.ffn_norm.len(), cfg.hidden_dim);
        assert_eq!(b0.w_gate.len(), cfg.ffn_dim * cfg.hidden_dim);
        assert_eq!(b0.w_up.len(), cfg.ffn_dim * cfg.hidden_dim);
        assert_eq!(b0.w_down.len(), cfg.hidden_dim * cfg.ffn_dim);

        // Structural sanity on the dequantized values of one weight.
        assert!(b0.wq.iter().all(|v| v.is_finite()));
        let max_abs = b0.wq.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
        assert!(max_abs > 0.0 && max_abs < 10.0);
    }

    #[test]
    fn smollm2_block_storage_view_matches_block_weights_shape() {
        // `BlockStorage::view()` should produce a `BlockWeights`
        // that `block::forward_block` will accept without panicking
        // on shape assertions.
        use crate::forward::block::{BlockConfig, BlockScratch, forward_block};

        if !Path::new(SMOLLM2).exists() {
            eprintln!("skipping: {SMOLLM2} not present");
            return;
        }
        let gguf = parse_path(SMOLLM2).unwrap();
        let cfg = LlamaConfig::from_gguf(&gguf).unwrap();
        let weights = LlamaWeights::load(SMOLLM2, &cfg).unwrap();

        let block_cfg = BlockConfig {
            hidden: cfg.hidden_dim,
            ffn_dim: cfg.ffn_dim,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            rope_freq_base: cfg.rope_freq_base,
            rope_dim: cfg.rope_dim,
            norm_eps: cfg.norm_eps,
        };
        let seq = 2;
        let mut scratch = BlockScratch::new(&block_cfg, seq);
        let mut x = vec![0.01_f32; seq * cfg.hidden_dim];
        let view = weights.blocks[0].view();
        forward_block(&mut x, &block_cfg, &view, seq, 0, &mut scratch);
        // Just verify the output is finite — the correctness gate
        // is A8. Running one real block against real weights checks
        // every shape lines up.
        assert!(x.iter().all(|v| v.is_finite()));
    }
}
