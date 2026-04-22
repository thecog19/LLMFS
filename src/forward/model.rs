//! Full-model forward pass: tokens → logits.
//!
//! [`ForwardModel::forward`] runs a batch of `seq_len` tokens
//! through:
//!
//! 1. embedding lookup per token,
//! 2. N transformer blocks (each appends to its layer's KV cache
//!    and attends over the full window 0 .. cache.current_len +
//!    seq_len),
//! 3. final RMSNorm,
//! 4. LM-head matmul on the *last* position only (we don't need
//!    per-token logits for perplexity — that's A8's concern when
//!    it decides what to evaluate).
//!
//! Returns `[vocab_size]` unnormalized logits for the last token
//! in the batch. Caller softmaxes or argmaxes as needed.

use std::path::Path;

use thiserror::Error;

use crate::forward::block::{
    BlockConfig, BlockObserver, BlockScratch, BlockWeights, NoopObserver, forward_block,
};
use crate::forward::config::{ConfigError, LlamaConfig};
use crate::forward::kv_cache::KvCache;
use crate::forward::ops::{embed, matmul, rmsnorm};
use crate::forward::perplexity::{PerplexityError, perplexity};
use crate::forward::tokenizer::{DecodeError, EncodeError, Tokenizer, TokenizerError};
use crate::forward::weights::{LlamaWeights, WeightLoadError};
use crate::gguf::parser::{ParseError, parse_path};

/// Dequantized llama-arch model ready to forward. Bundles the
/// tokenizer alongside the weights so callers can go from `&str`
/// to logits in two method calls.
pub struct ForwardModel {
    pub config: LlamaConfig,
    pub weights: LlamaWeights,
    pub tokenizer: Tokenizer,
    block_cfg: BlockConfig,
}

/// Reusable scratch for one forward call. One `BlockScratch` plus
/// the `[seq, hidden]` activation buffer, a post-norm buffer sized
/// for either the last position only (`forward`) or every position
/// (`forward_all_logits`), and the logits buffer.
pub struct ModelScratch {
    pub x: Vec<f32>,           // [batch, hidden]
    pub norm: Vec<f32>,        // [batch, hidden] — per-position final norm
    pub logits: Vec<f32>,      // [batch, vocab_size]
    pub block_scratch: BlockScratch,
    batch: usize,
    vocab_size: usize,
    hidden_dim: usize,
}

impl ModelScratch {
    pub fn new(cfg: &LlamaConfig, batch: usize, max_ctx: usize) -> Self {
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
        Self {
            x: vec![0.0; batch * cfg.hidden_dim],
            norm: vec![0.0; batch * cfg.hidden_dim],
            logits: vec![0.0; batch * cfg.vocab_size],
            block_scratch: BlockScratch::new(&block_cfg, batch, max_ctx),
            batch,
            vocab_size: cfg.vocab_size,
            hidden_dim: cfg.hidden_dim,
        }
    }
}

#[derive(Debug, Error)]
pub enum ModelLoadError {
    #[error("gguf parse: {0}")]
    Parse(#[from] ParseError),
    #[error("config: {0}")]
    Config(#[from] ConfigError),
    #[error("weights: {0}")]
    Weights(#[from] WeightLoadError),
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] TokenizerError),
}

impl ForwardModel {
    /// Load + dequantize a GGUF file, including its tokenizer.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ModelLoadError> {
        let path = path.as_ref();
        let gguf = parse_path(path)?;
        let config = LlamaConfig::from_gguf(&gguf)?;
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        let weights = LlamaWeights::load(path, &config)?;
        let block_cfg = BlockConfig {
            hidden: config.hidden_dim,
            ffn_dim: config.ffn_dim,
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            rope_freq_base: config.rope_freq_base,
            rope_dim: config.rope_dim,
            norm_eps: config.norm_eps,
        };
        Ok(Self {
            config,
            weights,
            tokenizer,
            block_cfg,
        })
    }

    /// Tokenize `text` via the model's embedded tokenizer. BOS /
    /// EOS handling follows the tokenizer config (see
    /// [`Tokenizer::encode`]).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, EncodeError> {
        self.tokenizer.encode(text)
    }

    /// Decode token ids back to a string.
    pub fn decode(&self, ids: &[u32]) -> Result<String, DecodeError> {
        self.tokenizer.decode(ids)
    }

    /// Perplexity on a pre-tokenized stream at the given context
    /// length. Non-overlapping chunks of `ctx_len` tokens each;
    /// cross-entropy accumulated, exponentiated at the end.
    pub fn perplexity(
        &self,
        tokens: &[u32],
        ctx_len: usize,
    ) -> Result<f32, PerplexityError> {
        perplexity(self, tokens, ctx_len)
    }

    /// Run the forward pass on `tokens`. Appends to `cache` and
    /// returns logits for the last token. Panics if
    /// `cache.layers.len()` doesn't match the model.
    pub fn forward<'a>(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        scratch: &'a mut ModelScratch,
    ) -> &'a [f32] {
        let seq_len = self.forward_common(tokens, cache, scratch, &mut NoopObserver);
        let hidden = self.config.hidden_dim;

        // Final RMSNorm on the LAST position only.
        let last_row_start = (seq_len - 1) * hidden;
        let dst = &mut scratch.norm[..hidden];
        dst.copy_from_slice(&scratch.x[last_row_start..last_row_start + hidden]);
        rmsnorm(dst, &self.weights.final_norm, self.config.norm_eps);

        // LM-head projection: logits = last_norm @ lm_head.T.
        matmul(
            &scratch.norm[..hidden],
            &self.weights.lm_head,
            &mut scratch.logits[..self.config.vocab_size],
            1,
            hidden,
            self.config.vocab_size,
        );

        &scratch.logits[..self.config.vocab_size]
    }

    /// Same forward, but returns logits for *every* position in the
    /// batch: `[seq_len, vocab_size]` row-major. Used by the
    /// perplexity harness (A8) — next-token decoding only needs
    /// [`Self::forward`]'s last-position output.
    pub fn forward_all_logits<'a>(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        scratch: &'a mut ModelScratch,
    ) -> &'a [f32] {
        self.forward_all_logits_with_observer(tokens, cache, scratch, &mut NoopObserver)
    }

    /// Same as [`Self::forward_all_logits`], but routes every
    /// matmul-input site through `observer`. Used by the AWQ
    /// collector during calibration (see [`crate::forward::awq`]).
    pub fn forward_all_logits_with_observer<'a>(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        scratch: &'a mut ModelScratch,
        observer: &mut dyn BlockObserver,
    ) -> &'a [f32] {
        let seq_len = self.forward_common(tokens, cache, scratch, observer);
        let hidden = self.config.hidden_dim;
        let vocab = self.config.vocab_size;

        // Final RMSNorm on every row.
        for i in 0..seq_len {
            let src = &scratch.x[i * hidden..(i + 1) * hidden];
            let dst = &mut scratch.norm[i * hidden..(i + 1) * hidden];
            dst.copy_from_slice(src);
            rmsnorm(dst, &self.weights.final_norm, self.config.norm_eps);
        }

        // LM-head: logits[i, :] = norm[i, :] @ lm_head.T, for all i.
        matmul(
            &scratch.norm[..seq_len * hidden],
            &self.weights.lm_head,
            &mut scratch.logits[..seq_len * vocab],
            seq_len,
            hidden,
            vocab,
        );
        &scratch.logits[..seq_len * vocab]
    }

    /// Embedding + N blocks. Common prefix of `forward` and
    /// `forward_all_logits`. Returns `seq_len` for the caller.
    fn forward_common(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        scratch: &mut ModelScratch,
        observer: &mut dyn BlockObserver,
    ) -> usize {
        let seq_len = tokens.len();
        assert!(seq_len > 0, "forward called with empty token batch");
        assert!(
            seq_len <= scratch.batch,
            "forward: seq_len {} > scratch batch {}",
            seq_len,
            scratch.batch,
        );
        assert_eq!(
            cache.layers.len(),
            self.config.n_layers,
            "cache layer count mismatch",
        );
        assert_eq!(scratch.vocab_size, self.config.vocab_size);
        assert_eq!(scratch.hidden_dim, self.config.hidden_dim);
        let hidden = self.config.hidden_dim;

        // Embedding lookup.
        for (i, &tok) in tokens.iter().enumerate() {
            let row = &mut scratch.x[i * hidden..(i + 1) * hidden];
            embed(tok, &self.weights.embedding, hidden, row);
        }

        // Transformer blocks.
        let x_slice = &mut scratch.x[..seq_len * hidden];
        for (layer_idx, block) in self.weights.blocks.iter().enumerate() {
            let weights: BlockWeights = block.view();
            forward_block(
                x_slice,
                &self.block_cfg,
                &weights,
                seq_len,
                &mut cache.layers[layer_idx],
                &mut scratch.block_scratch,
                layer_idx,
                observer,
            );
        }

        seq_len
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    const SMOLLM2: &str = "models/smollm2-135m-f16.gguf";

    fn model_or_skip() -> Option<ForwardModel> {
        if !Path::new(SMOLLM2).exists() {
            eprintln!("skipping: {SMOLLM2} not present");
            return None;
        }
        Some(ForwardModel::load(SMOLLM2).expect("load smollm2"))
    }

    #[test]
    fn forward_logits_have_vocab_shape_and_are_finite() {
        let Some(model) = model_or_skip() else {
            return;
        };
        let max_ctx = 8;
        let mut cache = KvCache::new(&model.config, max_ctx);
        let mut scratch = ModelScratch::new(&model.config, max_ctx, max_ctx);
        // Some arbitrary tokens — just need them to be inside the
        // vocab. The tokenizer's BOS is 1 for smollm2.
        let tokens: Vec<u32> = vec![1, 30, 42, 100];
        let logits = model.forward(&tokens, &mut cache, &mut scratch).to_vec();
        assert_eq!(logits.len(), model.config.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logit");
        // The logit distribution should be non-trivial (not all zero
        // or constant).
        let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max - min > 1.0,
            "logit spread {} collapsed — forward pass is degenerate",
            max - min,
        );
    }

    #[test]
    fn forward_is_deterministic() {
        let Some(model) = model_or_skip() else {
            return;
        };
        let max_ctx = 8;
        let tokens: Vec<u32> = vec![1, 30, 42, 100];

        let mut c1 = KvCache::new(&model.config, max_ctx);
        let mut s1 = ModelScratch::new(&model.config, max_ctx, max_ctx);
        let logits_a = model.forward(&tokens, &mut c1, &mut s1).to_vec();

        let mut c2 = KvCache::new(&model.config, max_ctx);
        let mut s2 = ModelScratch::new(&model.config, max_ctx, max_ctx);
        let logits_b = model.forward(&tokens, &mut c2, &mut s2).to_vec();
        assert_eq!(logits_a, logits_b, "forward is not deterministic");
    }

    #[test]
    fn softmax_of_logits_sums_to_one() {
        let Some(model) = model_or_skip() else {
            return;
        };
        let max_ctx = 8;
        let mut cache = KvCache::new(&model.config, max_ctx);
        let mut scratch = ModelScratch::new(&model.config, max_ctx, max_ctx);
        let tokens: Vec<u32> = vec![1, 30, 42, 100];
        let mut probs = model.forward(&tokens, &mut cache, &mut scratch).to_vec();
        crate::forward::ops::softmax(&mut probs);
        let sum: f32 = probs.iter().sum();
        // f32 accumulation drift over 49 152 values is ~1e-4 in
        // the worst case (each addition loses ~2^-23 of relative
        // precision). 1e-3 is a generous but meaningful bound.
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "softmax of logits didn't sum to 1: got {sum}",
        );
        // Sanity that softmax produced a real distribution.
        assert!(probs.iter().all(|p| *p >= 0.0 && *p <= 1.0));
    }

    #[test]
    fn kv_cache_split_matches_single_shot_logits() {
        // Full-model version of the A5 block-level coherence test.
        // If KV caching is wired wrong across even one of the 30
        // layers, last-token logits differ — this catches layer-
        // index mixups or cache-index drift.
        let Some(model) = model_or_skip() else {
            return;
        };
        let max_ctx = 8;
        let t_all: Vec<u32> = vec![1, 30, 42, 100];

        // Single-shot prefill.
        let mut c_single = KvCache::new(&model.config, max_ctx);
        let mut s_single = ModelScratch::new(&model.config, t_all.len(), max_ctx);
        let logits_single = model.forward(&t_all, &mut c_single, &mut s_single).to_vec();

        // Split: prefill [1, 30, 42], decode [100].
        let mut c_step = KvCache::new(&model.config, max_ctx);
        let mut s3 = ModelScratch::new(&model.config, 3, max_ctx);
        let _ = model.forward(&t_all[..3], &mut c_step, &mut s3);
        let mut s1 = ModelScratch::new(&model.config, 1, max_ctx);
        let logits_step = model.forward(&t_all[3..], &mut c_step, &mut s1).to_vec();

        assert_eq!(logits_single.len(), logits_step.len());
        let mut max_abs_diff = 0.0_f32;
        for (a, b) in logits_single.iter().zip(logits_step.iter()) {
            max_abs_diff = max_abs_diff.max((a - b).abs());
        }
        // F32 accumulation across 30 blocks picks up small rounding;
        // 1e-2 is a generous but meaningful bound (the logits
        // themselves span a range of tens).
        assert!(
            max_abs_diff < 1e-2,
            "KV cache incoherent: last-token logits diverged by {max_abs_diff}",
        );
    }
}
