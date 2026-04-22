//! Perplexity harness — the A8 PPL gate.
//!
//! Runs the model over a pre-tokenized stream in non-overlapping
//! chunks of `ctx_len` tokens each. For each chunk, predicts
//! positions `[1..chunk_len]` from the logits at positions
//! `[0..chunk_len-1]`, accumulates cross-entropy, and at the end
//! returns `exp(mean(-log p_target))` — the standard perplexity
//! metric llama-perplexity reports.
//!
//! Implementation notes:
//!
//! - Chunks use a fresh KV cache (`cache.clear()` between chunks).
//!   This matches llama-perplexity's default behavior: each chunk
//!   is scored *at the specified context length*, not with an
//!   accumulated history.
//! - Log-softmax is computed numerically-stably per-position
//!   (subtract max, then `logsumexp`).
//! - The last partial chunk (fewer than 2 tokens) is skipped —
//!   there's nothing to evaluate.

use thiserror::Error;

use crate::forward::kv_cache::KvCache;
use crate::forward::model::{ForwardModel, ModelScratch};

#[derive(Debug, Error)]
pub enum PerplexityError {
    #[error("ctx_len must be ≥ 2 (got {0}) — need at least one prediction")]
    CtxLenTooSmall(usize),
    #[error("tokens has no evaluatable positions: need ≥ 2 tokens, got {0}")]
    TokensTooShort(usize),
    #[error("token id {id} ≥ vocab size {vocab}")]
    OutOfVocab { id: u32, vocab: usize },
}

/// Mean perplexity over `tokens`. `ctx_len` sets the max context
/// window per chunk — callers should match it to the `n_ctx` used
/// when generating the reference (typical llama-perplexity default
/// is 512 or 2048).
pub fn perplexity(
    model: &ForwardModel,
    tokens: &[u32],
    ctx_len: usize,
) -> Result<f32, PerplexityError> {
    if ctx_len < 2 {
        return Err(PerplexityError::CtxLenTooSmall(ctx_len));
    }
    if tokens.len() < 2 {
        return Err(PerplexityError::TokensTooShort(tokens.len()));
    }
    let vocab = model.config.vocab_size;
    for &t in tokens {
        if (t as usize) >= vocab {
            return Err(PerplexityError::OutOfVocab { id: t, vocab });
        }
    }

    let mut cache = KvCache::new(&model.config, ctx_len);
    let mut scratch = ModelScratch::new(&model.config, ctx_len, ctx_len);

    let mut total_nll = 0.0_f64;
    let mut n_eval: u64 = 0;

    // Non-overlapping chunks of up to `ctx_len` tokens.
    let mut chunk_start = 0;
    while chunk_start + 1 < tokens.len() {
        let end = (chunk_start + ctx_len).min(tokens.len());
        let chunk = &tokens[chunk_start..end];
        if chunk.len() < 2 {
            break;
        }

        cache.clear();
        let logits = model.forward_all_logits(chunk, &mut cache, &mut scratch);
        debug_assert_eq!(logits.len(), chunk.len() * vocab);

        // Position i in `chunk` (1..chunk.len()) is predicted by
        // the logits at position i-1.
        for i in 1..chunk.len() {
            let target = chunk[i] as usize;
            let row = &logits[(i - 1) * vocab..i * vocab];
            let log_p = log_softmax_at(row, target);
            total_nll += -log_p as f64;
            n_eval += 1;
        }

        chunk_start = end;
    }

    let mean_nll = (total_nll / n_eval as f64) as f32;
    Ok(mean_nll.exp())
}

/// `log_softmax(x)[target]` — numerically stable via subtract-max
/// and `logsumexp`. Equivalent to
/// `x[target] - (max + log(Σ exp(x - max)))`.
fn log_softmax_at(x: &[f32], target: usize) -> f32 {
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum_exp = 0.0_f64;
    for &v in x.iter() {
        sum_exp += ((v - max) as f64).exp();
    }
    let lse = max as f64 + sum_exp.ln();
    (x[target] as f64 - lse) as f32
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::forward::model::ForwardModel;

    const SMOLLM2: &str = "models/smollm2-135m-f16.gguf";

    fn model_or_skip() -> Option<ForwardModel> {
        if !Path::new(SMOLLM2).exists() {
            eprintln!("skipping: {SMOLLM2} not present");
            return None;
        }
        Some(ForwardModel::load(SMOLLM2).expect("load smollm2"))
    }

    #[test]
    fn log_softmax_at_matches_hand_compute() {
        // Input [0, ln 2, ln 3] → softmax = [1/6, 1/3, 1/2] →
        // log_softmax = [-ln 6, -ln 3, -ln 2].
        let x = [0.0_f32, 2.0_f32.ln(), 3.0_f32.ln()];
        assert!((log_softmax_at(&x, 0) - -6.0_f32.ln()).abs() < 1e-6);
        assert!((log_softmax_at(&x, 1) - -3.0_f32.ln()).abs() < 1e-6);
        assert!((log_softmax_at(&x, 2) - -2.0_f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn log_softmax_at_is_numerically_stable_on_large_inputs() {
        // Magnitude-1000 inputs: naive softmax would explode, but
        // the subtract-max + f64 sum_exp + log path stays finite.
        let x = [1000.0_f32, 1000.0, 1000.0];
        let lp = log_softmax_at(&x, 1);
        assert!(lp.is_finite());
        assert!((lp - (1.0_f32 / 3.0_f32).ln()).abs() < 1e-6);
    }

    #[test]
    fn ppl_errors_on_tiny_inputs() {
        let Some(model) = model_or_skip() else {
            return;
        };
        let err = perplexity(&model, &[1], 8).unwrap_err();
        assert!(matches!(err, PerplexityError::TokensTooShort(_)));
        let err = perplexity(&model, &[1, 2, 3], 1).unwrap_err();
        assert!(matches!(err, PerplexityError::CtxLenTooSmall(_)));
    }

    #[test]
    fn ppl_rejects_out_of_vocab_token() {
        let Some(model) = model_or_skip() else {
            return;
        };
        let vocab = model.config.vocab_size as u32;
        let err = perplexity(&model, &[1, 2, vocab], 8).unwrap_err();
        assert!(matches!(err, PerplexityError::OutOfVocab { .. }));
    }

    #[test]
    fn ppl_on_small_slice_is_finite_and_in_sane_range() {
        // Real-weight structural test: over a short sequence of
        // real tokens, PPL should be finite and plausibly low for
        // a trained LM (< vocab size by orders of magnitude, > 1).
        // No reference-value comparison here — that's the
        // ignored llama-perplexity gate.
        let Some(model) = model_or_skip() else {
            return;
        };
        // "Hello, world!" encoded with BOS prepended — a short
        // ~5-token sequence we know appears in training-like text.
        let tokens: Vec<u32> = vec![1, 19556, 28, 905, 17];
        let ppl = perplexity(&model, &tokens, 8).unwrap();
        assert!(ppl.is_finite(), "ppl not finite: {ppl}");
        // Upper bound: if the model is completely useless, PPL ≈
        // vocab_size ≈ 49 152. Anything below 10 000 is "the
        // forward pass isn't obviously broken".
        assert!(
            ppl > 1.0 && ppl < 10_000.0,
            "ppl {ppl} is suspicious (expected 1 < ppl < 10k for trained LM)",
        );
    }
}
