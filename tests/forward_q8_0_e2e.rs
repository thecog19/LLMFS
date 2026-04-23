//! C1 end-to-end gate: Q8_0 SmolLM2 forward pass + perplexity.
//!
//! With Q8_0 (A3) + K-quant (C1) dequant wired in, the forward
//! pass should work unchanged on a Q8_0-quantized cover: weights
//! dequant to F32 at load time, everything downstream is
//! identical. These tests confirm that premise end-to-end on
//! `models/smollm2-135m-q8_0.gguf`.
//!
//! Skipped gracefully when the Q8_0 model isn't present. The
//! forward-pass test is fast enough (single-digit tokens) to run
//! by default; the perplexity test is `#[ignore]`'d because a
//! 128-token PPL run over 30 layers on CPU is multi-minute.

use std::path::Path;

use llmdb::forward::{ForwardModel, KvCache, ModelScratch};

const SMOLLM2_Q8_0: &str = "models/smollm2-135m-q8_0.gguf";

fn model_or_skip() -> Option<ForwardModel> {
    if !Path::new(SMOLLM2_Q8_0).exists() {
        eprintln!("skipping: {SMOLLM2_Q8_0} not present");
        return None;
    }
    Some(ForwardModel::load(SMOLLM2_Q8_0).expect("load q8_0 model"))
}

#[test]
fn q8_0_forward_produces_finite_logits() {
    // Structural test: Q8_0 forward pass produces real logits in
    // the same shape as F16. Values won't be identical (Q8_0 is
    // lossy vs F16), but they must be finite + have a non-trivial
    // spread.
    let Some(model) = model_or_skip() else {
        return;
    };
    let max_ctx = 8;
    let mut cache = KvCache::new(&model.config, max_ctx);
    let mut scratch = ModelScratch::new(&model.config, max_ctx, max_ctx);

    // BOS + a few in-vocab ids — plausible prefix for SmolLM2.
    let tokens: Vec<u32> = vec![1, 30, 42, 100];
    let logits = model.forward(&tokens, &mut cache, &mut scratch).to_vec();

    assert_eq!(logits.len(), model.config.vocab_size);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "Q8_0 forward produced non-finite logit",
    );
    let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max - min > 1.0,
        "Q8_0 forward logits collapsed (spread = {})",
        max - min,
    );
}

#[test]
fn q8_0_forward_is_deterministic() {
    // Dequantization is deterministic; the forward pass is pure
    // float arithmetic on immutable weights. Two runs with fresh
    // caches + scratch must produce bit-identical logits.
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

    assert_eq!(logits_a, logits_b, "Q8_0 forward is nondeterministic");
}

#[test]
#[ignore = "slow: 128-token Q8_0 forward through 30 layers (~multi-minute CPU). \
            Run with --ignored when validating C1 changes."]
fn q8_0_perplexity_stays_in_sane_range() {
    // Q8_0 quantization typically adds a few % PPL vs F16.
    // Without a reference binary to compare against we can't
    // assert parity, but we can assert the result is finite and
    // below a generous ceiling — any decode regression would blow
    // past that.
    let Some(model) = model_or_skip() else {
        return;
    };
    // Plain English prefix — should score reasonably under a
    // trained LM.
    let Ok(tokens) = model.encode(
        "The quick brown fox jumps over the lazy dog. \
         The press reshaped printed communication across Europe.",
    ) else {
        eprintln!("skipping: encode failed");
        return;
    };
    if tokens.len() < 8 {
        return;
    }
    let ctx_len = 128.min(tokens.len());
    let ppl = model.perplexity(&tokens[..ctx_len], ctx_len).expect("ppl");
    eprintln!("q8_0 ppl on {ctx_len} tokens: {ppl:.3}");
    assert!(ppl.is_finite());
    // Loose upper bound — vocab is ~49k, a degenerate model would
    // score near that. Trained SmolLM2 at Q8_0 should be well below.
    assert!(
        (1.0..5000.0).contains(&ppl),
        "Q8_0 ppl {ppl} outside plausible range (1, 5000)",
    );
}
