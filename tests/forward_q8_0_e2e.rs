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
const SMOLLM2_F16: &str = "models/smollm2-135m-f16.gguf";

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
#[ignore = "diagnostic: loads F16 + Q8_0 SmolLM2, runs perplexity on the same \
            128-token prose through both, prints the Q8_0/F16 ratio. ~18 s on \
            a release build post-rayon-matmul (89605c1). Run with --ignored \
            --nocapture when sanity-checking quant fidelity."]
fn q8_0_vs_f16_perplexity_diagnostic() {
    // Diagnostic — not a gate. Tells us whether Q8_0 PPL is in
    // the same ballpark as F16 (good — quantization noise) or
    // catastrophically worse (bad — dequant bug). 19-token prose
    // was too noisy to be meaningful; 128 tokens tightens the
    // signal substantially.
    if !Path::new(SMOLLM2_F16).exists() || !Path::new(SMOLLM2_Q8_0).exists() {
        eprintln!("skipping: need both F16 + Q8_0 SmolLM2");
        return;
    }
    let f16 = ForwardModel::load(SMOLLM2_F16).expect("load f16");
    let q8 = ForwardModel::load(SMOLLM2_Q8_0).expect("load q8_0");
    // In-distribution English prose, long enough to tokenize past
    // 128 tokens with SmolLM2's BPE (roughly 1.3 tokens/word).
    let text = "The invention of the printing press in fifteenth-century \
                Europe fundamentally altered the trajectory of human \
                communication. Before Gutenberg, books were copied by hand, \
                an expensive and error-prone process that restricted \
                literacy to a narrow clergy and aristocracy. Mechanical \
                type changed both the economics and the reliability of the \
                written word. Within a generation, presses operated in \
                every major European city, producing bibles, pamphlets, \
                scientific treatises, and eventually newspapers. Ideas \
                that once required years to circulate across the continent \
                could now traverse it in weeks. Scholars could assume \
                their colleagues shared a common text rather than a family \
                of divergent manuscripts. Reformers, from Luther to \
                Galileo, exploited the new medium to challenge \
                institutions that had relied on controlling access to \
                information. Literacy rates climbed, vernacular languages \
                flourished at the expense of Latin, and the public sphere \
                — the space of informed debate — emerged as a recognizable \
                feature of political life.";
    let tokens_f16 = f16.encode(text).expect("encode f16");
    let tokens_q8 = q8.encode(text).expect("encode q8");
    assert_eq!(
        tokens_f16, tokens_q8,
        "F16 and Q8_0 should tokenize identically — different ids would invalidate \
         a direct PPL comparison",
    );
    assert!(
        tokens_f16.len() >= 128,
        "diagnostic passage tokenized to {} tokens, want ≥ 128 for a stable PPL",
        tokens_f16.len(),
    );
    let ctx_len = 128;
    let ppl_f16 = f16.perplexity(&tokens_f16[..ctx_len], ctx_len).expect("ppl f16");
    let ppl_q8 = q8.perplexity(&tokens_q8[..ctx_len], ctx_len).expect("ppl q8_0");
    eprintln!("F16  ppl on {ctx_len} tokens: {ppl_f16:.3}");
    eprintln!("Q8_0 ppl on {ctx_len} tokens: {ppl_q8:.3}");
    eprintln!(
        "Q8_0/F16 ratio: {:.3} (≈1.0 = quantization-clean; \
         ≫1 = decode regression)",
        ppl_q8 / ppl_f16,
    );
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
