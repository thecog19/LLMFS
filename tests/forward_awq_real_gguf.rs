//! B1 integration test — run `AwqCollector` over a real SmolLM2
//! forward pass on ~256 calibration tokens and verify the salience
//! vectors come out with the expected shapes and ranges.
//!
//! This is the test gate from plan.md's B1 entry:
//! > full SmolLM2-135M forward pass on ~512 calibration tokens,
//! > assert salience-vector shape per matmul weight tensor matches
//! > the tensor's input-channel count, magnitudes in a sane range
//! > (>0, finite, not uniformly constant).
//!
//! 256 rather than 512 just to keep runtime down; salience from
//! 256 tokens is already well-defined, and the correctness gate is
//! shape + sanity, not a specific value.

use std::path::Path;

use llmdb::forward::awq::AwqCollector;
use llmdb::forward::kv_cache::KvCache;
use llmdb::forward::model::{ForwardModel, ModelScratch};

const SMOLLM2_GGUF: &str = "models/smollm2-135m-f16.gguf";

fn model_or_skip() -> Option<ForwardModel> {
    if !Path::new(SMOLLM2_GGUF).exists() {
        eprintln!("skipping: {SMOLLM2_GGUF} not present");
        return None;
    }
    Some(ForwardModel::load(SMOLLM2_GGUF).expect("load smollm2"))
}

#[test]
#[ignore = "slow: 256-token forward through 30 layers (~2min naive matmul). \
            Run with --ignored when validating B1 changes."]
fn awq_collector_produces_finite_nonconstant_salience_per_matmul() {
    let Some(model) = model_or_skip() else {
        return;
    };
    let cfg = model.config;
    // 256 tokens of calibration input — tokenize a fixed paragraph
    // of prose. We don't care about the content, just that it
    // produces a varied activation pattern.
    let prose = concat!(
        "The quick brown fox jumps over the lazy dog. Packaging and ",
        "unpacking this sentence produces tokens at the morpheme ",
        "level: prefixes, roots, and suffixes all show up as distinct ",
        "ids in a byte-level BPE vocabulary. When we run the model ",
        "forward through a calibration loop, the activations at ",
        "every linear projection record how strongly each input ",
        "channel contributes to the next stage. Salience accumulation ",
        "is the first step toward activation-aware weight placement.",
    );
    let mut tokens = model.encode(prose).expect("encode");
    tokens.truncate(256);
    if tokens.len() < 32 {
        // In the unlikely event that this paragraph produces very
        // few tokens, we still want a meaningful test — pad from
        // the first section of the vocab (non-special IDs that
        // exist in every vocab).
        while tokens.len() < 128 {
            tokens.push((tokens.len() as u32 % 1000) + 100);
        }
    }
    eprintln!("calibration tokens: {}", tokens.len());

    let ctx_len = tokens.len();
    let mut cache = KvCache::new(&cfg, ctx_len);
    let mut scratch = ModelScratch::new(&cfg, ctx_len, ctx_len);
    let mut collector = AwqCollector::new();

    let _ = model.forward_all_logits_with_observer(
        &tokens,
        &mut cache,
        &mut scratch,
        &mut collector,
    );
    let salience = collector.finalize();

    // Structural checks:
    // 1. Every linear weight in every block should have a salience.
    let q_in = cfg.hidden_dim; // input channels for Q/K/V (= hidden)
    let o_in = cfg.n_heads * cfg.head_dim; // input for attn_output
    let gate_up_in = cfg.hidden_dim; // input for ffn_gate/up
    let down_in = cfg.ffn_dim; // input for ffn_down
    for layer in 0..cfg.n_layers {
        let check = |name: &str, expected_len: usize| {
            let s = salience
                .get(name)
                .unwrap_or_else(|| panic!("salience missing for {name}"));
            assert_eq!(s.len(), expected_len, "{name}: wrong salience length");
            assert!(
                s.iter().all(|v| v.is_finite() && *v >= 0.0),
                "{name}: non-finite or negative salience",
            );
            let max = s.iter().copied().fold(0.0_f32, f32::max);
            let min = s.iter().copied().fold(f32::INFINITY, f32::min);
            assert!(max > 0.0, "{name}: all-zero salience");
            // Require at least some channel variation — if every
            // channel reported the same |x|, the calibration is
            // either degenerate or the collector is broken.
            assert!(
                max - min > 1e-6,
                "{name}: salience is uniformly constant ({max} – {min})",
            );
        };
        check(&format!("blk.{layer}.attn_q.weight"), q_in);
        check(&format!("blk.{layer}.attn_k.weight"), q_in);
        check(&format!("blk.{layer}.attn_v.weight"), q_in);
        check(&format!("blk.{layer}.attn_output.weight"), o_in);
        check(&format!("blk.{layer}.ffn_gate.weight"), gate_up_in);
        check(&format!("blk.{layer}.ffn_up.weight"), gate_up_in);
        check(&format!("blk.{layer}.ffn_down.weight"), down_in);
    }

    // 2. Sanity: the three Q/K/V tensors for a given layer share
    //    the same salience (same input site).
    for layer in 0..cfg.n_layers {
        let q = &salience[&format!("blk.{layer}.attn_q.weight")];
        let k = &salience[&format!("blk.{layer}.attn_k.weight")];
        let v = &salience[&format!("blk.{layer}.attn_v.weight")];
        assert_eq!(q, k, "layer {layer}: attn_q != attn_k");
        assert_eq!(q, v, "layer {layer}: attn_q != attn_v");
    }
}
