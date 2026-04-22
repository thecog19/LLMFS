//! Integration tests for `llmdb::forward::tokenizer::TokenizerConfig`.
//!
//! Parses the tokenizer metadata out of real GGUFs and sanity-checks
//! the shape. Skipped gracefully when the model files aren't present
//! (they're gitignored — see `.gitignore`'s `/models/*` rule).

use std::path::Path;

use llmdb::forward::tokenizer::{TokenizerConfig, TokenizerModel};
use llmdb::gguf::parser::parse_path;

const SMOLLM2: &str = "models/smollm2-135m-f16.gguf";
const QWEN25: &str = "models/qwen2.5-0.5b-instruct-f16.gguf";

fn parse_or_skip(path: &str) -> Option<TokenizerConfig> {
    if !Path::new(path).exists() {
        eprintln!("skipping: {path} not present");
        return None;
    }
    let gguf = parse_path(path).expect("parse gguf");
    Some(TokenizerConfig::from_gguf(&gguf).expect("parse tokenizer"))
}

#[test]
fn smollm2_tokenizer_parses_with_expected_shape() {
    let Some(tk) = parse_or_skip(SMOLLM2) else {
        return;
    };
    assert_eq!(tk.model, TokenizerModel::Gpt2, "smollm2 uses gpt2 tokenizer");
    assert_eq!(tk.pre_tokenizer.as_deref(), Some("smollm"));
    assert_eq!(
        tk.vocab_size(),
        49152,
        "smollm2 vocab is 49_152 per its config"
    );
    assert_eq!(
        tk.token_types.len(),
        tk.vocab_size(),
        "token_type array must match vocab length"
    );
    assert!(
        tk.scores.is_none(),
        "gpt2 tokenizer has no scores (SentencePiece-only field)"
    );
    assert_eq!(tk.merges.len(), 48_900, "smollm2 ships 48 900 BPE merges");
    // Sanity on the first merge: no empty sides, no leftover spaces.
    for (l, r) in tk.merges.iter().take(5) {
        assert!(!l.is_empty() && !r.is_empty());
        assert!(!l.contains(' ') && !r.contains(' '));
    }
    // Special tokens are all set + non-negative (u32 is always so).
    assert_eq!(tk.special.bos, Some(1));
    assert_eq!(tk.special.eos, Some(2));
    assert_eq!(tk.special.padding, Some(2));
    assert_eq!(tk.special.unknown, Some(0));
    assert!(!tk.special.add_bos, "smollm2 sets add_bos_token = false");
}

#[test]
fn qwen25_tokenizer_parses_with_expected_shape() {
    let Some(tk) = parse_or_skip(QWEN25) else {
        return;
    };
    assert_eq!(tk.model, TokenizerModel::Gpt2);
    assert_eq!(tk.pre_tokenizer.as_deref(), Some("qwen2"));
    assert_eq!(tk.vocab_size(), 151_936);
    assert_eq!(tk.token_types.len(), tk.vocab_size());
    assert!(tk.scores.is_none());
    assert_eq!(tk.merges.len(), 151_387);
    assert_eq!(tk.special.bos, Some(151_643));
    assert_eq!(tk.special.eos, Some(151_645));
    assert_eq!(tk.special.padding, Some(151_643));
    assert!(!tk.special.add_bos);
}

#[test]
fn vocab_size_equals_tokens_len() {
    let Some(tk) = parse_or_skip(SMOLLM2) else {
        return;
    };
    assert_eq!(tk.vocab_size(), tk.tokens.len());
}

#[test]
fn special_token_ids_are_in_vocab() {
    let Some(tk) = parse_or_skip(SMOLLM2) else {
        return;
    };
    let vocab = tk.vocab_size() as u32;
    if let Some(id) = tk.special.bos {
        assert!(id < vocab, "bos id {id} out of vocab (size {vocab})");
    }
    if let Some(id) = tk.special.eos {
        assert!(id < vocab, "eos id {id} out of vocab (size {vocab})");
    }
    if let Some(id) = tk.special.padding {
        assert!(id < vocab, "padding id {id} out of vocab (size {vocab})");
    }
    if let Some(id) = tk.special.unknown {
        assert!(id < vocab, "unknown id {id} out of vocab (size {vocab})");
    }
}
