//! Integration tests for `llmdb::forward::tokenizer`.
//!
//! Parses the tokenizer metadata out of real GGUFs and sanity-checks
//! the shape + decode path. Skipped gracefully when the model files
//! aren't present (they're gitignored — see `.gitignore`'s
//! `/models/*` rule).

use std::path::Path;

use llmdb::forward::tokenizer::{Tokenizer, TokenizerConfig, TokenizerModel};
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
    assert_eq!(
        tk.model,
        TokenizerModel::Gpt2,
        "smollm2 uses gpt2 tokenizer"
    );
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

// ─── A2a: build Tokenizer + decode round-trips against real vocab ─────────

fn tokenizer_or_skip(path: &str) -> Option<Tokenizer> {
    if !Path::new(path).exists() {
        eprintln!("skipping: {path} not present");
        return None;
    }
    let gguf = parse_path(path).expect("parse gguf");
    Some(Tokenizer::from_gguf(&gguf).expect("build tokenizer"))
}

#[test]
fn smollm2_tokenizer_builds_and_has_full_vocab_map() {
    let Some(t) = tokenizer_or_skip(SMOLLM2) else {
        return;
    };
    assert_eq!(t.config().vocab_size(), 49_152);
    // Every token in the vocab must be reachable by string lookup.
    // 49k lookups is fast enough for a fast test.
    for (id, tok) in t.config().tokens.iter().enumerate() {
        assert_eq!(
            t.token_id(tok),
            Some(id as u32),
            "vocab_map disagreed with tokens[{id}] = {tok:?}",
        );
    }
}

#[test]
fn smollm2_decode_single_printable_ascii_byte_tokens_round_trip() {
    // A byte-level BPE vocab only lets you *decode* single-byte
    // tokens in isolation when the resulting byte is valid UTF-8
    // by itself — i.e., ASCII bytes 0..=127. Anything ≥ 128 is
    // either a UTF-8 continuation byte (invalid as a start) or a
    // leading byte whose completion lives in a neighboring token;
    // decoding it alone surfaces as `DecodeError::InvalidUtf8`,
    // which is correct behavior for UTF-8-strict decoding.
    //
    // So we test the only range where single-byte decode is
    // meaningful: printable ASCII (`33..=126`, space-through-
    // tilde-minus-space). Every byte-level BPE vocab has to
    // cover this or it can't encode English text.
    let Some(t) = tokenizer_or_skip(SMOLLM2) else {
        return;
    };

    for b in 33u8..=126 {
        let char_str: String = std::iter::once(byte_to_char_test(b)).collect();
        let Some(id) = t.token_id(&char_str) else {
            panic!("printable-ASCII byte {b} ({char_str:?}) missing from smollm2 vocab",);
        };
        let decoded = t
            .decode(&[id])
            .unwrap_or_else(|e| panic!("decode byte {b}: {e}"));
        assert_eq!(
            decoded.as_bytes(),
            &[b],
            "byte {b} failed round-trip via single-token decode",
        );
    }
}

#[test]
fn smollm2_decode_high_byte_pair_round_trips_via_utf8_boundary() {
    // Two-byte UTF-8 codepoints decode as a *pair* of byte-level
    // tokens. Pick "Ą" (U+0104, UTF-8 `0xC4 0x84`) as a
    // representative: its byte-level BPE form is two single-byte
    // tokens, one per source byte. Decoding them together must
    // produce "Ą".
    let Some(t) = tokenizer_or_skip(SMOLLM2) else {
        return;
    };
    let hi_char: String = std::iter::once(byte_to_char_test(0xC4)).collect();
    let lo_char: String = std::iter::once(byte_to_char_test(0x84)).collect();
    let Some(hi_id) = t.token_id(&hi_char) else {
        eprintln!("skipping: smollm2 vocab doesn't expose byte 0xC4 as a standalone token");
        return;
    };
    let Some(lo_id) = t.token_id(&lo_char) else {
        eprintln!("skipping: smollm2 vocab doesn't expose byte 0x84 as a standalone token");
        return;
    };
    let decoded = t.decode(&[hi_id, lo_id]).expect("decode Ą");
    assert_eq!(decoded, "Ą");
}

/// Local copy of the byte→char mapping — the real function is
/// `pub(crate)` in the library, not exposed to integration tests,
/// so we recompute it here for test purposes. Keep this in lock-
/// step with `src/forward/tokenizer.rs::byte_to_char`.
fn byte_to_char_test(b: u8) -> char {
    let self_mapped =
        (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
    if self_mapped {
        return char::from_u32(b as u32).unwrap();
    }
    let remapped: Vec<u8> = (0u16..256)
        .map(|x| x as u8)
        .filter(|byte| {
            !((33..=126).contains(byte) || (161..=172).contains(byte) || (174..=255).contains(byte))
        })
        .collect();
    let idx = remapped.iter().position(|&x| x == b).unwrap();
    char::from_u32(256 + idx as u32).unwrap()
}

#[test]
fn smollm2_decode_multi_token_string() {
    // Decode the tokenization of "Hello, world!" that we already
    // know from `llama-tokenize`: [19556, 28, 905, 17]. (Captured
    // during A0 manual verification.)
    let Some(t) = tokenizer_or_skip(SMOLLM2) else {
        return;
    };
    let decoded = t.decode(&[19556, 28, 905, 17]).expect("decode");
    assert_eq!(decoded, "Hello, world!");
}

// encode parity is exercised in depth by the 30-input fixture
// in `tests/forward_tokenizer_encode_parity.rs`; we don't duplicate
// it here.
