//! Tokenizer configuration parsed from GGUF metadata.
//!
//! Milestone A1 — types only. A2 builds the actual encoder/decoder
//! on top of this. Supports whatever tokenizer type our target
//! models declare via `tokenizer.ggml.model`; the primary
//! targets for Milestone A are both `gpt2` (byte-level BPE) per
//! Phase A0 discovery on
//! `models/smollm2-135m-f16.gguf` + `models/qwen2.5-0.5b-instruct-f16.gguf`.
//!
//! # GGUF keys this module consumes
//!
//! ```text
//! tokenizer.ggml.model            string   "gpt2" | "llama" | "bert" | ...
//! tokenizer.ggml.pre              string   pre-tokenizer variant (e.g. "smollm", "qwen2")
//! tokenizer.ggml.tokens           [string] vocabulary (indexed by token id)
//! tokenizer.ggml.scores           [float]  SentencePiece only; absent for gpt2
//! tokenizer.ggml.token_type       [int]    per-token type enum (1=normal, 3=control, ...)
//! tokenizer.ggml.merges           [string] BPE merges as "left right" pairs
//! tokenizer.ggml.bos_token_id     u32
//! tokenizer.ggml.eos_token_id     u32
//! tokenizer.ggml.padding_token_id u32
//! tokenizer.ggml.unknown_token_id u32
//! tokenizer.ggml.add_bos_token    bool
//! tokenizer.ggml.add_eos_token    bool
//! ```

use std::collections::HashMap;

use thiserror::Error;

use crate::gguf::parser::{GgufFile, GgufMetadataValue};

// ─── GPT-2 byte ↔ unicode mapping (A2a) ───────────────────────────────────
//
// Byte-level BPE needs to turn arbitrary bytes into characters a BPE
// merges table can key on. GPT-2's trick: reserve the printable-ASCII
// and Latin-1 ranges for themselves, and remap the "awkward" bytes
// (control chars, space, DEL, NBSP, etc.) to unused codepoints in the
// 256.. range. Deterministic, reversible, documented in the original
// OpenAI BPE code.
//
// Mapping rules:
//   bytes 33..=126 (printable ASCII minus space) → char == byte
//   bytes 161..=172                              → char == byte
//   bytes 174..=255                              → char == byte
//   all other bytes (0..=32, 127, 128..=160, 173) → char == 256 + offset,
//     where offset counts the remaining bytes in ascending order.

const fn byte_is_self_mapped(byte: u8) -> bool {
    (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || (byte >= 174)
}

/// The 68 "awkward" bytes that don't map to themselves.
const REMAPPED_BYTES: [u8; 68] = {
    let mut out = [0u8; 68];
    let mut i = 0usize;
    let mut b = 0u16;
    while b < 256 {
        let byte = b as u8;
        if !byte_is_self_mapped(byte) {
            out[i] = byte;
            i += 1;
        }
        b += 1;
    }
    assert!(i == 68);
    out
};

/// Map `byte → char` (u32 codepoint) as a compile-time table.
const BYTE_TO_CHAR: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut b = 0usize;
    while b < 256 {
        let byte = b as u8;
        if byte_is_self_mapped(byte) {
            table[b] = byte as u32;
        } else {
            // Find this byte's index in REMAPPED_BYTES and map to 256+idx.
            let mut i = 0usize;
            while i < REMAPPED_BYTES.len() {
                if REMAPPED_BYTES[i] == byte {
                    table[b] = 256 + i as u32;
                    break;
                }
                i += 1;
            }
        }
        b += 1;
    }
    table
};

/// Encode one byte as a single `char` per the GPT-2 mapping.
/// Currently exercised only by unit tests; the encoder in A2c
/// will be the first production caller.
#[allow(dead_code)]
pub(crate) fn byte_to_char(b: u8) -> char {
    // SAFETY: all values in `BYTE_TO_CHAR` are valid Unicode scalars
    // (< 0x110000 and not surrogates). Constructed at const time from
    // a deterministic table; checked in `byte_to_char_round_trip`.
    char::from_u32(BYTE_TO_CHAR[b as usize]).expect("BYTE_TO_CHAR invariant")
}

/// Inverse map: `char codepoint → byte`. Returns `None` when the
/// codepoint isn't one of the 256 we emit.
pub(crate) fn char_to_byte(c: char) -> Option<u8> {
    let cp = c as u32;
    for (b, mapped) in BYTE_TO_CHAR.iter().enumerate() {
        if *mapped == cp {
            return Some(b as u8);
        }
    }
    None
}

/// Tokenizer family declared by `tokenizer.ggml.model`. Maps
/// directly to the vocab-building + encode algorithm Milestone A2
/// will dispatch on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerModel {
    /// SentencePiece-style unigram / BPE with token scores and
    /// byte-fallback. Llama-1 / Llama-2 family.
    Llama,
    /// Byte-level BPE with regex pre-tokenization. GPT-2 family,
    /// also used by Llama-3 and every model we've inspected so
    /// far in this repo (SmolLM2, Qwen2.5).
    Gpt2,
    /// WordPiece. BERT family. Not currently targeted.
    Bert,
    /// Unknown tokenizer type — name is preserved so callers can
    /// surface an actionable error.
    Other(String),
}

impl TokenizerModel {
    fn from_str(s: &str) -> Self {
        match s {
            "llama" => Self::Llama,
            "gpt2" => Self::Gpt2,
            "bert" => Self::Bert,
            other => Self::Other(other.to_owned()),
        }
    }
}

/// Special-token IDs + add-at-encode flags. All IDs are optional;
/// a model may not declare every category.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpecialTokens {
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub padding: Option<u32>,
    pub unknown: Option<u32>,
    /// Whether the tokenizer should prepend BOS at encode time.
    pub add_bos: bool,
    /// Whether the tokenizer should append EOS at encode time.
    pub add_eos: bool,
}

/// Fully-parsed tokenizer configuration. All vocab arrays are
/// eagerly materialized from the GGUF metadata; encoding /
/// decoding happens on copies of this struct.
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// `tokenizer.ggml.model` — selects the encode/decode algorithm.
    pub model: TokenizerModel,
    /// `tokenizer.ggml.pre` — pre-tokenizer regex variant. Most
    /// gpt2-family GGUFs carry one; some older ones don't.
    pub pre_tokenizer: Option<String>,
    /// Vocabulary. `tokens[id]` is the surface form of token `id`.
    pub tokens: Vec<String>,
    /// SentencePiece unigram scores per token. `None` for
    /// gpt2-style BPE (it doesn't use scores).
    pub scores: Option<Vec<f32>>,
    /// Per-token type flag. 1 = normal, 3 = control, 4 = unused,
    /// 6 = byte, etc. — GGUF convention mirrored from llama.cpp.
    /// Same length as `tokens`.
    pub token_types: Vec<u8>,
    /// BPE merges, each as `(left, right)`. The GGUF stores them
    /// as space-separated strings; we split once at load time.
    /// Empty for tokenizer models that don't use merges.
    pub merges: Vec<(String, String)>,
    pub special: SpecialTokens,
}

impl TokenizerConfig {
    /// Parse a `TokenizerConfig` from a `GgufFile`. Fails when a
    /// required field is missing or has an unexpected type.
    ///
    /// "Required" means: `tokenizer.ggml.model`, `tokens`, and
    /// `token_type` must be present. Everything else has a sane
    /// default (empty, None, false).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, TokenizerError> {
        let model_str = read_string(gguf, "tokenizer.ggml.model")
            .ok_or(TokenizerError::MissingKey("tokenizer.ggml.model"))?;
        let model = TokenizerModel::from_str(model_str);

        let pre_tokenizer = read_string(gguf, "tokenizer.ggml.pre").map(str::to_owned);

        let tokens = read_string_array(gguf, "tokenizer.ggml.tokens")
            .ok_or(TokenizerError::MissingKey("tokenizer.ggml.tokens"))?;

        let scores = read_f32_array(gguf, "tokenizer.ggml.scores");

        let token_types = read_u8_array(gguf, "tokenizer.ggml.token_type")
            .ok_or(TokenizerError::MissingKey("tokenizer.ggml.token_type"))?;
        if token_types.len() != tokens.len() {
            return Err(TokenizerError::VocabSizeMismatch {
                tokens: tokens.len(),
                token_types: token_types.len(),
            });
        }
        if let Some(s) = &scores
            && s.len() != tokens.len()
        {
            return Err(TokenizerError::ScoresLengthMismatch {
                tokens: tokens.len(),
                scores: s.len(),
            });
        }

        let raw_merges = read_string_array(gguf, "tokenizer.ggml.merges").unwrap_or_default();
        let mut merges = Vec::with_capacity(raw_merges.len());
        for raw in raw_merges {
            merges.push(parse_merge(&raw)?);
        }

        let special = SpecialTokens {
            bos: read_u32(gguf, "tokenizer.ggml.bos_token_id"),
            eos: read_u32(gguf, "tokenizer.ggml.eos_token_id"),
            padding: read_u32(gguf, "tokenizer.ggml.padding_token_id"),
            unknown: read_u32(gguf, "tokenizer.ggml.unknown_token_id"),
            add_bos: read_bool(gguf, "tokenizer.ggml.add_bos_token").unwrap_or(false),
            add_eos: read_bool(gguf, "tokenizer.ggml.add_eos_token").unwrap_or(false),
        };

        Ok(Self {
            model,
            pre_tokenizer,
            tokens,
            scores,
            token_types,
            merges,
            special,
        })
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TokenizerError {
    #[error("required GGUF metadata key `{0}` is missing")]
    MissingKey(&'static str),

    #[error(
        "vocab size mismatch: `tokenizer.ggml.tokens` has {tokens} entries, `tokenizer.ggml.token_type` has {token_types}"
    )]
    VocabSizeMismatch { tokens: usize, token_types: usize },

    #[error(
        "scores length mismatch: `tokenizer.ggml.tokens` has {tokens} entries, `tokenizer.ggml.scores` has {scores}"
    )]
    ScoresLengthMismatch { tokens: usize, scores: usize },

    #[error(
        "malformed merge entry: expected two parts separated by a single ASCII space, got {0:?}"
    )]
    MalformedMerge(String),

    #[error(
        "unsupported tokenizer model `{0:?}` — Milestone A implements gpt2 only"
    )]
    UnsupportedModel(TokenizerModel),
}

// ─── Tokenizer (encode / decode handle) ───────────────────────────────────

/// A configured tokenizer ready to `encode` / `decode`.
///
/// Wraps [`TokenizerConfig`] with the precomputed lookup tables
/// encoders/decoders need (vocab-string → id, merge → priority rank).
/// Build once per model load; reuse across calls.
///
/// Milestone A targets gpt2-family (byte-level BPE) tokenizers
/// exclusively. Calling `Tokenizer::from_gguf` on a Llama /
/// SentencePiece GGUF fails with `UnsupportedModel` — a future
/// milestone handles that family.
#[derive(Debug)]
pub struct Tokenizer {
    config: TokenizerConfig,
    /// `vocab_map[token_str] = token_id`. Built from
    /// `config.tokens` at load time. Owns its keys.
    vocab_map: HashMap<String, u32>,
    /// Merge rank: `merge_ranks[(left, right)] = position in the
    /// merges list`. Lower rank = higher priority (applied first
    /// in BPE). Unused in A2a; populated here so A2c doesn't have
    /// to revisit load logic.
    #[allow(dead_code)]
    merge_ranks: HashMap<(String, String), u32>,
}

impl Tokenizer {
    /// Build a Tokenizer from a GGUF file. Equivalent to
    /// `Tokenizer::from_config(TokenizerConfig::from_gguf(gguf)?)`.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, TokenizerError> {
        Self::from_config(TokenizerConfig::from_gguf(gguf)?)
    }

    /// Build a Tokenizer from an already-parsed config. Lets tests
    /// drive the encoder/decoder with hand-crafted configs.
    pub fn from_config(config: TokenizerConfig) -> Result<Self, TokenizerError> {
        if config.model != TokenizerModel::Gpt2 {
            return Err(TokenizerError::UnsupportedModel(config.model.clone()));
        }

        let mut vocab_map = HashMap::with_capacity(config.tokens.len());
        for (id, token) in config.tokens.iter().enumerate() {
            // If the vocab has duplicate tokens (shouldn't happen
            // in practice), the lower id wins — matches llama.cpp.
            vocab_map.entry(token.clone()).or_insert(id as u32);
        }

        let merge_ranks: HashMap<(String, String), u32> = config
            .merges
            .iter()
            .enumerate()
            .map(|(rank, (l, r))| ((l.clone(), r.clone()), rank as u32))
            .collect();

        Ok(Self {
            config,
            vocab_map,
            merge_ranks,
        })
    }

    /// The underlying config (for callers that need tokenizer
    /// metadata — vocab size, special token IDs, etc.).
    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }

    /// Look up a token id by its surface form. Useful for tests
    /// and for special-token dispatch.
    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.vocab_map.get(s).copied()
    }

    /// Encode a string to a sequence of token ids.
    ///
    /// **A2a stub.** Returns `EncodeError::Unimplemented` for any
    /// non-trivial input; wired through so the API shape is stable
    /// while A2b (pre-tokenization) and A2c (BPE merges) fill in
    /// the algorithm.
    pub fn encode(&self, _text: &str) -> Result<Vec<u32>, EncodeError> {
        Err(EncodeError::Unimplemented)
    }

    /// Decode a sequence of token ids back to a UTF-8 string.
    ///
    /// Walks the ids, looks up each token's surface form in the
    /// vocab, concatenates the characters, maps each character
    /// back to its source byte via the GPT-2 byte↔unicode table,
    /// and decodes the resulting byte sequence as UTF-8.
    pub fn decode(&self, ids: &[u32]) -> Result<String, DecodeError> {
        let mut bytes = Vec::with_capacity(ids.len() * 2);
        for &id in ids {
            let idx = id as usize;
            let token = self
                .config
                .tokens
                .get(idx)
                .ok_or(DecodeError::OutOfVocabulary(id))?;
            for c in token.chars() {
                let b = char_to_byte(c).ok_or(DecodeError::InvalidTokenChar {
                    id,
                    char: c,
                })?;
                bytes.push(b);
            }
        }
        String::from_utf8(bytes).map_err(|e| DecodeError::InvalidUtf8 {
            valid_up_to: e.utf8_error().valid_up_to(),
        })
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum EncodeError {
    #[error("encode is not yet implemented — pre-tokenizer and BPE merges land in A2b/A2c")]
    Unimplemented,

    /// A byte-level BPE encoder may see a pre-token whose
    /// fully-merged form isn't in the vocabulary. Shouldn't
    /// happen for well-formed vocabs (byte fallback guarantees
    /// coverage) but surfaced explicitly for debuggability.
    #[error("token sequence {merged:?} not in vocabulary")]
    UnknownMerged { merged: String },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DecodeError {
    #[error("token id {0} is past the end of the vocabulary")]
    OutOfVocabulary(u32),

    #[error(
        "token {id} contains char {char:?} that isn't part of the GPT-2 byte↔unicode alphabet"
    )]
    InvalidTokenChar { id: u32, char: char },

    #[error("decoded byte sequence is not valid UTF-8 (stopped at byte {valid_up_to})")]
    InvalidUtf8 { valid_up_to: usize },
}

fn parse_merge(raw: &str) -> Result<(String, String), TokenizerError> {
    let (left, right) = raw
        .split_once(' ')
        .ok_or_else(|| TokenizerError::MalformedMerge(raw.to_owned()))?;
    if left.is_empty() || right.is_empty() {
        return Err(TokenizerError::MalformedMerge(raw.to_owned()));
    }
    Ok((left.to_owned(), right.to_owned()))
}

// ─── GGUF metadata helpers ────────────────────────────────────────────────

fn read_string<'a>(gguf: &'a GgufFile, key: &str) -> Option<&'a str> {
    match gguf.find_metadata_value(key)? {
        GgufMetadataValue::String(s) => Some(s.as_str()),
        _ => None,
    }
}

fn read_bool(gguf: &GgufFile, key: &str) -> Option<bool> {
    match gguf.find_metadata_value(key)? {
        GgufMetadataValue::Bool(b) => Some(*b),
        _ => None,
    }
}

fn read_u32(gguf: &GgufFile, key: &str) -> Option<u32> {
    match gguf.find_metadata_value(key)? {
        GgufMetadataValue::Uint32(v) => Some(*v),
        GgufMetadataValue::Uint64(v) => u32::try_from(*v).ok(),
        GgufMetadataValue::Int32(v) if *v >= 0 => Some(*v as u32),
        GgufMetadataValue::Int64(v) if *v >= 0 => u32::try_from(*v).ok(),
        _ => None,
    }
}

fn read_string_array(gguf: &GgufFile, key: &str) -> Option<Vec<String>> {
    let GgufMetadataValue::Array { values, .. } = gguf.find_metadata_value(key)? else {
        return None;
    };
    let mut out = Vec::with_capacity(values.len());
    for v in values {
        if let GgufMetadataValue::String(s) = v {
            out.push(s.clone());
        } else {
            return None;
        }
    }
    Some(out)
}

fn read_f32_array(gguf: &GgufFile, key: &str) -> Option<Vec<f32>> {
    let GgufMetadataValue::Array { values, .. } = gguf.find_metadata_value(key)? else {
        return None;
    };
    let mut out = Vec::with_capacity(values.len());
    for v in values {
        match v {
            GgufMetadataValue::Float32(f) => out.push(*f),
            GgufMetadataValue::Float64(f) => out.push(*f as f32),
            _ => return None,
        }
    }
    Some(out)
}

fn read_u8_array(gguf: &GgufFile, key: &str) -> Option<Vec<u8>> {
    let GgufMetadataValue::Array { values, .. } = gguf.find_metadata_value(key)? else {
        return None;
    };
    let mut out = Vec::with_capacity(values.len());
    for v in values {
        match v {
            GgufMetadataValue::Uint8(x) => out.push(*x),
            GgufMetadataValue::Int8(x) if *x >= 0 => out.push(*x as u8),
            GgufMetadataValue::Uint32(x) if *x <= u8::MAX as u32 => out.push(*x as u8),
            GgufMetadataValue::Int32(x) if (0..=u8::MAX as i32).contains(x) => out.push(*x as u8),
            _ => return None,
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_merge_splits_on_first_space() {
        assert_eq!(
            parse_merge("Ġ t").unwrap(),
            ("Ġ".to_owned(), "t".to_owned())
        );
        assert_eq!(
            parse_merge("a bc").unwrap(),
            ("a".to_owned(), "bc".to_owned())
        );
    }

    #[test]
    fn parse_merge_rejects_no_space() {
        assert!(matches!(
            parse_merge("Ġt"),
            Err(TokenizerError::MalformedMerge(_))
        ));
    }

    #[test]
    fn parse_merge_rejects_empty_side() {
        assert!(matches!(
            parse_merge(" t"),
            Err(TokenizerError::MalformedMerge(_))
        ));
        assert!(matches!(
            parse_merge("a "),
            Err(TokenizerError::MalformedMerge(_))
        ));
    }

    #[test]
    fn tokenizer_model_from_str_known_variants() {
        assert_eq!(TokenizerModel::from_str("llama"), TokenizerModel::Llama);
        assert_eq!(TokenizerModel::from_str("gpt2"), TokenizerModel::Gpt2);
        assert_eq!(TokenizerModel::from_str("bert"), TokenizerModel::Bert);
        assert_eq!(
            TokenizerModel::from_str("rwkv"),
            TokenizerModel::Other("rwkv".to_owned())
        );
    }

    // ─── A2a byte↔unicode table ────────────────────────────────────────

    #[test]
    fn remapped_bytes_has_68_entries_strictly_ascending() {
        assert_eq!(REMAPPED_BYTES.len(), 68);
        for w in REMAPPED_BYTES.windows(2) {
            assert!(w[0] < w[1], "REMAPPED_BYTES must be strictly ascending");
        }
    }

    #[test]
    fn byte_to_char_self_mapped_ranges_are_identity() {
        for b in 33u8..=126 {
            assert_eq!(byte_to_char(b) as u32, b as u32);
        }
        for b in 161u8..=172 {
            assert_eq!(byte_to_char(b) as u32, b as u32);
        }
        for b in 174u8..=255 {
            assert_eq!(byte_to_char(b) as u32, b as u32);
        }
    }

    #[test]
    fn byte_to_char_remapped_bytes_land_in_256_range() {
        // The 68 remapped bytes get codepoints 256..=323 in
        // ascending order.
        for (i, &b) in REMAPPED_BYTES.iter().enumerate() {
            assert_eq!(byte_to_char(b) as u32, 256 + i as u32);
        }
    }

    #[test]
    fn byte_to_char_round_trip_covers_every_byte() {
        for b in 0u8..=255 {
            let c = byte_to_char(b);
            assert_eq!(char_to_byte(c), Some(b), "byte {b} failed round-trip");
        }
    }

    #[test]
    fn byte_to_char_is_injective() {
        // 256 distinct bytes → 256 distinct codepoints.
        let mut seen = std::collections::HashSet::new();
        for b in 0u8..=255 {
            assert!(seen.insert(byte_to_char(b) as u32), "duplicate byte→char at {b}");
        }
    }

    #[test]
    fn char_to_byte_rejects_codepoints_outside_the_table() {
        // Codepoints not in the table should return None.
        assert_eq!(char_to_byte('\u{1F600}'), None); // emoji, not in table
        assert_eq!(char_to_byte('\u{0020}'), None); // space itself (it's remapped to 256..)
        assert_eq!(char_to_byte('\u{0080}'), None); // first high-control char (remapped)
    }

    // ─── A2a Tokenizer build + decode + encode stub ────────────────────

    fn synth_gpt2_config() -> TokenizerConfig {
        // Mini vocab: two single-byte tokens ("a", "!") and one
        // merged pair ("a!"). Enough to exercise build + decode.
        let tokens = vec!["a".to_owned(), "!".to_owned(), "a!".to_owned()];
        TokenizerConfig {
            model: TokenizerModel::Gpt2,
            pre_tokenizer: Some("gpt2".to_owned()),
            tokens,
            scores: None,
            token_types: vec![1, 1, 1],
            merges: vec![("a".to_owned(), "!".to_owned())],
            special: SpecialTokens::default(),
        }
    }

    #[test]
    fn tokenizer_build_rejects_non_gpt2() {
        let mut cfg = synth_gpt2_config();
        cfg.model = TokenizerModel::Llama;
        let err = Tokenizer::from_config(cfg).unwrap_err();
        assert!(matches!(err, TokenizerError::UnsupportedModel(_)));
    }

    #[test]
    fn tokenizer_build_populates_vocab_map() {
        let t = Tokenizer::from_config(synth_gpt2_config()).unwrap();
        assert_eq!(t.token_id("a"), Some(0));
        assert_eq!(t.token_id("!"), Some(1));
        assert_eq!(t.token_id("a!"), Some(2));
        assert_eq!(t.token_id("missing"), None);
    }

    #[test]
    fn decode_single_tokens_round_trips_bytes() {
        let t = Tokenizer::from_config(synth_gpt2_config()).unwrap();
        assert_eq!(t.decode(&[0]).unwrap(), "a");
        assert_eq!(t.decode(&[1]).unwrap(), "!");
        assert_eq!(t.decode(&[2]).unwrap(), "a!");
        assert_eq!(t.decode(&[0, 1]).unwrap(), "a!");
    }

    #[test]
    fn decode_rejects_out_of_vocab_id() {
        let t = Tokenizer::from_config(synth_gpt2_config()).unwrap();
        assert_eq!(t.decode(&[999]), Err(DecodeError::OutOfVocabulary(999)));
    }

    #[test]
    fn decode_rejects_token_with_char_outside_byte_alphabet() {
        // Inject an emoji into a vocab — not a valid byte-level
        // BPE token. Decoder must report which token + char.
        let mut cfg = synth_gpt2_config();
        cfg.tokens.push("🤖".to_owned());
        cfg.token_types.push(1);
        let t = Tokenizer::from_config(cfg).unwrap();
        let err = t.decode(&[3]).unwrap_err();
        assert!(matches!(err, DecodeError::InvalidTokenChar { id: 3, .. }));
    }

    #[test]
    fn encode_is_stubbed_in_a2a() {
        let t = Tokenizer::from_config(synth_gpt2_config()).unwrap();
        assert_eq!(t.encode("anything"), Err(EncodeError::Unimplemented));
    }
}
