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

use thiserror::Error;

use crate::gguf::parser::{GgufFile, GgufMetadataValue};

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
}
