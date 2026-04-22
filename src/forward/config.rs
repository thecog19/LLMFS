//! Typed GGUF metadata → `LlamaConfig` for the forward pass.
//!
//! Parses the `llama.*` namespace (and `general.architecture`) into
//! a plain struct. Milestone A is llama-only; other architectures
//! return [`ConfigError::UnsupportedArch`]. Adding Qwen-2 or similar
//! is mechanical once the forward pass shape is validated — the
//! dispatch in [`LlamaConfig::from_gguf`] would grow one more arm
//! that remaps the same fields under a different key prefix.

use thiserror::Error;

use crate::gguf::parser::{GgufFile, GgufMetadataValue};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LlamaConfig {
    /// Token vocabulary size. Taken from the tokenizer (authoritative)
    /// rather than any arch key — some GGUFs omit `llama.vocab_size`.
    pub vocab_size: usize,
    /// Residual-stream dimension. `llama.embedding_length`.
    pub hidden_dim: usize,
    /// FFN hidden dimension. `llama.feed_forward_length`.
    pub ffn_dim: usize,
    /// Number of transformer blocks. `llama.block_count`.
    pub n_layers: usize,
    /// Query-head count. `llama.attention.head_count`.
    pub n_heads: usize,
    /// Key/value-head count (GQA). Defaults to `n_heads` when absent.
    pub n_kv_heads: usize,
    /// Per-head dimension. Derived: `hidden_dim / n_heads`.
    pub head_dim: usize,
    /// Channels rotated by RoPE. `llama.rope.dimension_count`;
    /// defaults to `head_dim`.
    pub rope_dim: usize,
    /// RoPE base frequency. `llama.rope.freq_base`; defaults to 10_000.
    pub rope_freq_base: f32,
    /// RMSNorm epsilon. `llama.attention.layer_norm_rms_epsilon`;
    /// defaults to 1e-5.
    pub norm_eps: f32,
    /// Max context length the model was trained for. Informational;
    /// doesn't constrain the forward pass.
    pub context_length: usize,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("missing metadata key: {0}")]
    MissingKey(String),

    #[error("metadata key {key}: expected {expected}, found {found}")]
    WrongType {
        key: String,
        expected: &'static str,
        found: &'static str,
    },

    #[error("unsupported architecture: {arch:?} (Milestone A implements `llama` only)")]
    UnsupportedArch { arch: String },

    #[error(
        "config inconsistency: hidden_dim {hidden} not divisible by n_heads {n_heads}"
    )]
    HeadDimIndivisible { hidden: usize, n_heads: usize },

    #[error("tokenizer vocab is empty (expected `tokenizer.ggml.tokens` array)")]
    EmptyTokenizer,
}

impl LlamaConfig {
    /// Parse metadata out of a GGUF file. Returns `ConfigError` if
    /// the architecture isn't supported, a required key is missing,
    /// or the config is internally inconsistent.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, ConfigError> {
        let arch = read_string(gguf, "general.architecture")?;
        if arch != "llama" {
            return Err(ConfigError::UnsupportedArch { arch });
        }

        let hidden_dim = read_u64_as_usize(gguf, "llama.embedding_length")?;
        let ffn_dim = read_u64_as_usize(gguf, "llama.feed_forward_length")?;
        let n_layers = read_u64_as_usize(gguf, "llama.block_count")?;
        let n_heads = read_u64_as_usize(gguf, "llama.attention.head_count")?;
        let n_kv_heads = read_u64_as_usize_opt(gguf, "llama.attention.head_count_kv")
            .unwrap_or(n_heads);

        if !hidden_dim.is_multiple_of(n_heads) {
            return Err(ConfigError::HeadDimIndivisible {
                hidden: hidden_dim,
                n_heads,
            });
        }
        let head_dim = hidden_dim / n_heads;

        let rope_dim =
            read_u64_as_usize_opt(gguf, "llama.rope.dimension_count").unwrap_or(head_dim);
        let rope_freq_base =
            read_f32_opt(gguf, "llama.rope.freq_base").unwrap_or(10_000.0);
        let norm_eps =
            read_f32_opt(gguf, "llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let context_length =
            read_u64_as_usize_opt(gguf, "llama.context_length").unwrap_or(0);

        // Vocab size comes from the tokenizer, not the arch section —
        // `llama.vocab_size` is sometimes omitted.
        let vocab_size = match gguf.find_metadata_value("tokenizer.ggml.tokens") {
            Some(GgufMetadataValue::Array { values, .. }) => values.len(),
            _ => return Err(ConfigError::EmptyTokenizer),
        };
        if vocab_size == 0 {
            return Err(ConfigError::EmptyTokenizer);
        }

        Ok(Self {
            vocab_size,
            hidden_dim,
            ffn_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_dim,
            rope_freq_base,
            norm_eps,
            context_length,
        })
    }
}

// ─── Small metadata-reader helpers ────────────────────────────────

fn read_string(gguf: &GgufFile, key: &str) -> Result<String, ConfigError> {
    match gguf.find_metadata_value(key) {
        Some(GgufMetadataValue::String(s)) => Ok(s.clone()),
        Some(other) => Err(ConfigError::WrongType {
            key: key.to_owned(),
            expected: "string",
            found: value_kind(other),
        }),
        None => Err(ConfigError::MissingKey(key.to_owned())),
    }
}

fn read_u64_as_usize(gguf: &GgufFile, key: &str) -> Result<usize, ConfigError> {
    read_u64_as_usize_opt(gguf, key).ok_or_else(|| match gguf.find_metadata_value(key) {
        Some(other) => ConfigError::WrongType {
            key: key.to_owned(),
            expected: "integer",
            found: value_kind(other),
        },
        None => ConfigError::MissingKey(key.to_owned()),
    })
}

fn read_u64_as_usize_opt(gguf: &GgufFile, key: &str) -> Option<usize> {
    match gguf.find_metadata_value(key)? {
        GgufMetadataValue::Uint32(v) => Some(*v as usize),
        GgufMetadataValue::Uint64(v) => Some(*v as usize),
        GgufMetadataValue::Int32(v) if *v >= 0 => Some(*v as usize),
        GgufMetadataValue::Int64(v) if *v >= 0 => Some(*v as usize),
        _ => None,
    }
}

fn read_f32_opt(gguf: &GgufFile, key: &str) -> Option<f32> {
    match gguf.find_metadata_value(key)? {
        GgufMetadataValue::Float32(v) => Some(*v),
        GgufMetadataValue::Float64(v) => Some(*v as f32),
        _ => None,
    }
}

fn value_kind(v: &GgufMetadataValue) -> &'static str {
    match v {
        GgufMetadataValue::Uint8(_) => "u8",
        GgufMetadataValue::Int8(_) => "i8",
        GgufMetadataValue::Uint16(_) => "u16",
        GgufMetadataValue::Int16(_) => "i16",
        GgufMetadataValue::Uint32(_) => "u32",
        GgufMetadataValue::Int32(_) => "i32",
        GgufMetadataValue::Uint64(_) => "u64",
        GgufMetadataValue::Int64(_) => "i64",
        GgufMetadataValue::Float32(_) => "f32",
        GgufMetadataValue::Float64(_) => "f64",
        GgufMetadataValue::Bool(_) => "bool",
        GgufMetadataValue::String(_) => "string",
        GgufMetadataValue::Array { .. } => "array",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::parser::{GgufHeader, GgufMetadataEntry, GgufMetadataValueType};

    fn make_entry(key: &str, value: GgufMetadataValue) -> GgufMetadataEntry {
        GgufMetadataEntry {
            key: key.to_owned(),
            value,
        }
    }

    fn minimal_gguf() -> GgufFile {
        // Build a minimal in-memory GGUF with just the metadata we
        // need — no tensors, no data. The parser doesn't expose a
        // test constructor so we construct the struct directly.
        let tokens = GgufMetadataValue::Array {
            element_type: GgufMetadataValueType::String,
            values: (0..1024)
                .map(|i| GgufMetadataValue::String(format!("t{i}")))
                .collect(),
        };
        GgufFile {
            header: GgufHeader {
                version: 3,
                tensor_count: 0,
                metadata_count: 0,
            },
            metadata: vec![
                make_entry(
                    "general.architecture",
                    GgufMetadataValue::String("llama".to_owned()),
                ),
                make_entry("llama.embedding_length", GgufMetadataValue::Uint32(128)),
                make_entry("llama.feed_forward_length", GgufMetadataValue::Uint32(256)),
                make_entry("llama.block_count", GgufMetadataValue::Uint32(4)),
                make_entry("llama.attention.head_count", GgufMetadataValue::Uint32(8)),
                make_entry(
                    "llama.attention.head_count_kv",
                    GgufMetadataValue::Uint32(2),
                ),
                make_entry("llama.context_length", GgufMetadataValue::Uint32(2048)),
                make_entry(
                    "llama.rope.freq_base",
                    GgufMetadataValue::Float32(100_000.0),
                ),
                make_entry(
                    "llama.attention.layer_norm_rms_epsilon",
                    GgufMetadataValue::Float32(1e-6),
                ),
                make_entry("tokenizer.ggml.tokens", tokens),
            ],
            tensors: Vec::new(),
            alignment: 32,
            tensor_data_offset: 0,
        }
    }

    #[test]
    fn from_gguf_parses_all_fields() {
        let gguf = minimal_gguf();
        let cfg = LlamaConfig::from_gguf(&gguf).unwrap();
        assert_eq!(cfg.vocab_size, 1024);
        assert_eq!(cfg.hidden_dim, 128);
        assert_eq!(cfg.ffn_dim, 256);
        assert_eq!(cfg.n_layers, 4);
        assert_eq!(cfg.n_heads, 8);
        assert_eq!(cfg.n_kv_heads, 2);
        assert_eq!(cfg.head_dim, 16); // 128/8
        assert_eq!(cfg.rope_dim, 16); // defaults to head_dim
        assert_eq!(cfg.rope_freq_base, 100_000.0);
        assert_eq!(cfg.norm_eps, 1e-6);
        assert_eq!(cfg.context_length, 2048);
    }

    #[test]
    fn missing_n_kv_heads_defaults_to_n_heads() {
        let mut gguf = minimal_gguf();
        gguf.metadata
            .retain(|e| e.key != "llama.attention.head_count_kv");
        let cfg = LlamaConfig::from_gguf(&gguf).unwrap();
        assert_eq!(cfg.n_kv_heads, cfg.n_heads);
    }

    #[test]
    fn missing_rope_freq_base_defaults_to_10000() {
        let mut gguf = minimal_gguf();
        gguf.metadata.retain(|e| e.key != "llama.rope.freq_base");
        let cfg = LlamaConfig::from_gguf(&gguf).unwrap();
        assert_eq!(cfg.rope_freq_base, 10_000.0);
    }

    #[test]
    fn non_llama_arch_is_rejected() {
        let mut gguf = minimal_gguf();
        for e in gguf.metadata.iter_mut() {
            if e.key == "general.architecture" {
                e.value = GgufMetadataValue::String("qwen2".to_owned());
            }
        }
        let err = LlamaConfig::from_gguf(&gguf).unwrap_err();
        assert!(matches!(err, ConfigError::UnsupportedArch { arch } if arch == "qwen2"));
    }

    #[test]
    fn hidden_not_divisible_by_heads_errors() {
        let mut gguf = minimal_gguf();
        for e in gguf.metadata.iter_mut() {
            if e.key == "llama.embedding_length" {
                e.value = GgufMetadataValue::Uint32(127); // not divisible by 8
            }
        }
        let err = LlamaConfig::from_gguf(&gguf).unwrap_err();
        assert!(matches!(err, ConfigError::HeadDimIndivisible { .. }));
    }

    #[test]
    fn missing_tokenizer_is_empty_error() {
        let mut gguf = minimal_gguf();
        gguf.metadata.retain(|e| e.key != "tokenizer.ggml.tokens");
        let err = LlamaConfig::from_gguf(&gguf).unwrap_err();
        assert!(matches!(err, ConfigError::EmptyTokenizer));
    }
}
