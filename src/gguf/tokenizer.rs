#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerMetadata {
    pub source_key: &'static str,
}

impl Default for TokenizerMetadata {
    fn default() -> Self {
        Self {
            source_key: "tokenizer.ggml",
        }
    }
}
