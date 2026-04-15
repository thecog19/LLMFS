#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressionBootstrap {
    pub enabled_by_default: bool,
}

impl Default for CompressionBootstrap {
    fn default() -> Self {
        Self {
            enabled_by_default: true,
        }
    }
}
