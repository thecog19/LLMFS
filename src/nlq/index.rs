#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticIndexBootstrap {
    pub supports_paging: bool,
}

impl Default for SemanticIndexBootstrap {
    fn default() -> Self {
        Self {
            supports_paging: true,
        }
    }
}
