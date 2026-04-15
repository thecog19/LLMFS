#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedirectionBootstrap {
    pub entries_per_block: usize,
}

impl Default for RedirectionBootstrap {
    fn default() -> Self {
        Self {
            entries_per_block: 1022,
        }
    }
}
