#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileTableBootstrap {
    pub entries_per_block: usize,
}

impl Default for FileTableBootstrap {
    fn default() -> Self {
        Self {
            entries_per_block: 16,
        }
    }
}
