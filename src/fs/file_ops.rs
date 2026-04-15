#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileOpsBootstrap {
    pub supports_flat_files: bool,
}

impl Default for FileOpsBootstrap {
    fn default() -> Self {
        Self {
            supports_flat_files: true,
        }
    }
}
