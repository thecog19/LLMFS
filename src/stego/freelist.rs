#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreeListBootstrap {
    pub block_size: usize,
}

impl Default for FreeListBootstrap {
    fn default() -> Self {
        Self {
            block_size: crate::BLOCK_SIZE,
        }
    }
}
