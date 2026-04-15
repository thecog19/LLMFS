#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NbdProtocolBootstrap {
    pub block_size: usize,
}

impl Default for NbdProtocolBootstrap {
    fn default() -> Self {
        Self {
            block_size: crate::BLOCK_SIZE,
        }
    }
}
