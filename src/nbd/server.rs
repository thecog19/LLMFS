#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NbdServerBootstrap {
    pub handles_partial_requests: bool,
}

impl Default for NbdServerBootstrap {
    fn default() -> Self {
        Self {
            handles_partial_requests: true,
        }
    }
}
