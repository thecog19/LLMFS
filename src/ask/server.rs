#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AskServerBootstrap {
    pub subprocess_backend: bool,
}

impl Default for AskServerBootstrap {
    fn default() -> Self {
        Self {
            subprocess_backend: true,
        }
    }
}
