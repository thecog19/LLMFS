#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AskBridgeBootstrap {
    pub tool_count: usize,
}

impl Default for AskBridgeBootstrap {
    fn default() -> Self {
        Self { tool_count: 3 }
    }
}
