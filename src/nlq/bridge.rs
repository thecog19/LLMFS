#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryBridgeBootstrap {
    pub tool_name: &'static str,
}

impl Default for QueryBridgeBootstrap {
    fn default() -> Self {
        Self { tool_name: "query" }
    }
}
