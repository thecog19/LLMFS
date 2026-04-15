#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticsBootstrap {
    pub tracks_tiers: bool,
}

impl Default for DiagnosticsBootstrap {
    fn default() -> Self {
        Self { tracks_tiers: true }
    }
}
