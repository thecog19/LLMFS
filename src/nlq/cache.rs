#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheBootstrap {
    pub tracks_hits: bool,
}

impl Default for CacheBootstrap {
    fn default() -> Self {
        Self { tracks_hits: true }
    }
}
