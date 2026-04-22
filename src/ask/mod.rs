pub mod bridge;
pub mod server;

use thiserror::Error;

use crate::v2::fs::FsError as V2FsError;

#[derive(Debug, Error)]
pub enum AskError {
    #[error("failed to spawn `llama-server`: {0}")]
    SpawnFailed(String),
    #[error("`llama-server` health check timed out after {seconds}s at {url}")]
    HealthTimeout { url: String, seconds: u64 },
    #[error("http error talking to `llama-server`: {0}")]
    HttpError(String),
    #[error("malformed chat response: {0}")]
    MalformedResponse(String),
    #[error("tool call loop exceeded {limit} iterations without a final reply")]
    ToolCallLimitExceeded { limit: usize },
    #[error("invalid tool call: {0}")]
    InvalidToolCall(String),
    #[error("v2 filesystem error: {0}")]
    V2Fs(#[from] V2FsError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}
