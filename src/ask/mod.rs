pub mod bridge;
pub mod server;

use thiserror::Error;

use crate::fs::file_ops::FsError;
use crate::stego::device::DeviceError;

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
    #[error("device error: {0}")]
    Device(#[from] DeviceError),
    #[error("file system error: {0}")]
    Fs(#[from] FsError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}
