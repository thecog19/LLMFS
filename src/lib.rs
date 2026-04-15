pub mod compress;
pub mod diagnostics;
pub mod gguf;
pub mod nlq;
pub mod stego;
pub mod vfs;

pub const APP_NAME: &str = "llmdb";
pub const BLOCK_SIZE: usize = 4096;

pub fn bootstrap_status() -> &'static str {
    "bootstrap"
}
