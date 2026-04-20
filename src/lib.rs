pub mod ask;
pub mod diagnostics;
pub mod fs;
pub mod fuse;
pub mod gguf;
pub mod nbd;
pub mod stego;

pub const APP_NAME: &str = "llmdb";
pub const BLOCK_SIZE: usize = 4096;

pub fn bootstrap_status() -> &'static str {
    "bootstrap"
}
