//! FUSE driver exposing the LLMDB file table as a mountable filesystem.
//! Virtual directories are synthesized from `/`-separated filenames;
//! storage stays flat per DESIGN-NEW §6.

pub mod dir;
pub mod fs;
pub mod inode;
pub mod session;

pub use fs::LlmdbFs;
pub use session::{MountConfig, mount_foreground, spawn_background};
