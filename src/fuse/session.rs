//! Mount-session wrappers. Foreground mounts block until the kernel
//! unmounts us (Ctrl-C then `fusermount3 -u <mount_point>`). Background
//! mounts return a guard that drops the mount on `drop`, so callers can
//! run operations against the mounted tree and unmount by dropping.

use std::path::Path;

use fuser::{BackgroundSession, MountOption, Session};

use crate::fuse::fs::LlmdbFs;

/// Mount options we always set. `FSName` tags the device for
/// `findmnt`/`mount` output; `DefaultPermissions` hands permission
/// checks to the kernel so we don't have to reimplement them.
fn base_options() -> Vec<MountOption> {
    vec![
        MountOption::FSName("llmdb".to_owned()),
        MountOption::DefaultPermissions,
        MountOption::NoAtime,
    ]
}

pub struct MountConfig {
    pub allow_other: bool,
}

impl MountConfig {
    pub fn options(&self) -> Vec<MountOption> {
        let mut opts = base_options();
        if self.allow_other {
            opts.push(MountOption::AllowOther);
        }
        opts
    }
}

/// Mount the driver at `mount_point` and block the calling thread until
/// the mount is released (kernel unmount). Returns when the session loop
/// exits — normally triggered by `fusermount3 -u`.
pub fn mount_foreground(
    fs: LlmdbFs,
    mount_point: impl AsRef<Path>,
    config: &MountConfig,
) -> std::io::Result<()> {
    fuser::mount2(fs, mount_point, &config.options())
}

/// Mount the driver in a background thread. The returned session
/// unmounts on drop; callers can stash it and use the mount point
/// normally while it's alive.
pub fn spawn_background(
    fs: LlmdbFs,
    mount_point: impl AsRef<Path>,
    config: &MountConfig,
) -> std::io::Result<BackgroundSession> {
    let session = Session::new(fs, mount_point.as_ref(), &config.options())?;
    session.spawn()
}
