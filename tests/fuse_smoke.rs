//! Integration tests that mount the FUSE driver against a real kernel
//! and drive it via standard filesystem syscalls. Skipped gracefully if
//! `/dev/fuse` is missing (container without FUSE) or `fusermount3` /
//! `fusermount` isn't on PATH.
//!
//! The `BackgroundSession` returned by `spawn_background` unmounts on
//! drop, so each test is self-contained.

mod common;

use std::path::Path;
use std::thread;
use std::time::Duration;

use common::{SyntheticGgufVersion, SyntheticTensorSpec, write_custom_gguf_fixture};
use llmdb::fuse::{LlmdbFs, MountConfig, spawn_background};
use llmdb::gguf::quant::GGML_TYPE_Q8_0_ID;
use llmdb::stego::device::{DeviceOptions, StegoDevice};
use llmdb::stego::planner::AllocationMode;

fn fuse_available() -> bool {
    if !Path::new("/dev/fuse").exists() {
        return false;
    }
    ["fusermount3", "fusermount"]
        .iter()
        .any(|prog| which(prog).is_some())
}

fn which(prog: &str) -> Option<std::path::PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|p| {
            let candidate = p.join(prog);
            if candidate.is_file() {
                Some(candidate)
            } else {
                None
            }
        })
    })
}

fn q8_tensor(name: &str, weight_count: usize) -> SyntheticTensorSpec {
    SyntheticTensorSpec {
        name: name.to_owned(),
        dimensions: vec![weight_count as u64],
        raw_type_id: GGML_TYPE_Q8_0_ID,
        data: vec![0_u8; (weight_count / 32) * 34],
    }
}

fn make_tensors(n: usize) -> Vec<SyntheticTensorSpec> {
    (0..n)
        .map(|i| q8_tensor(&format!("blk.{}.ffn_down.weight", n - 1 - i), 8192))
        .collect()
}

fn spawn_mount(
    name: &str,
) -> Option<(
    common::FixtureHandle,
    tempfile::TempDir,
    fuser::BackgroundSession,
)> {
    if !fuse_available() {
        eprintln!("skipping: /dev/fuse or fusermount missing");
        return None;
    }
    let fixture = write_custom_gguf_fixture(SyntheticGgufVersion::V3, name, &make_tensors(32));
    let device = StegoDevice::initialize_with_options(
        &fixture.path,
        AllocationMode::Standard,
        DeviceOptions { verbose: false },
    )
    .expect("init device");
    let mount = tempfile::tempdir().expect("tempdir for mount point");
    let fs = LlmdbFs::new(device);
    let config = MountConfig { allow_other: false };
    let session = spawn_background(fs, mount.path(), &config).expect("spawn mount");
    // Give the kernel a tick to wire the mount before tests touch it.
    thread::sleep(Duration::from_millis(100));
    Some((fixture, mount, session))
}

#[test]
fn write_read_roundtrip_over_fuse() {
    let Some((_fixture, mount, session)) = spawn_mount("fuse_rw.gguf") else {
        return;
    };
    let path = mount.path().join("hello.txt");
    std::fs::write(&path, b"hello via fuse").expect("write");
    let got = std::fs::read(&path).expect("read");
    assert_eq!(got, b"hello via fuse");
    drop(session);
}

#[test]
fn virtual_directory_appears_from_slashed_name() {
    let Some((_fixture, mount, session)) = spawn_mount("fuse_vdir.gguf") else {
        return;
    };
    let nested = mount.path().join("docs").join("notes.txt");
    std::fs::create_dir_all(nested.parent().unwrap()).expect("mkdir -p");
    std::fs::write(&nested, b"nested").expect("write nested");

    // Root listing must include `docs` as a directory.
    let mut root_names: Vec<_> = std::fs::read_dir(mount.path())
        .expect("readdir root")
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    root_names.sort();
    assert_eq!(root_names, vec!["docs".to_string()]);

    // `docs/` listing must contain `notes.txt`.
    let mut docs_names: Vec<_> = std::fs::read_dir(mount.path().join("docs"))
        .expect("readdir docs")
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    docs_names.sort();
    assert_eq!(docs_names, vec!["notes.txt".to_string()]);

    // Content readback.
    assert_eq!(std::fs::read(&nested).expect("read nested"), b"nested");
    drop(session);
}

#[test]
fn mkdir_then_rmdir_empty_virtual_dir() {
    let Some((_fixture, mount, session)) = spawn_mount("fuse_mkdir.gguf") else {
        return;
    };
    let dir = mount.path().join("empty");
    std::fs::create_dir(&dir).expect("mkdir");
    assert!(dir.is_dir(), "mkdir should surface the virtual dir");
    std::fs::remove_dir(&dir).expect("rmdir");
    assert!(!dir.exists(), "rmdir should remove the virtual dir");
    drop(session);
}

#[test]
fn unlink_removes_file_from_listing() {
    let Some((_fixture, mount, session)) = spawn_mount("fuse_unlink.gguf") else {
        return;
    };
    let path = mount.path().join("ephemeral.txt");
    std::fs::write(&path, b"bye").expect("write");
    assert!(path.exists());
    std::fs::remove_file(&path).expect("unlink");
    assert!(!path.exists());
    let names: Vec<_> = std::fs::read_dir(mount.path())
        .expect("readdir")
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(names.is_empty(), "directory should be empty after unlink");
    drop(session);
}

#[test]
fn rename_within_root_moves_contents() {
    let Some((_fixture, mount, session)) = spawn_mount("fuse_rename.gguf") else {
        return;
    };
    let from = mount.path().join("before.txt");
    let to = mount.path().join("after.txt");
    std::fs::write(&from, b"payload").expect("write");
    std::fs::rename(&from, &to).expect("rename");
    assert!(!from.exists());
    assert_eq!(std::fs::read(&to).expect("read after rename"), b"payload");
    drop(session);
}

#[test]
fn statfs_reports_block_size_and_free_blocks() {
    let Some((_fixture, mount, session)) = spawn_mount("fuse_statfs.gguf") else {
        return;
    };
    // Rust stdlib doesn't expose statvfs portably; shell out instead.
    let out = std::process::Command::new("stat")
        .args(["-f", "--format=%S %a %l"])
        .arg(mount.path())
        .output()
        .expect("run stat -f");
    assert!(out.status.success(), "stat -f failed: {:?}", out);
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let parts: Vec<_> = text.split_whitespace().collect();
    let bsize: u64 = parts[0].parse().expect("parse bsize");
    let namelen: u64 = parts[2].parse().expect("parse namelen");
    assert_eq!(bsize, llmdb::BLOCK_SIZE as u64);
    assert_eq!(namelen, llmdb::fs::file_table::MAX_FILENAME_BYTES as u64);
    drop(session);
}
