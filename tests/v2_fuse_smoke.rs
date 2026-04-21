//! V2 FUSE smoke tests — mount the driver against the real kernel
//! and exercise POSIX ops.
//!
//! Skipped gracefully if `/dev/fuse` is missing (container / sandbox
//! without FUSE) or `fusermount{3,}` isn't on PATH. On machines where
//! FUSE is available, each test spawns a background session,
//! performs its ops through `std::fs`, and drops the session to
//! unmount.
//!
//! Scope: prove the adapter routes kernel ops to `v2::Filesystem`
//! correctly. Correctness of the underlying V2 ops is tested in
//! `tests/v2_fs_directory.rs`; these tests only cover the FUSE edge.

use std::path::Path;
use std::thread;
use std::time::Duration;

use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};
use llmdb::v2::cdc::FastCdcParams;
use llmdb::v2::fs::Filesystem;
use llmdb::v2::fuse::{LlmdbV2Fs, MountConfig, spawn_background};

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

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp32 = ((bits >> 23) & 0xFF) as i32;
    let mantissa32 = bits & 0x7FFFFF;
    if exp32 == 0 {
        return sign << 15;
    }
    let exp16 = exp32 - 127 + 15;
    if exp16 <= 0 {
        return sign << 15;
    }
    if exp16 >= 31 {
        return (sign << 15) | (0x1F << 10);
    }
    let mantissa16 = (mantissa32 >> 13) as u16;
    (sign << 15) | ((exp16 as u16) << 10) | mantissa16
}

fn make_cover() -> (Vec<u8>, TensorMap) {
    let weight_count = 20_000_u64;
    let values: Vec<f32> = (0..weight_count)
        .map(|i| {
            let sign = if i % 3 == 0 { -1.0 } else { 1.0 };
            sign * ((i + 1) as f32) * 0.00002
        })
        .collect();
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for v in &values {
        bytes.extend_from_slice(&f32_to_f16_bits(*v).to_le_bytes());
    }
    let slot = TensorSlot {
        name: "fuse.f16".to_owned(),
        quant_type: GgufQuantType::F16,
        tier: TensorTier::Tier1,
        data_offset: 0,
        weight_count,
        stealable_bits_per_weight: GgufQuantType::F16.stealable_bits_hint(),
        capacity_bits: weight_count * GgufQuantType::F16.stealable_bits_hint() as u64,
        bit_start: 0,
        bit_end: weight_count * GgufQuantType::F16.stealable_bits_hint() as u64,
    };
    let map = TensorMap {
        slots: vec![slot.clone()],
        total_capacity_bits: slot.capacity_bits,
        total_capacity_bytes: slot.capacity_bits / 8,
    };
    (bytes, map)
}

fn small_cdc() -> FastCdcParams {
    FastCdcParams {
        min_size: 32,
        avg_size: 64,
        max_size: 128,
    }
}

/// Boot a V2 filesystem on a synthetic cover and mount it
/// in the background. Returns `None` when FUSE isn't available in
/// the sandbox — callers should early-return in that case.
fn spawn_mount() -> Option<(tempfile::TempDir, fuser::BackgroundSession)> {
    if !fuse_available() {
        eprintln!("skipping: /dev/fuse or fusermount missing");
        return None;
    }
    let (cover, map) = make_cover();
    let fs = Filesystem::init_with_cdc_params(cover, map, small_cdc()).expect("init v2");
    let driver = LlmdbV2Fs::new(fs);
    let mount = tempfile::tempdir().expect("tempdir for mount point");
    let config = MountConfig { allow_other: false };
    let session = spawn_background(driver, mount.path(), &config).expect("spawn mount");
    // Give the kernel a tick to wire the mount up before ops run.
    thread::sleep(Duration::from_millis(100));
    Some((mount, session))
}

// ------------------------------------------------------------------
// Tests — each skips gracefully when FUSE isn't available
// ------------------------------------------------------------------

#[test]
fn write_then_read_over_fuse() {
    let Some((mount, session)) = spawn_mount() else {
        return;
    };
    let path = mount.path().join("hello.txt");
    std::fs::write(&path, b"hello via v2 fuse").expect("write");
    let got = std::fs::read(&path).expect("read");
    assert_eq!(got, b"hello via v2 fuse");
    drop(session);
}

#[test]
fn mkdir_and_nested_file() {
    let Some((mount, session)) = spawn_mount() else {
        return;
    };
    let dir = mount.path().join("docs");
    std::fs::create_dir(&dir).expect("mkdir");
    let file = dir.join("note.md");
    std::fs::write(&file, b"nested bytes").expect("write nested");

    // Root should list exactly `docs`.
    let mut root_names: Vec<_> = std::fs::read_dir(mount.path())
        .expect("readdir root")
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    root_names.sort();
    assert_eq!(root_names, vec!["docs".to_string()]);

    // `docs` should list exactly `note.md`.
    let mut docs_names: Vec<_> = std::fs::read_dir(&dir)
        .expect("readdir docs")
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    docs_names.sort();
    assert_eq!(docs_names, vec!["note.md".to_string()]);

    assert_eq!(std::fs::read(&file).expect("read nested"), b"nested bytes");
    drop(session);
}

#[test]
fn unlink_removes_file_from_listing() {
    let Some((mount, session)) = spawn_mount() else {
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
fn rmdir_removes_empty_directory() {
    let Some((mount, session)) = spawn_mount() else {
        return;
    };
    let dir = mount.path().join("empty");
    std::fs::create_dir(&dir).expect("mkdir");
    assert!(dir.is_dir());
    std::fs::remove_dir(&dir).expect("rmdir");
    assert!(!dir.exists());
    drop(session);
}

#[test]
fn rmdir_on_nonempty_errors() {
    let Some((mount, session)) = spawn_mount() else {
        return;
    };
    let dir = mount.path().join("full");
    std::fs::create_dir(&dir).expect("mkdir");
    std::fs::write(dir.join("child"), b"x").expect("write child");
    let err = std::fs::remove_dir(&dir).expect_err("rmdir should fail");
    assert_eq!(err.raw_os_error(), Some(libc::ENOTEMPTY));
    drop(session);
}

#[test]
fn statfs_reports_block_size() {
    let Some((mount, session)) = spawn_mount() else {
        return;
    };
    let out = std::process::Command::new("stat")
        .args(["-f", "--format=%S %l"])
        .arg(mount.path())
        .output()
        .expect("run stat -f");
    assert!(out.status.success(), "stat -f failed: {out:?}");
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let parts: Vec<_> = text.split_whitespace().collect();
    let bsize: u64 = parts[0].parse().expect("parse bsize");
    let namelen: u64 = parts[1].parse().expect("parse namelen");
    assert_eq!(bsize, 4096);
    assert_eq!(namelen, 255);
    drop(session);
}

#[test]
fn overwrite_existing_file() {
    let Some((mount, session)) = spawn_mount() else {
        return;
    };
    let path = mount.path().join("mutable.txt");
    std::fs::write(&path, b"original").expect("write 1");
    std::fs::write(&path, b"replaced").expect("write 2");
    assert_eq!(std::fs::read(&path).expect("read"), b"replaced");
    drop(session);
}

#[test]
fn lookup_missing_returns_enoent() {
    let Some((mount, session)) = spawn_mount() else {
        return;
    };
    let err = std::fs::metadata(mount.path().join("nope")).expect_err("stat should fail");
    assert_eq!(err.raw_os_error(), Some(libc::ENOENT));
    drop(session);
}
