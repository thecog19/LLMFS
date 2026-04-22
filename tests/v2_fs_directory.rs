//! V2 Filesystem — hierarchical directory semantics.
//!
//! Covers:
//! 1. Fresh init has an empty root directory.
//! 2. `mkdir` + `readdir` at root and at nested paths.
//! 3. `create_file` / `read_file` at root and nested paths;
//!    overwriting reuses the same path; file bytes round-trip.
//! 4. `unlink` removes a file; `rmdir` removes an empty directory;
//!    non-empty `rmdir` errors; `unlink` on a directory errors and
//!    vice versa.
//! 5. `exists` correctly distinguishes present / absent.
//! 6. Path-validation rejects relative paths, "..", ".".
//! 7. Full directory tree survives unmount → remount.
//! 8. Files with the same name in different directories don't
//!    collide.

use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};
use llmdb::v2::cdc::FastCdcParams;
use llmdb::v2::cover::CoverStorage;
use llmdb::v2::directory::EntryKind;
use llmdb::v2::fs::{Filesystem, FsError};

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
        name: "dir.f16".to_owned(),
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

fn init(cover: Vec<u8>, map: TensorMap) -> Filesystem {
    Filesystem::init_with_cdc_params(cover, map, small_cdc()).expect("init")
}

// ------------------------------------------------------------------
// (1) Empty root after init
// ------------------------------------------------------------------

#[test]
fn fresh_init_has_empty_root_directory() {
    let (cover, map) = make_cover();
    let fs = init(cover, map);
    let entries = fs.readdir("/").expect("readdir root");
    assert!(entries.is_empty(), "fresh init should have no entries");
    assert!(fs.exists("/"));
    assert!(!fs.exists("/nothing"));
}

// ------------------------------------------------------------------
// (2) mkdir + readdir
// ------------------------------------------------------------------

#[test]
fn mkdir_adds_entry_at_root() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map);
    fs.mkdir("/subdir").expect("mkdir");
    let entries = fs.readdir("/").unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].name, "subdir");
    assert_eq!(entries[0].kind, EntryKind::Directory);
    assert!(fs.exists("/subdir"));
}

#[test]
fn mkdir_nested_requires_parent() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map);
    // Parent doesn't exist yet.
    let err = fs.mkdir("/a/b").unwrap_err();
    assert!(matches!(err, FsError::PathNotFound(_)), "got {err:?}");

    fs.mkdir("/a").unwrap();
    fs.mkdir("/a/b").unwrap();
    fs.mkdir("/a/b/c").unwrap();

    let root = fs.readdir("/").unwrap();
    let a = fs.readdir("/a").unwrap();
    let b = fs.readdir("/a/b").unwrap();
    let c = fs.readdir("/a/b/c").unwrap();

    assert_eq!(root.len(), 1);
    assert_eq!(root[0].name, "a");
    assert_eq!(a.len(), 1);
    assert_eq!(a[0].name, "b");
    assert_eq!(b.len(), 1);
    assert_eq!(b[0].name, "c");
    assert!(c.is_empty());
}

#[test]
fn failed_mkdir_is_a_noop() {
    let (cover, map) = make_cover();
    let mut fs = init(cover.clone(), map.clone());
    let baseline = init(cover, map);
    let free_before = fs.allocator_free_weights();
    let generation_before = fs.generation();

    let err = fs.mkdir("/missing/child").unwrap_err();
    assert!(matches!(err, FsError::PathNotFound(_)), "got {err:?}");
    assert_eq!(fs.allocator_free_weights(), free_before);
    assert_eq!(fs.generation(), generation_before);
    assert!(fs.readdir("/").unwrap().is_empty());
    assert_eq!(
        fs.unmount().expect("unmount").bytes(),
        baseline.unmount().expect("unmount baseline").bytes()
    );
}

#[test]
fn mkdir_collision_errors() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map);
    fs.mkdir("/dup").unwrap();
    let err = fs.mkdir("/dup").unwrap_err();
    assert!(matches!(err, FsError::AlreadyExists(_)), "got {err:?}");
}

#[test]
fn readdir_reports_entries_sorted() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map);
    fs.mkdir("/zeta").unwrap();
    fs.mkdir("/alpha").unwrap();
    fs.mkdir("/mu").unwrap();
    let names: Vec<_> = fs.readdir("/").unwrap().into_iter().map(|e| e.name).collect();
    assert_eq!(names, vec!["alpha", "mu", "zeta"]);
}

// ------------------------------------------------------------------
// (3) create_file + read_file
// ------------------------------------------------------------------

#[test]
fn create_and_read_file_at_root() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map);
    let data = b"hello, world".to_vec();
    fs.create_file("/greeting.txt", &data).unwrap();
    assert_eq!(fs.read_file("/greeting.txt").unwrap(), data);
    assert!(fs.exists("/greeting.txt"));
}

#[test]
fn create_and_read_file_in_subdirectory() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map);
    fs.mkdir("/docs").unwrap();
    let data = b"a nested file".to_vec();
    fs.create_file("/docs/readme.md", &data).unwrap();
    assert_eq!(fs.read_file("/docs/readme.md").unwrap(), data);
    let entries = fs.readdir("/docs").unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].name, "readme.md");
    assert_eq!(entries[0].kind, EntryKind::File);
}

#[test]
fn create_file_overwrites_existing_file() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map);
    fs.create_file("/f", b"original").unwrap();
    fs.create_file("/f", b"replaced").unwrap();
    assert_eq!(fs.read_file("/f").unwrap(), b"replaced".to_vec());
}

#[test]
fn create_file_where_directory_exists_errors() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map);
    fs.mkdir("/foo").unwrap();
    let err = fs.create_file("/foo", b"data").unwrap_err();
    assert!(matches!(err, FsError::IsADirectory(_)), "got {err:?}");
}

#[test]
fn failed_create_file_is_a_noop() {
    let (cover, map) = make_cover();
    let mut fs = init(cover.clone(), map.clone());
    let mut baseline = init(cover, map);
    fs.mkdir("/foo").unwrap();
    baseline.mkdir("/foo").unwrap();
    let free_before = fs.allocator_free_weights();
    let generation_before = fs.generation();

    let err = fs.create_file("/foo", b"data").unwrap_err();
    assert!(matches!(err, FsError::IsADirectory(_)), "got {err:?}");
    assert_eq!(fs.allocator_free_weights(), free_before);
    assert_eq!(fs.generation(), generation_before);
    assert!(fs.readdir("/foo").unwrap().is_empty());
    assert_eq!(
        fs.unmount().expect("unmount").bytes(),
        baseline.unmount().expect("unmount baseline").bytes()
    );
}

#[test]
fn read_file_missing_errors() {
    let (cover, map) = make_cover();
    let fs = init(cover, map);
    let err = fs.read_file("/missing").unwrap_err();
    assert!(matches!(err, FsError::PathNotFound(_)), "got {err:?}");
}

#[test]
fn read_file_on_directory_errors() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map);
    fs.mkdir("/dir").unwrap();
    let err = fs.read_file("/dir").unwrap_err();
    assert!(matches!(err, FsError::IsADirectory(_)), "got {err:?}");
}

// ------------------------------------------------------------------
// (4) unlink + rmdir
// ------------------------------------------------------------------

#[test]
fn unlink_removes_file() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map);
    fs.create_file("/doomed", b"goodbye").unwrap();
    assert!(fs.exists("/doomed"));
    fs.unlink("/doomed").unwrap();
    assert!(!fs.exists("/doomed"));
    let err = fs.read_file("/doomed").unwrap_err();
    assert!(matches!(err, FsError::PathNotFound(_)));
}

#[test]
fn unlink_missing_errors() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map);
    let err = fs.unlink("/absent").unwrap_err();
    assert!(matches!(err, FsError::PathNotFound(_)), "got {err:?}");
}

#[test]
fn unlink_on_directory_errors() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map);
    fs.mkdir("/d").unwrap();
    let err = fs.unlink("/d").unwrap_err();
    assert!(matches!(err, FsError::IsADirectory(_)), "got {err:?}");
    // The directory should still be there.
    assert!(fs.exists("/d"));
}

#[test]
fn rmdir_removes_empty_directory() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map);
    fs.mkdir("/empty").unwrap();
    fs.rmdir("/empty").unwrap();
    assert!(!fs.exists("/empty"));
}

#[test]
fn rmdir_non_empty_errors() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map);
    fs.mkdir("/d").unwrap();
    fs.create_file("/d/child", b"stuff").unwrap();
    let err = fs.rmdir("/d").unwrap_err();
    assert!(matches!(err, FsError::DirectoryNotEmpty(_)), "got {err:?}");
    assert!(fs.exists("/d"));
    assert!(fs.exists("/d/child"));
}

#[test]
fn rmdir_on_file_errors() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map);
    fs.create_file("/f", b"x").unwrap();
    let err = fs.rmdir("/f").unwrap_err();
    assert!(matches!(err, FsError::NotADirectory(_)), "got {err:?}");
    assert!(fs.exists("/f"));
}

#[test]
fn rmdir_missing_errors() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map);
    let err = fs.rmdir("/absent").unwrap_err();
    assert!(matches!(err, FsError::PathNotFound(_)), "got {err:?}");
}

// ------------------------------------------------------------------
// (5) readdir errors
// ------------------------------------------------------------------

#[test]
fn readdir_on_file_errors() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map);
    fs.create_file("/file", b"content").unwrap();
    let err = fs.readdir("/file").unwrap_err();
    assert!(matches!(err, FsError::NotADirectory(_)), "got {err:?}");
}

#[test]
fn readdir_missing_errors() {
    let (cover, map) = make_cover();
    let fs = init(cover, map);
    let err = fs.readdir("/absent").unwrap_err();
    assert!(matches!(err, FsError::PathNotFound(_)), "got {err:?}");
}

// ------------------------------------------------------------------
// (6) Path validation
// ------------------------------------------------------------------

#[test]
fn relative_paths_rejected() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map);
    let err = fs.mkdir("nope").unwrap_err();
    assert!(matches!(err, FsError::InvalidPath(_)), "got {err:?}");
    let err = fs.create_file("foo", b"").unwrap_err();
    assert!(matches!(err, FsError::InvalidPath(_)), "got {err:?}");
}

#[test]
fn dot_and_dotdot_rejected() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map);
    let err = fs.mkdir("/./x").unwrap_err();
    assert!(matches!(err, FsError::InvalidPath(_)), "got {err:?}");
    let err = fs.mkdir("/a/../b").unwrap_err();
    assert!(matches!(err, FsError::InvalidPath(_)), "got {err:?}");
}

#[test]
fn double_slashes_treated_as_single() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map);
    fs.mkdir("//dir").unwrap();
    assert!(fs.exists("/dir"));
    fs.create_file("//dir//f", b"content").unwrap();
    assert_eq!(fs.read_file("/dir/f").unwrap(), b"content".to_vec());
}

// ------------------------------------------------------------------
// (7) Round-trip across unmount/remount
// ------------------------------------------------------------------

#[test]
fn directory_tree_survives_remount() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map.clone());
    fs.mkdir("/a").unwrap();
    fs.mkdir("/a/b").unwrap();
    fs.create_file("/a/b/file.txt", b"persistent").unwrap();
    fs.create_file("/top.txt", b"top-level").unwrap();

    let cover_after = fs.unmount().expect("unmount");
    let fs2 = Filesystem::mount_with_cdc_params(cover_after, map, small_cdc())
        .expect("mount");

    assert!(fs2.exists("/a"));
    assert!(fs2.exists("/a/b"));
    assert!(fs2.exists("/a/b/file.txt"));
    assert!(fs2.exists("/top.txt"));
    assert_eq!(fs2.read_file("/a/b/file.txt").unwrap(), b"persistent".to_vec());
    assert_eq!(fs2.read_file("/top.txt").unwrap(), b"top-level".to_vec());

    let a_entries = fs2.readdir("/a").unwrap();
    assert_eq!(a_entries.len(), 1);
    assert_eq!(a_entries[0].name, "b");
    assert_eq!(a_entries[0].kind, EntryKind::Directory);
}

// ------------------------------------------------------------------
// (8) Same name in different directories
// ------------------------------------------------------------------

#[test]
fn same_filename_in_different_directories_does_not_collide() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map);
    fs.mkdir("/d1").unwrap();
    fs.mkdir("/d2").unwrap();
    fs.create_file("/d1/file", b"one").unwrap();
    fs.create_file("/d2/file", b"two").unwrap();
    assert_eq!(fs.read_file("/d1/file").unwrap(), b"one".to_vec());
    assert_eq!(fs.read_file("/d2/file").unwrap(), b"two".to_vec());
}

// ------------------------------------------------------------------
// (9) mkdir on root errors — there's nothing to create
// ------------------------------------------------------------------

#[test]
fn operations_on_root_path_error() {
    let (cover, map) = make_cover();
    let mut fs = init(cover, map);
    let err = fs.mkdir("/").unwrap_err();
    assert!(matches!(err, FsError::PathCannotBeRoot), "got {err:?}");
    let err = fs.create_file("/", b"").unwrap_err();
    assert!(matches!(err, FsError::PathCannotBeRoot), "got {err:?}");
    let err = fs.unlink("/").unwrap_err();
    assert!(matches!(err, FsError::PathCannotBeRoot), "got {err:?}");
    let err = fs.rmdir("/").unwrap_err();
    assert!(matches!(err, FsError::PathCannotBeRoot), "got {err:?}");
}
