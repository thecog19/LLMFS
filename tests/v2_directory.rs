//! V2 Directory codec unit tests.
//!
//! A directory is just a file whose bytes are the serialized entry
//! list; these tests exercise the codec only — integration with the
//! filesystem layer (path resolution, mkdir, etc.) is covered by
//! `tests/v2_fs_directory.rs`.

use llmdb::v2::directory::{
    DIRECTORY_HEADER_BYTES, DirEntry, Directory, DirectoryError, ENTRY_HEADER_BYTES, EntryKind,
    MAX_NAME_LEN,
};
use llmdb::v2::pointer::Pointer;

fn ptr(slot: u16, w: u32) -> Pointer {
    Pointer {
        slot,
        start_weight: w,
        length_in_bits: 128,
        flags: 0,
        reserved: 0,
    }
}

// ------------------------------------------------------------------
// Empty + basic round-trips
// ------------------------------------------------------------------

#[test]
fn empty_directory_serializes_to_count_zero() {
    let dir = Directory::new();
    let bytes = dir.serialize();
    assert_eq!(bytes.len(), DIRECTORY_HEADER_BYTES);
    assert_eq!(bytes, vec![0u8; DIRECTORY_HEADER_BYTES]);
}

#[test]
fn empty_round_trip() {
    let dir = Directory::new();
    let bytes = dir.serialize();
    let restored = Directory::deserialize(&bytes).unwrap();
    assert_eq!(dir, restored);
    assert_eq!(restored.len(), 0);
    assert!(restored.is_empty());
}

#[test]
fn round_trip_preserves_every_entry() {
    let mut dir = Directory::new();
    dir.insert(DirEntry {
        kind: EntryKind::File,
        name: "a_file".to_owned(),
        inode: ptr(0, 1),
    })
    .unwrap();
    dir.insert(DirEntry {
        kind: EntryKind::Directory,
        name: "sub_dir".to_owned(),
        inode: ptr(2, 999),
    })
    .unwrap();
    dir.insert(DirEntry {
        kind: EntryKind::File,
        name: "another.txt".to_owned(),
        inode: ptr(1, 42),
    })
    .unwrap();

    let bytes = dir.serialize();
    let restored = Directory::deserialize(&bytes).unwrap();
    assert_eq!(dir, restored);
}

#[test]
fn serialize_length_matches_header_plus_entries() {
    let mut dir = Directory::new();
    dir.insert(DirEntry {
        kind: EntryKind::File,
        name: "ab".to_owned(), // 2 bytes
        inode: ptr(0, 0),
    })
    .unwrap();
    dir.insert(DirEntry {
        kind: EntryKind::Directory,
        name: "xyzzy".to_owned(), // 5 bytes
        inode: ptr(1, 0),
    })
    .unwrap();
    let expected = DIRECTORY_HEADER_BYTES + 2 * ENTRY_HEADER_BYTES + 2 + 5;
    assert_eq!(dir.serialize().len(), expected);
}

// ------------------------------------------------------------------
// Sort order
// ------------------------------------------------------------------

#[test]
fn insert_preserves_sorted_order() {
    let mut dir = Directory::new();
    dir.insert(DirEntry {
        kind: EntryKind::File,
        name: "zeta".to_owned(),
        inode: ptr(0, 0),
    })
    .unwrap();
    dir.insert(DirEntry {
        kind: EntryKind::Directory,
        name: "alpha".to_owned(),
        inode: ptr(0, 100),
    })
    .unwrap();
    dir.insert(DirEntry {
        kind: EntryKind::File,
        name: "beta".to_owned(),
        inode: ptr(0, 200),
    })
    .unwrap();

    let names: Vec<_> = dir.entries().iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, vec!["alpha", "beta", "zeta"]);
}

#[test]
fn deserialize_enforces_sorted_order() {
    // Two entries intentionally out of order on the wire: "z" then "a".
    let mut bytes = vec![2u8, 0, 0, 0]; // count = 2
    // entry 1: kind=file, name_len=1, reserved, ptr, name="z"
    bytes.push(0);
    bytes.push(1);
    bytes.extend_from_slice(&[0, 0]);
    bytes.extend_from_slice(&ptr(0, 1).encode());
    bytes.push(b'z');
    // entry 2: kind=file, name_len=1, reserved, ptr, name="a"
    bytes.push(0);
    bytes.push(1);
    bytes.extend_from_slice(&[0, 0]);
    bytes.extend_from_slice(&ptr(0, 2).encode());
    bytes.push(b'a');

    // The deserializer sorts on read, so the result reads back
    // sorted even though the bytes were out-of-order.
    let dir = Directory::deserialize(&bytes).unwrap();
    let names: Vec<_> = dir.entries().iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, vec!["a", "z"]);
}

// ------------------------------------------------------------------
// Find / remove / replace
// ------------------------------------------------------------------

#[test]
fn find_returns_matching_entry() {
    let mut dir = Directory::new();
    dir.insert(DirEntry {
        kind: EntryKind::Directory,
        name: "child".to_owned(),
        inode: ptr(1, 42),
    })
    .unwrap();

    let found = dir.find("child").unwrap();
    assert_eq!(found.name, "child");
    assert_eq!(found.kind, EntryKind::Directory);
    assert_eq!(found.inode, ptr(1, 42));

    assert!(dir.find("nonexistent").is_none());
}

#[test]
fn remove_returns_and_removes_entry() {
    let mut dir = Directory::new();
    dir.insert(DirEntry {
        kind: EntryKind::File,
        name: "victim".to_owned(),
        inode: ptr(0, 5),
    })
    .unwrap();
    dir.insert(DirEntry {
        kind: EntryKind::File,
        name: "survivor".to_owned(),
        inode: ptr(0, 10),
    })
    .unwrap();

    let removed = dir.remove("victim").unwrap();
    assert_eq!(removed.name, "victim");
    assert_eq!(dir.len(), 1);
    assert!(dir.find("victim").is_none());
    assert!(dir.find("survivor").is_some());
}

#[test]
fn remove_absent_name_returns_none() {
    let mut dir = Directory::new();
    assert!(dir.remove("absent").is_none());
}

#[test]
fn replace_updates_kind_and_pointer() {
    let mut dir = Directory::new();
    dir.insert(DirEntry {
        kind: EntryKind::File,
        name: "x".to_owned(),
        inode: ptr(0, 1),
    })
    .unwrap();
    let old = dir.replace("x", EntryKind::Directory, ptr(2, 55)).unwrap();
    assert_eq!(old.kind, EntryKind::File);
    assert_eq!(old.inode, ptr(0, 1));
    let cur = dir.find("x").unwrap();
    assert_eq!(cur.kind, EntryKind::Directory);
    assert_eq!(cur.inode, ptr(2, 55));
}

#[test]
fn replace_absent_returns_none() {
    let mut dir = Directory::new();
    assert!(dir.replace("nope", EntryKind::File, ptr(0, 0)).is_none());
}

// ------------------------------------------------------------------
// Name validation + collision rejection
// ------------------------------------------------------------------

#[test]
fn insert_rejects_name_collision() {
    let mut dir = Directory::new();
    dir.insert(DirEntry {
        kind: EntryKind::File,
        name: "dup".to_owned(),
        inode: ptr(0, 0),
    })
    .unwrap();
    let err = dir
        .insert(DirEntry {
            kind: EntryKind::File,
            name: "dup".to_owned(),
            inode: ptr(0, 1),
        })
        .unwrap_err();
    assert!(matches!(err, DirectoryError::NameCollision(_)));
}

#[test]
fn insert_rejects_empty_name() {
    let mut dir = Directory::new();
    let err = dir
        .insert(DirEntry {
            kind: EntryKind::File,
            name: String::new(),
            inode: ptr(0, 0),
        })
        .unwrap_err();
    assert!(matches!(err, DirectoryError::InvalidName));
}

#[test]
fn insert_rejects_name_too_long() {
    let mut dir = Directory::new();
    let err = dir
        .insert(DirEntry {
            kind: EntryKind::File,
            name: "a".repeat(MAX_NAME_LEN + 1),
            inode: ptr(0, 0),
        })
        .unwrap_err();
    assert!(matches!(err, DirectoryError::InvalidName));
}

#[test]
fn insert_accepts_max_length_name() {
    let mut dir = Directory::new();
    dir.insert(DirEntry {
        kind: EntryKind::File,
        name: "a".repeat(MAX_NAME_LEN),
        inode: ptr(0, 0),
    })
    .unwrap();
    assert_eq!(dir.len(), 1);
}

#[test]
fn insert_rejects_slash_in_name() {
    let mut dir = Directory::new();
    let err = dir
        .insert(DirEntry {
            kind: EntryKind::File,
            name: "a/b".to_owned(),
            inode: ptr(0, 0),
        })
        .unwrap_err();
    assert!(matches!(err, DirectoryError::InvalidName));
}

#[test]
fn insert_rejects_nul_in_name() {
    let mut dir = Directory::new();
    let err = dir
        .insert(DirEntry {
            kind: EntryKind::File,
            name: "a\0b".to_owned(),
            inode: ptr(0, 0),
        })
        .unwrap_err();
    assert!(matches!(err, DirectoryError::InvalidName));
}

// ------------------------------------------------------------------
// Deserialization rejects malformed input
// ------------------------------------------------------------------

#[test]
fn deserialize_rejects_missing_header() {
    let err = Directory::deserialize(&[]).unwrap_err();
    assert!(matches!(err, DirectoryError::TruncatedHeader(0)));
    let err = Directory::deserialize(&[0, 0]).unwrap_err();
    assert!(matches!(err, DirectoryError::TruncatedHeader(2)));
}

#[test]
fn deserialize_rejects_missing_entry_header() {
    // Count=1, but no entry bytes follow.
    let bytes = [1u8, 0, 0, 0];
    let err = Directory::deserialize(&bytes).unwrap_err();
    assert!(matches!(err, DirectoryError::TruncatedEntryHeader { .. }));
}

#[test]
fn deserialize_rejects_missing_name_bytes() {
    // Count=1, entry header claims name_len=5, no name bytes follow.
    let mut bytes = vec![1u8, 0, 0, 0];
    bytes.push(0); // kind = file
    bytes.push(5); // name_len
    bytes.extend_from_slice(&[0, 0]); // reserved
    bytes.extend_from_slice(&ptr(0, 0).encode());
    // Missing 5 name bytes.
    let err = Directory::deserialize(&bytes).unwrap_err();
    assert!(matches!(err, DirectoryError::TruncatedName { .. }));
}

#[test]
fn deserialize_rejects_invalid_kind() {
    let mut bytes = vec![1u8, 0, 0, 0];
    bytes.push(99); // invalid kind
    bytes.push(1);
    bytes.extend_from_slice(&[0, 0]);
    bytes.extend_from_slice(&ptr(0, 0).encode());
    bytes.push(b'x');
    let err = Directory::deserialize(&bytes).unwrap_err();
    assert!(matches!(err, DirectoryError::InvalidKind(99)));
}

#[test]
fn deserialize_rejects_nonzero_reserved_bytes() {
    let mut bytes = vec![1u8, 0, 0, 0];
    bytes.push(0); // file
    bytes.push(1);
    bytes.extend_from_slice(&[0xAA, 0xBB]); // reserved must stay zero
    bytes.extend_from_slice(&ptr(0, 0).encode());
    bytes.push(b'x');
    let err = Directory::deserialize(&bytes).unwrap_err();
    assert!(matches!(err, DirectoryError::ReservedNotZero { .. }));
}

#[test]
fn deserialize_rejects_trailing_bytes() {
    let dir = Directory::new();
    let mut bytes = dir.serialize();
    bytes.push(0xAA); // trailing junk
    let err = Directory::deserialize(&bytes).unwrap_err();
    assert!(matches!(err, DirectoryError::TrailingBytes { count: 1 }));
}

#[test]
fn deserialize_rejects_duplicate_names_in_blob() {
    // Two entries with the same name "x".
    let mut bytes = vec![2u8, 0, 0, 0];
    for _ in 0..2 {
        bytes.push(0); // file
        bytes.push(1);
        bytes.extend_from_slice(&[0, 0]);
        bytes.extend_from_slice(&ptr(0, 0).encode());
        bytes.push(b'x');
    }
    let err = Directory::deserialize(&bytes).unwrap_err();
    assert!(matches!(err, DirectoryError::NameCollision(_)));
}

#[test]
fn deserialize_rejects_invalid_utf8_name() {
    let mut bytes = vec![1u8, 0, 0, 0];
    bytes.push(0);
    bytes.push(2);
    bytes.extend_from_slice(&[0, 0]);
    bytes.extend_from_slice(&ptr(0, 0).encode());
    // 0xFF 0xFE is not valid UTF-8.
    bytes.extend_from_slice(&[0xFF, 0xFE]);
    let err = Directory::deserialize(&bytes).unwrap_err();
    assert!(matches!(err, DirectoryError::InvalidName));
}
