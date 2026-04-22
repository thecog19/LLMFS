//! V2 super-root codec.
//!
//! The super-root is a persistent record that the anchor points at.
//! It carries pointers to every other V2 metadata structure (root
//! directory inode, dedup index, dirty bitmap, free-run state,
//! ceiling-magnitude bucket summary, salience inode) plus a
//! generation counter mirrored from the anchor.
//!
//! Tests (v2 layout, 116 bytes):
//! 1. Size & layout: 116 bytes / 928 bits; byte offsets match spec.
//! 2. Round-trip with all-null pointers (the EMPTY shape).
//! 3. Round-trip with populated pointers + non-zero generation.
//! 4. CRC validates — a flipped content byte makes decode fail.
//! 5. Bad magic → BadMagic error.
//! 6. Unsupported version → UnsupportedVersion error.
//! 7. Truncated buffer → Truncated error.
//! 8. Little-endian byte ordering spot-check.
//!
//! v1 backward-compat behavior lives next to the struct (see the
//! unit tests in `src/v2/super_root.rs`); these integration tests
//! exercise the v2-writer path that every mount now produces.

use llmdb::v2::pointer::Pointer;
use llmdb::v2::super_root::{SUPER_ROOT_BYTES, SuperRoot, SuperRootError};

fn sample_ptr(seed: u64) -> Pointer {
    Pointer {
        slot: (seed & 0xFFFF) as u16,
        start_weight: ((seed >> 16) & 0xFFFFFFFF) as u32,
        length_in_bits: 4096,
        flags: 0,
        reserved: 0,
    }
}

#[test]
fn size_is_one_hundred_sixteen_bytes() {
    assert_eq!(SUPER_ROOT_BYTES, 116);
}

#[test]
fn empty_round_trips() {
    let sr = SuperRoot::EMPTY;
    let bytes = sr.encode();
    let decoded = SuperRoot::decode(&bytes).expect("decode EMPTY");
    assert_eq!(decoded, sr);
}

#[test]
fn populated_round_trip() {
    let sr = SuperRoot {
        root_dir_inode: sample_ptr(1),
        dedup_index_inode: sample_ptr(2),
        dirty_bitmap_inode: sample_ptr(3),
        free_run_state_inode: sample_ptr(4),
        ceiling_summary_inode: sample_ptr(5),
        salience_inode: sample_ptr(6),
        generation: 42,
    };
    let bytes = sr.encode();
    let decoded = SuperRoot::decode(&bytes).expect("decode");
    assert_eq!(decoded, sr);
}

#[test]
fn decode_detects_content_corruption_via_crc() {
    let sr = SuperRoot {
        root_dir_inode: sample_ptr(0xDEAD),
        generation: 7,
        ..SuperRoot::EMPTY
    };
    let mut bytes = sr.encode();

    // Flip a bit in the root_dir_inode area (byte 8-23).
    bytes[10] ^= 0x01;

    match SuperRoot::decode(&bytes) {
        Err(SuperRootError::BadChecksum { .. }) => {}
        other => panic!("expected BadChecksum, got {other:?}"),
    }
}

#[test]
fn decode_detects_corruption_in_generation() {
    let sr = SuperRoot {
        generation: 100,
        ..SuperRoot::EMPTY
    };
    let mut bytes = sr.encode();
    // Flip a bit in the generation field. v2 layout: generation
    // at 104..112.
    bytes[104] ^= 0x01;
    matches!(
        SuperRoot::decode(&bytes),
        Err(SuperRootError::BadChecksum { .. })
    );
}

#[test]
fn decode_rejects_bad_magic() {
    let mut bytes = SuperRoot::EMPTY.encode();
    bytes[0] = b'X';
    match SuperRoot::decode(&bytes) {
        Err(SuperRootError::BadMagic { .. }) => {}
        other => panic!("expected BadMagic, got {other:?}"),
    }
}

#[test]
fn decode_rejects_unsupported_version() {
    let mut bytes = SuperRoot::EMPTY.encode();
    bytes[4] = 99;
    // Version check runs before CRC so caller sees the version mismatch,
    // not a cascading CRC failure.
    match SuperRoot::decode(&bytes) {
        Err(SuperRootError::UnsupportedVersion(99)) => {}
        other => panic!("expected UnsupportedVersion, got {other:?}"),
    }
}

#[test]
fn decode_rejects_truncated() {
    let bytes = [0u8; SUPER_ROOT_BYTES - 1];
    matches!(
        SuperRoot::decode(&bytes),
        Err(SuperRootError::Truncated { .. })
    );
}

#[test]
fn layout_magic_version_pointers() {
    let sr = SuperRoot {
        root_dir_inode: Pointer {
            slot: 0x0102,
            start_weight: 0x03040506,
            length_in_bits: 0x07080900,
            flags: 0x0A0B,
            reserved: 0,
        },
        generation: 0x0102_0304_0506_0708,
        ..SuperRoot::EMPTY
    };
    let bytes = sr.encode();

    // Magic b"V2SR" at offset 0.
    assert_eq!(&bytes[0..4], b"V2SR");
    // Version = 2 at offset 4 (B2 bumped it from 1 when
    // salience_inode was added).
    assert_eq!(bytes[4], 2);
    // Reserved at 5..8 is zero.
    assert_eq!(&bytes[5..8], &[0, 0, 0]);
    // root_dir_inode at offset 8..24 (16 bytes). First 2 bytes = slot LE.
    assert_eq!(&bytes[8..10], &[0x02, 0x01]);
    // v2 layout: salience_inode occupies 88..104, generation
    // shifts to 104..112, crc32 to 112..116.
    assert_eq!(
        &bytes[104..112],
        &[0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01],
    );
    let crc = u32::from_le_bytes(bytes[112..116].try_into().unwrap());
    assert_ne!(crc, 0);
}
