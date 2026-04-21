//! V2 inode codec.
//!
//! Inode = the addressing unit for any file (user file, directory,
//! internal metadata structure). Per DESIGN-NEW §15.3:
//!
//! ```text
//! offset size  field
//! 0      8     length u64
//! 8      192   direct[12]  (12 × 16-byte Pointer)
//! 200    16    single_indirect Pointer
//! 216    16    double_indirect Pointer
//! 232    16    triple_indirect Pointer
//! 248
//! ```
//!
//! Step 5 lands the codec only — direct / indirect pointer
//! traversal arrives in later milestones. The full layout is
//! established now so later milestones don't need to bump on-disk
//! format.

use llmdb::v2::inode::{INODE_BYTES, Inode, InodeError, NUM_DIRECT};
use llmdb::v2::pointer::Pointer;

#[test]
fn size_matches_documented_layout() {
    // 8 (length) + 12 direct + 3 indirect = 15 pointers × 16 bytes + 8
    assert_eq!(INODE_BYTES, 8 + 15 * Pointer::SIZE);
    assert_eq!(INODE_BYTES, 248);
    assert_eq!(NUM_DIRECT, 12);
}

#[test]
fn empty_encodes_to_all_zeros() {
    let bytes = Inode::EMPTY.encode();
    assert_eq!(bytes, [0u8; INODE_BYTES]);
}

#[test]
fn empty_round_trip() {
    let bytes = Inode::EMPTY.encode();
    let decoded = Inode::decode(&bytes).expect("decode empty");
    assert_eq!(decoded, Inode::EMPTY);
    assert_eq!(decoded.length, 0);
    assert!(decoded.direct.iter().all(|p| p.is_null()));
    assert!(decoded.single_indirect.is_null());
    assert!(decoded.double_indirect.is_null());
    assert!(decoded.triple_indirect.is_null());
}

#[test]
fn non_trivial_round_trip() {
    let mut direct = [Pointer::NULL; NUM_DIRECT];
    for (i, p) in direct.iter_mut().enumerate() {
        *p = Pointer {
            slot: i as u16,
            start_weight: (i as u32) * 1000,
            length_in_bits: 4096 * 8,
            flags: 0,
            reserved: 0,
        };
    }
    let inode = Inode {
        length: 48_000,
        direct,
        single_indirect: Pointer {
            slot: 100,
            start_weight: 500_000,
            length_in_bits: 32_768,
            flags: 0,
            reserved: 0,
        },
        double_indirect: Pointer::NULL,
        triple_indirect: Pointer::NULL,
    };
    let bytes = inode.encode();
    let decoded = Inode::decode(&bytes).expect("round trip");
    assert_eq!(decoded, inode);
}

#[test]
fn decode_rejects_truncated() {
    let bytes = [0u8; INODE_BYTES - 1];
    let err = Inode::decode(&bytes).expect_err("truncated");
    matches!(err, InodeError::Truncated { .. });
}

#[test]
fn encode_is_little_endian_length() {
    let inode = Inode {
        length: 0x0102_0304_0506_0708,
        ..Inode::EMPTY
    };
    let bytes = inode.encode();
    assert_eq!(
        &bytes[0..8],
        &[0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01],
    );
}

#[test]
fn direct_pointers_occupy_bytes_8_through_200() {
    let mut inode = Inode::EMPTY;
    inode.direct[0] = Pointer {
        slot: 0x0102,
        start_weight: 0x03040506,
        length_in_bits: 0x07080900,
        flags: 0x0A0B,
        reserved: 0,
    };
    let bytes = inode.encode();
    // Pointer encoding (see tests/v2_pointer.rs):
    //   slot LE | start_weight LE | length_in_bits LE | flags LE | reserved LE
    assert_eq!(
        &bytes[8..8 + Pointer::SIZE],
        &[
            0x02, 0x01, // slot
            0x06, 0x05, 0x04, 0x03, // start_weight
            0x00, 0x09, 0x08, 0x07, // length_in_bits
            0x0B, 0x0A, // flags
            0, 0, 0, 0, // reserved
        ],
    );
}

#[test]
fn indirect_pointers_are_at_200_216_232() {
    let mut inode = Inode::EMPTY;
    inode.single_indirect = Pointer {
        slot: 1,
        ..Pointer::NULL
    };
    inode.double_indirect = Pointer {
        slot: 2,
        ..Pointer::NULL
    };
    inode.triple_indirect = Pointer {
        slot: 3,
        ..Pointer::NULL
    };
    let bytes = inode.encode();
    assert_eq!(&bytes[200..202], &[0x01, 0x00]);
    assert_eq!(&bytes[216..218], &[0x02, 0x00]);
    assert_eq!(&bytes[232..234], &[0x03, 0x00]);
}

#[test]
fn decode_propagates_pointer_errors() {
    // Build a buffer with a non-zero reserved field in direct[0].
    let mut bytes = [0u8; INODE_BYTES];
    // length = 0
    // direct[0] slot bytes = 8..10, start = 10..14, len = 14..18, flags = 18..20,
    // reserved = 20..24
    bytes[20] = 0xFF; // non-zero reserved in direct[0]
    bytes[14] = 0x01; // length_in_bits = 1 so direct[0] is non-null
    let err = Inode::decode(&bytes).expect_err("non-zero reserved");
    matches!(err, InodeError::PointerDecode { .. });
}
