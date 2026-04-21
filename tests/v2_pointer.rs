//! V2 `Pointer` type and codec.
//!
//! A `Pointer` addresses a contiguous run of stealable bits in a
//! single tensor slot — the unit every inode direct-pointer and
//! indirect-chunk entry stores. 16 bytes, power-of-two-aligned for
//! predictable in-block packing.
//!
//! Layout (little-endian throughout):
//! ```text
//! offset size  field
//! 0      2     slot             u16
//! 2      4     start_weight     u32
//! 6      4     length_in_bits   u32
//! 10     2     flags            u16
//! 12     4     reserved         u32  (must be zero on encode)
//! ```
//!
//! The null pointer is `length_in_bits == 0`.

use llmdb::v2::pointer::{Pointer, PointerError};

#[test]
fn size_is_sixteen_bytes() {
    assert_eq!(Pointer::SIZE, 16);
    assert_eq!(std::mem::size_of::<[u8; Pointer::SIZE]>(), 16);
}

#[test]
fn null_pointer_round_trips() {
    let p = Pointer::NULL;
    assert!(p.is_null());
    let bytes = p.encode();
    assert_eq!(bytes, [0; 16]);
    let decoded = Pointer::decode(&bytes).expect("null decodes");
    assert_eq!(decoded, p);
    assert!(decoded.is_null());
}

#[test]
fn valid_pointer_round_trips() {
    let p = Pointer {
        slot: 3,
        start_weight: 1_234_567,
        length_in_bits: 32_768,
        flags: 0x0002,
        reserved: 0,
    };
    let bytes = p.encode();
    let decoded = Pointer::decode(&bytes).expect("decode");
    assert_eq!(decoded, p);
    assert!(!decoded.is_null());
}

#[test]
fn decode_rejects_short_buffer() {
    let short = [0u8; 15];
    let err = Pointer::decode(&short).expect_err("too short");
    matches!(err, PointerError::Truncated { .. });
}

#[test]
fn encode_is_little_endian() {
    // slot = 0x0102, start_weight = 0x03040506, length_in_bits = 0x07080900,
    // flags = 0x0A0B, reserved = 0x0C0D0E0F
    let p = Pointer {
        slot: 0x0102,
        start_weight: 0x03040506,
        length_in_bits: 0x07080900,
        flags: 0x0A0B,
        reserved: 0x0C0D0E0F,
    };
    let bytes = p.encode();
    assert_eq!(
        bytes,
        [
            0x02, 0x01, // slot LE
            0x06, 0x05, 0x04, 0x03, // start_weight LE
            0x00, 0x09, 0x08, 0x07, // length_in_bits LE
            0x0B, 0x0A, // flags LE
            0x0F, 0x0E, 0x0D, 0x0C, // reserved LE
        ]
    );
}

#[test]
fn read_at_offset_round_trip() {
    let p1 = Pointer {
        slot: 1,
        start_weight: 100,
        length_in_bits: 1024,
        flags: 0,
        reserved: 0,
    };
    let p2 = Pointer {
        slot: 2,
        start_weight: 200,
        length_in_bits: 2048,
        flags: 0,
        reserved: 0,
    };

    let mut buf = vec![0u8; 40];
    p1.write_at(&mut buf, 0).expect("write p1");
    p2.write_at(&mut buf, 16).expect("write p2");

    let read_p1 = Pointer::read_at(&buf, 0).expect("read p1");
    let read_p2 = Pointer::read_at(&buf, 16).expect("read p2");
    assert_eq!(read_p1, p1);
    assert_eq!(read_p2, p2);
}

#[test]
fn read_at_past_end_returns_error() {
    let buf = vec![0u8; 10];
    let err = Pointer::read_at(&buf, 0).expect_err("buffer too small");
    matches!(err, PointerError::Truncated { .. });

    let buf = vec![0u8; 16];
    // offset + 16 > len
    let err = Pointer::read_at(&buf, 1).expect_err("offset past end");
    matches!(err, PointerError::Truncated { .. });
}

#[test]
fn write_at_past_end_returns_error() {
    let mut buf = vec![0u8; 10];
    let p = Pointer::NULL;
    let err = p.write_at(&mut buf, 0).expect_err("too small");
    matches!(err, PointerError::Truncated { .. });
}

#[test]
fn decode_rejects_non_zero_reserved() {
    // Reserved field must be zero; any non-zero value is an error so
    // forward-compat bits don't silently mean something to us now.
    let bad_bytes = [
        0, 0, // slot
        0, 0, 0, 0, // start_weight
        1, 0, 0, 0, // length_in_bits = 1 (not null)
        0, 0, // flags
        0xFF, 0, 0, 0, // reserved != 0
    ];
    let err = Pointer::decode(&bad_bytes).expect_err("reserved must be zero");
    matches!(err, PointerError::NonZeroReserved { .. });
}
