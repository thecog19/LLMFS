//! V2 chunk I/O — reads / writes a variable-length run of stealable
//! bits addressed by a [`Pointer`]. Stacks on bit_io with the
//! pointer's `(slot, start_weight, length_in_bits)` triple determining
//! which stealable-bit positions the chunk occupies.
//!
//! Position mapping for chunk-bit `i` (`0 ≤ i < length_in_bits`):
//!   `weight_index = pointer.start_weight + i / bits_per_weight`
//!   `bit_index    = i % bits_per_weight`
//!
//! Byte mapping: byte `n` of the chunk occupies chunk-bits `[n*8 ..
//! n*8 + 8)`, LSB first (byte value `b`, bit `k ∈ 0..8` →
//! chunk-bit `n*8 + k` is set iff `(b >> k) & 1 == 1`). Matches the
//! byte_io convention so the two modules are inter-operable.

use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};
use llmdb::v2::chunk::{ChunkError, read_chunk, write_chunk};
use llmdb::v2::pointer::Pointer;

// ------------------------------------------------------------------
// Fixtures
// ------------------------------------------------------------------

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

fn f16_slot(weight_count: u64, data_offset: u64) -> TensorSlot {
    let bits = GgufQuantType::F16.stealable_bits_hint() as u64;
    TensorSlot {
        name: format!("chunk.f16.{weight_count}"),
        quant_type: GgufQuantType::F16,
        tier: TensorTier::Tier1,
        data_offset,
        weight_count,
        stealable_bits_per_weight: bits as usize,
        capacity_bits: weight_count * bits,
        bit_start: 0,
        bit_end: weight_count * bits,
    }
}

fn q8_0_block(values: &[i8], scale: f32) -> (TensorSlot, Vec<u8>) {
    assert!(values.len() <= 32);
    let mut bytes = vec![0_u8; 34];
    bytes[0..2].copy_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
    for (i, v) in values.iter().enumerate() {
        bytes[2 + i] = *v as u8;
    }
    let bits = GgufQuantType::Q8_0.stealable_bits_hint() as u64;
    let slot = TensorSlot {
        name: "chunk.q8_0".to_owned(),
        quant_type: GgufQuantType::Q8_0,
        tier: TensorTier::Tier1,
        data_offset: 0,
        weight_count: values.len() as u64,
        stealable_bits_per_weight: bits as usize,
        capacity_bits: values.len() as u64 * bits,
        bit_start: 0,
        bit_end: values.len() as u64 * bits,
    };
    (slot, bytes)
}

fn f16_cover(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        out.extend_from_slice(&f32_to_f16_bits(*v).to_le_bytes());
    }
    out
}

fn single_slot_map(slot: TensorSlot) -> TensorMap {
    let total = slot.capacity_bits;
    TensorMap {
        slots: vec![slot],
        total_capacity_bits: total,
        total_capacity_bytes: total / 8,
    }
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

#[test]
fn single_byte_round_trip_f16() {
    // 2 F16 weights × 4 stealable bits = 8 bits = 1 byte
    let values = [0.1_f32, 0.2];
    let mut cover = f16_cover(&values);
    let slot = f16_slot(2, 0);
    let map = single_slot_map(slot);

    let ptr = Pointer {
        slot: 0,
        start_weight: 0,
        length_in_bits: 8,
        flags: 0,
        reserved: 0,
    };

    for target in [0x00, 0x01, 0xAA, 0x55, 0xFF] {
        write_chunk(&mut cover, &map, ptr, 0, &[target]).expect("write");
        let mut out = [0u8; 1];
        read_chunk(&cover, &map, ptr, 0, &mut out).expect("read");
        assert_eq!(out[0], target, "roundtrip 0x{target:02x}");
    }
}

#[test]
fn multi_byte_round_trip_f16() {
    // 8 F16 weights × 4 bits = 32 bits = 4 bytes
    let values: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
    let mut cover = f16_cover(&values);
    let slot = f16_slot(8, 0);
    let map = single_slot_map(slot);

    let ptr = Pointer {
        slot: 0,
        start_weight: 0,
        length_in_bits: 32,
        flags: 0,
        reserved: 0,
    };

    let data = [0xDE, 0xAD, 0xBE, 0xEF];
    write_chunk(&mut cover, &map, ptr, 0, &data).expect("write");
    let mut out = [0u8; 4];
    read_chunk(&cover, &map, ptr, 0, &mut out).expect("read");
    assert_eq!(out, data);
}

#[test]
fn byte_offset_lands_at_offset() {
    let values: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
    let mut cover = f16_cover(&values);
    let slot = f16_slot(8, 0);
    let map = single_slot_map(slot);

    let ptr = Pointer {
        slot: 0,
        start_weight: 0,
        length_in_bits: 32,
        flags: 0,
        reserved: 0,
    };

    // Prime all four bytes.
    write_chunk(&mut cover, &map, ptr, 0, &[0, 0, 0, 0]).unwrap();
    write_chunk(&mut cover, &map, ptr, 2, &[0xAB]).unwrap();

    let mut out = [0u8; 4];
    read_chunk(&cover, &map, ptr, 0, &mut out).unwrap();
    assert_eq!(out, [0x00, 0x00, 0xAB, 0x00]);
}

#[test]
fn oob_write_returns_error_and_preserves_cover() {
    let values = [0.1_f32, 0.2];
    let mut cover = f16_cover(&values);
    let slot = f16_slot(2, 0);
    let map = single_slot_map(slot);
    let ptr = Pointer {
        slot: 0,
        start_weight: 0,
        length_in_bits: 8, // only one byte of capacity
        flags: 0,
        reserved: 0,
    };
    let before = cover.clone();

    let err = write_chunk(&mut cover, &map, ptr, 0, &[0xDE, 0xAD])
        .expect_err("writing 2 bytes into 1-byte chunk");
    matches!(err, ChunkError::OutOfBounds { .. });
    assert_eq!(cover, before, "OOB write must not touch the cover");

    let err = write_chunk(&mut cover, &map, ptr, 1, &[0x00])
        .expect_err("writing at offset == capacity");
    matches!(err, ChunkError::OutOfBounds { .. });
    assert_eq!(cover, before);
}

#[test]
fn oob_read_returns_error() {
    let values = [0.1_f32, 0.2];
    let cover = f16_cover(&values);
    let slot = f16_slot(2, 0);
    let map = single_slot_map(slot);
    let ptr = Pointer {
        slot: 0,
        start_weight: 0,
        length_in_bits: 8,
        flags: 0,
        reserved: 0,
    };

    let mut buf = [0u8; 2];
    let err = read_chunk(&cover, &map, ptr, 0, &mut buf).expect_err("too much");
    matches!(err, ChunkError::OutOfBounds { .. });
}

#[test]
fn null_pointer_is_zero_capacity() {
    let cover = f16_cover(&[0.1, 0.2]);
    let slot = f16_slot(2, 0);
    let map = single_slot_map(slot);
    let ptr = Pointer::NULL;
    // Any non-empty read/write should OOB against a null pointer.
    let mut buf = [0u8; 1];
    let err = read_chunk(&cover, &map, ptr, 0, &mut buf).expect_err("null");
    matches!(err, ChunkError::OutOfBounds { .. });
}

#[test]
fn chunk_in_middle_of_slot() {
    // 12 weights, chunk starts at weight 4 and covers 4 weights = 16 bits = 2 bytes
    let values: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
    let mut cover = f16_cover(&values);
    let slot = f16_slot(12, 0);
    let map = single_slot_map(slot);

    let ptr = Pointer {
        slot: 0,
        start_weight: 4,
        length_in_bits: 16,
        flags: 0,
        reserved: 0,
    };

    let data = [0x5A, 0xA5];
    write_chunk(&mut cover, &map, ptr, 0, &data).expect("write");
    let mut out = [0u8; 2];
    read_chunk(&cover, &map, ptr, 0, &mut out).expect("read");
    assert_eq!(out, data);

    // Other chunk that *doesn't* overlap should see its own pristine positions.
    let other = Pointer {
        slot: 0,
        start_weight: 0,
        length_in_bits: 16,
        flags: 0,
        reserved: 0,
    };
    let mut other_out = [0u8; 2];
    read_chunk(&cover, &map, other, 0, &mut other_out).expect("read");
    // Don't assert any specific value — just that the write to
    // positions 4..8 didn't reach positions 0..4.
    // Prove by writing a distinct value and reading both back.
    write_chunk(&mut cover, &map, other, 0, &[0x11, 0x22]).unwrap();
    read_chunk(&cover, &map, other, 0, &mut other_out).unwrap();
    assert_eq!(other_out, [0x11, 0x22]);
    read_chunk(&cover, &map, ptr, 0, &mut out).unwrap();
    assert_eq!(out, data, "writing to earlier chunk must not disturb later one");
}

#[test]
fn q8_0_chunk_round_trip() {
    // One Q8_0 block = 32 weights × 4 stealable bits = 128 bits = 16 bytes
    let ints: Vec<i8> = (0..32).map(|i| (i * 3 - 48) as i8).collect();
    let (slot, mut cover) = q8_0_block(&ints, 0.1);
    let map = single_slot_map(slot);

    let ptr = Pointer {
        slot: 0,
        start_weight: 0,
        length_in_bits: 128,
        flags: 0,
        reserved: 0,
    };

    let data: [u8; 16] = [
        0x00, 0xFF, 0xA5, 0x5A, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99,
        0xAA, 0xBB, 0xCC,
    ];
    write_chunk(&mut cover, &map, ptr, 0, &data).expect("write");
    let mut out = [0u8; 16];
    read_chunk(&cover, &map, ptr, 0, &mut out).expect("read");
    assert_eq!(out, data);
}

#[test]
fn empty_slice_is_noop_in_bounds() {
    let values = [0.1_f32, 0.2];
    let mut cover = f16_cover(&values);
    let slot = f16_slot(2, 0);
    let map = single_slot_map(slot);
    let ptr = Pointer {
        slot: 0,
        start_weight: 0,
        length_in_bits: 8,
        flags: 0,
        reserved: 0,
    };
    let before = cover.clone();

    write_chunk(&mut cover, &map, ptr, 0, &[]).expect("zero-length write");
    assert_eq!(cover, before);

    let mut buf: [u8; 0] = [];
    read_chunk(&cover, &map, ptr, 1, &mut buf).expect("zero-length read");
}

#[test]
fn lsb_first_byte_to_bit_ordering() {
    // Byte 0x01 should set chunk-bit 0 only (which lands on bit 0 of
    // start_weight's stealable bits).
    let values = [0.0_f32, 0.0, 0.0, 0.0]; // zero-mantissa weights
    let mut cover = f16_cover(&values);
    let slot = f16_slot(4, 0);
    let map = single_slot_map(slot);
    let ptr = Pointer {
        slot: 0,
        start_weight: 0,
        length_in_bits: 16,
        flags: 0,
        reserved: 0,
    };

    write_chunk(&mut cover, &map, ptr, 0, &[0x01, 0x00]).unwrap();
    // Low byte of weight 0's fp16 should have bit 0 set, others clear.
    // Weight 0 fp16 starts at cover byte 0.
    assert_eq!(cover[0] & 0x0F, 0x01, "bit 0 only in stealable nibble");
    assert_eq!(cover[2] & 0x0F, 0x00, "weight 1 not touched");
}
