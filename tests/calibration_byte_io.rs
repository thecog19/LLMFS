//! Byte-level I/O over a `MetadataPlacement`. Stacks on bit_io: byte
//! `N` occupies `placement.positions[N*8 .. N*8+8]`, bit 0 is the LSB
//! of the byte.
//!
//! V2's Layer 1 (§15.3) lays out byte-oriented structures — a
//! superblock, a sensitivity table, a placement index — in the low-
//! magnitude region Layer 0 picks out. The V1 metadata writers are
//! byte-oriented; this adapter lets them target a Layer 0 region
//! without caring about the per-quant bit addressing.
//!
//! Invariants:
//! 1. Round-trip — write then read returns the original bytes.
//! 2. Byte capacity — `byte_capacity = positions.len() / 8`. Tail
//!    bits (`positions.len() % 8`) stay reachable only via bit_io.
//! 3. Bounds — `offset + len == capacity` succeeds; `> capacity`
//!    errors without touching the cover.
//! 4. LSB-first ordering — `write(0x01)` flips position 0 only; the
//!    readback byte has bit 0 set and bits 1..8 clear.
//! 5. Non-aliasing at byte granularity — writing byte `N` flips only
//!    the 8 positions that belong to byte `N`, not its neighbours.
//! 6. Multi-slot placements — a byte whose 8 bits cross the slot
//!    boundary round-trips transparently.
//! 7. Agreement with bit_io — byte writes produce the same cover as
//!    the 8 bit_io writes they decompose into.

use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::calibration::bit_io::{read_bit, write_bit};
use llmdb::stego::calibration::byte_io::{byte_capacity, read_bytes, write_bytes};
use llmdb::stego::calibration::placement::{MetadataPlacement, compute_metadata_placement};
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};

// ------------------------------------------------------------------
// Fixture helpers. Duplicated from `tests/calibration_placement_
// roundtrip.rs`; shared test helpers aren't extracted to keep each
// test file self-contained against drift in the packer conventions.
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

fn f16_slot_with_values(values: &[f32], data_offset: u64) -> (TensorSlot, Vec<u8>) {
    let mut bytes = vec![0_u8; values.len() * 2];
    for (i, v) in values.iter().enumerate() {
        let bits = f32_to_f16_bits(*v);
        bytes[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
    }
    let weight_count = values.len() as u64;
    let bits_per_weight = GgufQuantType::F16.stealable_bits_hint() as u64;
    let slot = TensorSlot {
        name: "byteio.f16".to_owned(),
        quant_type: GgufQuantType::F16,
        tier: TensorTier::Tier1,
        data_offset,
        weight_count,
        stealable_bits_per_weight: bits_per_weight as usize,
        capacity_bits: weight_count * bits_per_weight,
        bit_start: 0,
        bit_end: weight_count * bits_per_weight,
    };
    (slot, bytes)
}

fn q8_0_slot_block(values: &[i8], scale: f32, data_offset: u64) -> (TensorSlot, Vec<u8>) {
    assert!(values.len() <= 32, "single Q8_0 block fixture");
    let mut block = vec![0_u8; 34];
    block[0..2].copy_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
    for (i, v) in values.iter().enumerate() {
        block[2 + i] = *v as u8;
    }
    let weight_count = values.len() as u64;
    let bits_per_weight = GgufQuantType::Q8_0.stealable_bits_hint() as u64;
    let slot = TensorSlot {
        name: "byteio.q8_0".to_owned(),
        quant_type: GgufQuantType::Q8_0,
        tier: TensorTier::Tier1,
        data_offset,
        weight_count,
        stealable_bits_per_weight: bits_per_weight as usize,
        capacity_bits: weight_count * bits_per_weight,
        bit_start: 0,
        bit_end: weight_count * bits_per_weight,
    };
    (slot, block)
}

fn single_slot_map(mut slot: TensorSlot) -> TensorMap {
    slot.bit_start = 0;
    slot.bit_end = slot.capacity_bits;
    let total = slot.capacity_bits;
    TensorMap {
        slots: vec![slot],
        total_capacity_bits: total,
        total_capacity_bytes: total / 8,
    }
}

fn multi_slot_map(slots: Vec<TensorSlot>) -> TensorMap {
    let mut cursor = 0_u64;
    let mut adjusted = Vec::with_capacity(slots.len());
    for mut s in slots {
        s.bit_start = cursor;
        cursor += s.capacity_bits;
        s.bit_end = cursor;
        adjusted.push(s);
    }
    TensorMap {
        slots: adjusted,
        total_capacity_bits: cursor,
        total_capacity_bytes: cursor / 8,
    }
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

#[test]
fn byte_capacity_is_bit_count_floor_divided_by_eight() {
    // 10 F16 weights × 4 bits = 40 bits → 5 bytes, no tail.
    let (slot, _bytes) =
        f16_slot_with_values(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 0);
    let _map = single_slot_map(slot);
    let p = MetadataPlacement {
        positions: (0..40)
            .map(|i| llmdb::stego::calibration::placement::MetadataBitPos {
                slot_index: 0,
                weight_index: (i / 4) as u64,
                bit_index: (i % 4) as u8,
            })
            .collect(),
    };
    assert_eq!(byte_capacity(&p), 5);

    // Truncate to 37 bits — 4 full bytes addressable, 5 bits stranded.
    let p = MetadataPlacement {
        positions: p.positions.into_iter().take(37).collect(),
    };
    assert_eq!(byte_capacity(&p), 4);
}

#[test]
fn single_byte_roundtrips_f16() {
    // 2 F16 weights = 8 stealable bits = exactly one byte.
    let (slot, mut cover) = f16_slot_with_values(&[0.1, 0.2], 0);
    let map = single_slot_map(slot);
    let placement = compute_metadata_placement(&cover, &map, 8);
    assert_eq!(byte_capacity(&placement), 1);

    for target in [0x00, 0x01, 0x7F, 0x80, 0xAA, 0x55, 0xFF] {
        write_bytes(&mut cover, &map, &placement, 0, &[target]).expect("write in-bounds");
        let mut buf = [0_u8; 1];
        read_bytes(&cover, &map, &placement, 0, &mut buf).expect("read in-bounds");
        assert_eq!(buf[0], target, "roundtrip mismatch for 0x{target:02x}");
    }
}

#[test]
fn multi_byte_roundtrips_f16() {
    // 8 F16 weights × 4 bits = 32 bits = 4 bytes.
    let (slot, mut cover) = f16_slot_with_values(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 0);
    let map = single_slot_map(slot);
    let placement = compute_metadata_placement(&cover, &map, 32);
    assert_eq!(byte_capacity(&placement), 4);

    let pattern = [0xDE, 0xAD, 0xBE, 0xEF];
    write_bytes(&mut cover, &map, &placement, 0, &pattern).expect("write in-bounds");
    let mut buf = [0_u8; 4];
    read_bytes(&cover, &map, &placement, 0, &mut buf).expect("read in-bounds");
    assert_eq!(buf, pattern);
}

#[test]
fn write_with_byte_offset_lands_at_offset() {
    let (slot, mut cover) = f16_slot_with_values(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 0);
    let map = single_slot_map(slot);
    let placement = compute_metadata_placement(&cover, &map, 32);

    // Prime all four bytes to 0x00.
    write_bytes(&mut cover, &map, &placement, 0, &[0, 0, 0, 0]).unwrap();
    // Write 0xAB at offset 2.
    write_bytes(&mut cover, &map, &placement, 2, &[0xAB]).unwrap();

    let mut buf = [0_u8; 4];
    read_bytes(&cover, &map, &placement, 0, &mut buf).unwrap();
    assert_eq!(
        buf,
        [0x00, 0x00, 0xAB, 0x00],
        "byte at offset 2 wasn't 0xAB (or bled into neighbours)",
    );
}

#[test]
fn partial_write_preserves_surrounding_bytes() {
    let (slot, mut cover) = f16_slot_with_values(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 0);
    let map = single_slot_map(slot);
    let placement = compute_metadata_placement(&cover, &map, 32);

    // Seed all four bytes with a distinctive pattern.
    let seed = [0x11, 0x22, 0x33, 0x44];
    write_bytes(&mut cover, &map, &placement, 0, &seed).unwrap();

    // Overwrite only byte 1 and 2.
    write_bytes(&mut cover, &map, &placement, 1, &[0xEE, 0xFF]).unwrap();

    let mut buf = [0_u8; 4];
    read_bytes(&cover, &map, &placement, 0, &mut buf).unwrap();
    assert_eq!(buf, [0x11, 0xEE, 0xFF, 0x44]);
}

#[test]
fn write_oob_returns_error_and_does_not_touch_cover() {
    let (slot, mut cover) = f16_slot_with_values(&[0.1, 0.2], 0);
    let map = single_slot_map(slot);
    // Capacity = 1 byte.
    let placement = compute_metadata_placement(&cover, &map, 8);
    assert_eq!(byte_capacity(&placement), 1);

    let before = cover.clone();

    let err = write_bytes(&mut cover, &map, &placement, 0, &[0xDE, 0xAD])
        .expect_err("writing 2 bytes into 1-byte capacity must error");
    assert_eq!(err.offset, 0);
    assert_eq!(err.len, 2);
    assert_eq!(err.capacity_bytes, 1);
    assert_eq!(
        cover, before,
        "OOB write must not have partial side effects"
    );

    // Offset-triggered OOB.
    let err = write_bytes(&mut cover, &map, &placement, 1, &[0x00])
        .expect_err("writing at offset == capacity must error");
    assert_eq!(err.offset, 1);
    assert_eq!(err.len, 1);
    assert_eq!(err.capacity_bytes, 1);
    assert_eq!(cover, before);
}

#[test]
fn read_oob_returns_error() {
    let (slot, cover) = f16_slot_with_values(&[0.1, 0.2], 0);
    let map = single_slot_map(slot);
    let placement = compute_metadata_placement(&cover, &map, 8);

    let mut buf = [0_u8; 2];
    let err = read_bytes(&cover, &map, &placement, 0, &mut buf)
        .expect_err("reading 2 bytes from 1-byte capacity must error");
    assert_eq!(err.offset, 0);
    assert_eq!(err.len, 2);
    assert_eq!(err.capacity_bytes, 1);
}

#[test]
fn empty_io_is_noop() {
    let (slot, mut cover) = f16_slot_with_values(&[0.1, 0.2], 0);
    let map = single_slot_map(slot);
    let placement = compute_metadata_placement(&cover, &map, 8);
    let before = cover.clone();

    // Empty write returns Ok and doesn't touch the cover.
    write_bytes(&mut cover, &map, &placement, 0, &[]).expect("empty write is a no-op");
    assert_eq!(cover, before);

    // Even an out-of-capacity offset with zero-length is fine — the
    // "end" pointer is still within bounds.
    write_bytes(&mut cover, &map, &placement, 1, &[]).expect("empty write at end is a no-op");
    write_bytes(&mut cover, &map, &placement, 999, &[])
        .expect_err("zero-length but beyond-capacity offset still errors for clarity");

    // Empty read returns Ok and doesn't require capacity.
    let mut buf = [0_u8; 0];
    read_bytes(&cover, &map, &placement, 1, &mut buf).expect("empty read is a no-op");
}

#[test]
fn boundary_write_at_last_byte_succeeds() {
    let (slot, mut cover) = f16_slot_with_values(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 0);
    let map = single_slot_map(slot);
    let placement = compute_metadata_placement(&cover, &map, 32); // 4 bytes.
    assert_eq!(byte_capacity(&placement), 4);

    // offset=3, len=1 → end=4 == capacity. Must succeed.
    write_bytes(&mut cover, &map, &placement, 3, &[0x5A]).expect("boundary write at last byte");
    let mut buf = [0_u8; 1];
    read_bytes(&cover, &map, &placement, 3, &mut buf).unwrap();
    assert_eq!(buf[0], 0x5A);

    // offset=0, len=4 → end=4 == capacity. Must succeed.
    write_bytes(&mut cover, &map, &placement, 0, &[0xA1, 0xB2, 0xC3, 0xD4])
        .expect("full-capacity write");
    let mut buf = [0_u8; 4];
    read_bytes(&cover, &map, &placement, 0, &mut buf).unwrap();
    assert_eq!(buf, [0xA1, 0xB2, 0xC3, 0xD4]);
}

#[test]
fn lsb_first_ordering() {
    // The contract: byte N bit k goes to placement.positions[N*8 + k],
    // with k=0 the LSB of the byte. Writing 0x01 flips only positions[0]
    // and leaves positions[1..8] untouched.
    let (slot, mut cover) = f16_slot_with_values(&[0.0, 0.0], 0); // all-zero cover
    let map = single_slot_map(slot);
    let placement = compute_metadata_placement(&cover, &map, 8);

    write_bytes(&mut cover, &map, &placement, 0, &[0x01]).unwrap();
    assert!(
        read_bit(&cover, &map.slots[0], placement.positions[0]),
        "byte 0x01 should set position[0]",
    );
    for i in 1..8 {
        assert!(
            !read_bit(&cover, &map.slots[0], placement.positions[i]),
            "byte 0x01 should leave position[{i}] clear (got set)",
        );
    }

    // 0x80 is bit 7 only.
    write_bytes(&mut cover, &map, &placement, 0, &[0x80]).unwrap();
    assert!(
        !read_bit(&cover, &map.slots[0], placement.positions[0]),
        "byte 0x80 should leave position[0] clear",
    );
    assert!(
        read_bit(&cover, &map.slots[0], placement.positions[7]),
        "byte 0x80 should set position[7]",
    );
    for i in 1..7 {
        assert!(
            !read_bit(&cover, &map.slots[0], placement.positions[i]),
            "byte 0x80 should leave position[{i}] clear",
        );
    }
}

#[test]
fn write_touches_only_byte_n_positions() {
    // Write to byte 1. Confirm positions[0..8] are unchanged and
    // positions[8..16] took the value we wrote.
    let (slot, mut cover) = f16_slot_with_values(&[0.0, 0.0, 0.0, 0.0], 0); // all-zero cover
    let map = single_slot_map(slot);
    let placement = compute_metadata_placement(&cover, &map, 16);

    // Pre-set positions 0..8 to a distinctive value via bit_io so we can
    // detect any accidental clobber.
    for i in 0..8 {
        write_bit(
            &mut cover,
            &map.slots[0],
            placement.positions[i],
            i % 2 == 0,
        );
    }

    write_bytes(&mut cover, &map, &placement, 1, &[0xC3]).unwrap();

    for i in 0..8 {
        let got = read_bit(&cover, &map.slots[0], placement.positions[i]);
        assert_eq!(got, i % 2 == 0, "byte 1 write clobbered position[{i}]");
    }
    // 0xC3 = 0b11000011 → bits 0,1,6,7 set
    let expected = [true, true, false, false, false, false, true, true];
    for (k, exp) in expected.iter().enumerate() {
        let got = read_bit(&cover, &map.slots[0], placement.positions[8 + k]);
        assert_eq!(got, *exp, "byte 1 bit {k} mismatch");
    }
}

#[test]
fn multi_slot_placement_byte_roundtrips() {
    // F16 slot with 2 weights (8 bits) + Q8_0 slot with 8 weights (32
    // bits). Placement spans both. Write 4 bytes; at least one byte
    // must straddle the slot boundary (bits 8..16 cross from F16's bits
    // 0..8 into Q8_0's).
    let (mut f16_slot, f16_bytes) = f16_slot_with_values(&[0.01, 100.0], 0);
    let (mut q8_slot, q8_bytes) =
        q8_0_slot_block(&[1, -3, 2, -4, 0, 0, 0, 0], 0.25, f16_bytes.len() as u64);
    f16_slot.data_offset = 0;
    q8_slot.data_offset = f16_bytes.len() as u64;

    let mut cover = f16_bytes;
    cover.extend(q8_bytes);

    let map = multi_slot_map(vec![f16_slot, q8_slot]);
    // 8 F16 + 32 Q8_0 = 40 bits total → 5-byte capacity.
    let placement = compute_metadata_placement(&cover, &map, 40);
    assert_eq!(byte_capacity(&placement), 5);

    // Placement must span both slots at some point (sanity — this test
    // would be trivial if it didn't).
    let slot_set: std::collections::HashSet<u32> =
        placement.positions.iter().map(|p| p.slot_index).collect();
    assert!(slot_set.len() >= 2, "placement should cover both slots");

    let pattern = [0x10, 0x32, 0x54, 0x76, 0x98];
    write_bytes(&mut cover, &map, &placement, 0, &pattern).unwrap();
    let mut buf = [0_u8; 5];
    read_bytes(&cover, &map, &placement, 0, &mut buf).unwrap();
    assert_eq!(buf, pattern);
}

#[test]
fn byte_write_matches_eight_bit_writes() {
    // Oracle: the cover produced by `write_bytes(.., 0, &[b])` must be
    // identical to the cover produced by 8 `write_bit` calls at
    // positions[0..8] with the LSB-first bit decomposition of `b`.
    for byte in [0x00, 0x01, 0x7F, 0xA5, 0xFF] {
        let (slot, mut via_bytes) = f16_slot_with_values(&[0.0, 0.0, 0.0, 0.0], 0);
        let map = single_slot_map(slot.clone());
        let placement = compute_metadata_placement(&via_bytes, &map, 16);

        write_bytes(&mut via_bytes, &map, &placement, 0, &[byte]).unwrap();

        let (_slot2, mut via_bits) = f16_slot_with_values(&[0.0, 0.0, 0.0, 0.0], 0);
        for k in 0..8 {
            let bit = (byte >> k) & 1 == 1;
            write_bit(&mut via_bits, &map.slots[0], placement.positions[k], bit);
        }

        assert_eq!(
            via_bytes, via_bits,
            "byte 0x{byte:02x} via write_bytes differs from the equivalent bit writes",
        );
    }
}

#[test]
fn roundtrip_q8_0() {
    // Q8_0 carrier — 4 bits per weight. 4 weights = 2 bytes.
    let (slot, mut cover) = q8_0_slot_block(&[1, 2, 3, 4, 0, 0, 0, 0], 1.0, 0);
    let map = single_slot_map(slot);
    let placement = compute_metadata_placement(&cover, &map, 16);
    assert_eq!(byte_capacity(&placement), 2);

    for pair in [[0x00, 0x00], [0xFF, 0xFF], [0xA5, 0x5A], [0x5A, 0xA5]] {
        write_bytes(&mut cover, &map, &placement, 0, &pair).unwrap();
        let mut buf = [0_u8; 2];
        read_bytes(&cover, &map, &placement, 0, &mut buf).unwrap();
        assert_eq!(buf, pair, "Q8_0 roundtrip {pair:02x?}");
    }
}
