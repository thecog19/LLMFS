use llmdb::stego::packing::float::{
    read_f16_payload, read_f32_payload, write_f16_payload, write_f32_payload,
};
use llmdb::stego::packing::q8_0::{
    BLOCK_BYTES, PAYLOAD_BYTES_PER_BLOCK, read_payload_block, write_payload_block,
};

#[test]
fn q8_0_block_roundtrips_payload_and_preserves_scale_and_high_nibbles() {
    let mut block = [0_u8; BLOCK_BYTES];
    block[0] = 0xAB;
    block[1] = 0xCD;
    for (index, slot) in block[2..].iter_mut().enumerate() {
        *slot = 0xA0 | ((index as u8 * 3) & 0x0F);
    }

    let original = block;
    let payload: [u8; PAYLOAD_BYTES_PER_BLOCK] = [
        0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE, 0x0F, 0x1E, 0x2D, 0x3C, 0x4B, 0x5A, 0x69,
        0x78,
    ];

    write_payload_block(&mut block, &payload).expect("write q8 payload");
    let roundtrip = read_payload_block(&block).expect("read q8 payload");
    assert_eq!(roundtrip, payload);

    assert_eq!(block[0], 0xAB);
    assert_eq!(block[1], 0xCD);

    for index in 0..32 {
        assert_eq!(block[2 + index] & 0xF0, original[2 + index] & 0xF0);
    }
}

#[test]
fn f16_payload_roundtrips_and_preserves_upper_twelve_bits() {
    let mut storage = Vec::new();
    for index in 0..8_u16 {
        let value = 0x1230_u16 + (index * 0x111);
        storage.extend_from_slice(&value.to_le_bytes());
    }

    let original = storage.clone();
    let payload = [0x10, 0x32, 0x54, 0x76];

    write_f16_payload(&mut storage, &payload).expect("write f16 payload");
    let roundtrip = read_f16_payload(&storage).expect("read f16 payload");
    assert_eq!(roundtrip, payload);

    for (before, after) in original.chunks_exact(2).zip(storage.chunks_exact(2)) {
        let before_word = u16::from_le_bytes([before[0], before[1]]);
        let after_word = u16::from_le_bytes([after[0], after[1]]);
        assert_eq!(before_word & 0xFFF0, after_word & 0xFFF0);
    }
}

#[test]
fn f32_payload_roundtrips_and_preserves_upper_twenty_four_bits() {
    let mut storage = Vec::new();
    for index in 0..5_u32 {
        let value = 0x1234_5600_u32 + index * 0x0001_0101;
        storage.extend_from_slice(&value.to_le_bytes());
    }

    let original = storage.clone();
    let payload = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE];

    write_f32_payload(&mut storage, &payload).expect("write f32 payload");
    let roundtrip = read_f32_payload(&storage).expect("read f32 payload");
    assert_eq!(roundtrip, payload);

    for (before, after) in original.chunks_exact(4).zip(storage.chunks_exact(4)) {
        let before_word = u32::from_le_bytes([before[0], before[1], before[2], before[3]]);
        let after_word = u32::from_le_bytes([after[0], after[1], after[2], after[3]]);
        assert_eq!(before_word & 0xFFFF_FF00, after_word & 0xFFFF_FF00);
    }
}
