use llmdb::stego::packing::{q3_k, q4_k, q5_k, q6_k};

#[test]
fn q6_k_roundtrips_payload_and_preserves_non_stolen_bits() {
    let mut block = [0_u8; q6_k::BLOCK_BYTES];
    for (index, slot) in block.iter_mut().enumerate() {
        *slot = seeded_byte(index, 0x31);
    }

    let original = block;
    let payload = sequence::<{ q6_k::PAYLOAD_BYTES_PER_BLOCK }>(0x10, 0x13);

    q6_k::write_payload_block(&mut block, &payload).expect("write q6_k payload");
    let roundtrip = q6_k::read_payload_block(&block).expect("read q6_k payload");
    assert_eq!(roundtrip, payload);

    for index in 0..128 {
        assert_eq!(block[index] & 0xCC, original[index] & 0xCC);
    }
    assert_eq!(&block[128..], &original[128..]);
}

#[test]
fn q5_k_roundtrips_payload_and_preserves_non_stolen_bits() {
    let mut block = [0_u8; q5_k::BLOCK_BYTES];
    for (index, slot) in block.iter_mut().enumerate() {
        *slot = seeded_byte(index, 0x47);
    }

    let original = block;
    let payload = sequence::<{ q5_k::PAYLOAD_BYTES_PER_BLOCK }>(0x20, 0x07);

    q5_k::write_payload_block(&mut block, &payload).expect("write q5_k payload");
    let roundtrip = q5_k::read_payload_block(&block).expect("read q5_k payload");
    assert_eq!(roundtrip, payload);

    for index in 0..128 {
        assert_eq!(block[index] & 0xEE, original[index] & 0xEE);
    }
    assert_eq!(&block[128..], &original[128..]);
}

#[test]
fn q4_k_roundtrips_payload_and_preserves_non_stolen_bits() {
    let mut block = [0_u8; q4_k::BLOCK_BYTES];
    for (index, slot) in block.iter_mut().enumerate() {
        *slot = seeded_byte(index, 0x59);
    }

    let original = block;
    let payload = sequence::<{ q4_k::PAYLOAD_BYTES_PER_BLOCK }>(0x30, 0x05);

    q4_k::write_payload_block(&mut block, &payload).expect("write q4_k payload");
    let roundtrip = q4_k::read_payload_block(&block).expect("read q4_k payload");
    assert_eq!(roundtrip, payload);

    for index in 0..128 {
        assert_eq!(block[index] & 0xEE, original[index] & 0xEE);
    }
    assert_eq!(&block[128..], &original[128..]);
}

#[test]
fn q3_k_roundtrips_payload_and_preserves_non_stolen_bits() {
    let mut block = [0_u8; q3_k::BLOCK_BYTES];
    for (index, slot) in block.iter_mut().enumerate() {
        *slot = seeded_byte(index, 0x6D);
    }

    let original = block;
    let payload = sequence::<{ q3_k::PAYLOAD_BYTES_PER_BLOCK }>(0x40, 0x03);

    q3_k::write_payload_block(&mut block, &payload).expect("write q3_k payload");
    let roundtrip = q3_k::read_payload_block(&block).expect("read q3_k payload");
    assert_eq!(roundtrip, payload);

    for index in 0..64 {
        assert_eq!(block[index] & 0xAA, original[index] & 0xAA);
    }
    assert_eq!(&block[64..], &original[64..]);
}

fn seeded_byte(index: usize, salt: u8) -> u8 {
    (index as u8).wrapping_mul(17).wrapping_add(salt)
}

fn sequence<const N: usize>(start: u8, step: u8) -> [u8; N] {
    let mut out = [0_u8; N];
    let mut value = start;
    let mut index = 0;
    while index < N {
        out[index] = value;
        value = value.wrapping_add(step);
        index += 1;
    }
    out
}
