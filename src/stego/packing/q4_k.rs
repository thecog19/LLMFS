use super::{PackingError, QuantPacker, StegoPacker};

pub const NAME: &str = "q4_k";
pub const BLOCK_BYTES: usize = 144;
pub const PAYLOAD_BYTES_PER_BLOCK: usize = 32;
pub const WEIGHTS_PER_BLOCK: usize = 256;
const QS_BYTES: usize = 128;

pub struct Q4KPacker;

impl StegoPacker for Q4KPacker {
    const NAME: &'static str = NAME;
    const STEALABLE_BITS_PER_WEIGHT: usize = 1;
}

impl QuantPacker for Q4KPacker {
    fn bits_per_weight(&self) -> u32 {
        1
    }
    fn block_size_bytes(&self) -> usize {
        BLOCK_BYTES
    }
    fn weights_per_block(&self) -> usize {
        WEIGHTS_PER_BLOCK
    }
    fn extract(&self, block_bytes: &[u8]) -> Vec<u8> {
        read_payload_block(block_bytes)
            .expect("q4_k extract on invalid block")
            .to_vec()
    }
    fn embed(&self, block_bytes: &[u8], data: &[u8]) -> Vec<u8> {
        let mut block = block_bytes.to_vec();
        write_payload_block(&mut block, data).expect("q4_k embed on invalid block");
        block
    }
    fn stealable_byte_offsets(&self) -> Vec<usize> {
        (0..QS_BYTES).collect()
    }
}

pub fn read_payload_block(block: &[u8]) -> Result<[u8; PAYLOAD_BYTES_PER_BLOCK], PackingError> {
    if block.len() != BLOCK_BYTES {
        return Err(PackingError::InvalidStorageLength {
            context: NAME,
            unit: BLOCK_BYTES,
            actual: block.len(),
        });
    }

    let mut payload = [0_u8; PAYLOAD_BYTES_PER_BLOCK];
    for (payload_index, slot) in payload.iter_mut().enumerate() {
        let base = payload_index * 4;
        *slot = read_carrier_bits(block[base])
            | (read_carrier_bits(block[base + 1]) << 2)
            | (read_carrier_bits(block[base + 2]) << 4)
            | (read_carrier_bits(block[base + 3]) << 6);
    }
    Ok(payload)
}

pub fn write_stego_range(
    storage: &mut [u8],
    start_in_payload: usize,
    data: &[u8],
) -> Result<(), PackingError> {
    super::blockwise_write_range::<PAYLOAD_BYTES_PER_BLOCK>(
        storage,
        BLOCK_BYTES,
        start_in_payload,
        data,
        read_payload_block,
        write_payload_block,
    )
}

pub fn read_stego_range(
    storage: &[u8],
    start_in_payload: usize,
    len: usize,
) -> Result<Vec<u8>, PackingError> {
    super::blockwise_read_range::<PAYLOAD_BYTES_PER_BLOCK>(
        storage,
        BLOCK_BYTES,
        start_in_payload,
        len,
        read_payload_block,
    )
}

pub fn write_payload_block(block: &mut [u8], payload: &[u8]) -> Result<(), PackingError> {
    if block.len() != BLOCK_BYTES {
        return Err(PackingError::InvalidStorageLength {
            context: NAME,
            unit: BLOCK_BYTES,
            actual: block.len(),
        });
    }
    if payload.len() != PAYLOAD_BYTES_PER_BLOCK {
        return Err(PackingError::PayloadTooLarge {
            context: NAME,
            max_payload_bytes: PAYLOAD_BYTES_PER_BLOCK,
            actual: payload.len(),
        });
    }

    for (payload_index, byte) in payload.iter().copied().enumerate() {
        let base = payload_index * 4;
        write_carrier_bits(&mut block[base], byte & 0x03);
        write_carrier_bits(&mut block[base + 1], (byte >> 2) & 0x03);
        write_carrier_bits(&mut block[base + 2], (byte >> 4) & 0x03);
        write_carrier_bits(&mut block[base + 3], (byte >> 6) & 0x03);
    }
    Ok(())
}

fn read_carrier_bits(byte: u8) -> u8 {
    (byte & 0x01) | (((byte >> 4) & 0x01) << 1)
}

fn write_carrier_bits(slot: &mut u8, bits: u8) {
    let preserved = *slot & 0xEE;
    *slot = preserved | (bits & 0x01) | (((bits >> 1) & 0x01) << 4);
}

#[allow(dead_code)]
const _: usize = QS_BYTES;
