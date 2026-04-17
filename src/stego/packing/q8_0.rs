use super::{PackingError, QuantPacker, StegoPacker};

pub const NAME: &str = "q8_0";
pub const BLOCK_BYTES: usize = 34;
pub const QUANT_COUNT: usize = 32;
pub const PAYLOAD_BYTES_PER_BLOCK: usize = 16;
const SCALE_BYTES: usize = 2;

pub struct Q8_0Packer;

impl StegoPacker for Q8_0Packer {
    const NAME: &'static str = NAME;
    const STEALABLE_BITS_PER_WEIGHT: usize = 4;
}

impl QuantPacker for Q8_0Packer {
    fn bits_per_weight(&self) -> u32 {
        4
    }
    fn block_size_bytes(&self) -> usize {
        BLOCK_BYTES
    }
    fn weights_per_block(&self) -> usize {
        QUANT_COUNT
    }
    fn extract(&self, block_bytes: &[u8]) -> Vec<u8> {
        read_payload_block(block_bytes)
            .expect("q8_0 extract on invalid block")
            .to_vec()
    }
    fn embed(&self, block_bytes: &[u8], data: &[u8]) -> Vec<u8> {
        let mut block = block_bytes.to_vec();
        write_payload_block(&mut block, data).expect("q8_0 embed on invalid block");
        block
    }
    fn stealable_byte_offsets(&self) -> Vec<usize> {
        (SCALE_BYTES..SCALE_BYTES + QUANT_COUNT).collect()
    }
}

pub fn read_quant_nibble(block: &[u8], quant_index: usize) -> Result<u8, PackingError> {
    if block.len() != BLOCK_BYTES {
        return Err(PackingError::InvalidStorageLength {
            context: NAME,
            unit: BLOCK_BYTES,
            actual: block.len(),
        });
    }

    if quant_index >= QUANT_COUNT {
        return Err(PackingError::IndexOutOfRange {
            context: NAME,
            index: quant_index,
            len: QUANT_COUNT,
        });
    }

    Ok(block[SCALE_BYTES + quant_index] & 0x0F)
}

pub fn write_quant_nibble(
    block: &mut [u8],
    quant_index: usize,
    nibble: u8,
) -> Result<(), PackingError> {
    if block.len() != BLOCK_BYTES {
        return Err(PackingError::InvalidStorageLength {
            context: NAME,
            unit: BLOCK_BYTES,
            actual: block.len(),
        });
    }

    if quant_index >= QUANT_COUNT {
        return Err(PackingError::IndexOutOfRange {
            context: NAME,
            index: quant_index,
            len: QUANT_COUNT,
        });
    }

    let slot = &mut block[SCALE_BYTES + quant_index];
    *slot = (*slot & 0xF0) | (nibble & 0x0F);
    Ok(())
}

pub fn read_payload_block(block: &[u8]) -> Result<[u8; PAYLOAD_BYTES_PER_BLOCK], PackingError> {
    let mut payload = [0_u8; PAYLOAD_BYTES_PER_BLOCK];

    for (byte_index, slot) in payload.iter_mut().enumerate() {
        let lo = read_quant_nibble(block, byte_index * 2)?;
        let hi = read_quant_nibble(block, byte_index * 2 + 1)?;
        *slot = lo | (hi << 4);
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
    if payload.len() != PAYLOAD_BYTES_PER_BLOCK {
        return Err(PackingError::PayloadTooLarge {
            context: NAME,
            max_payload_bytes: PAYLOAD_BYTES_PER_BLOCK,
            actual: payload.len(),
        });
    }

    for (byte_index, value) in payload.iter().copied().enumerate() {
        write_quant_nibble(block, byte_index * 2, value & 0x0F)?;
        write_quant_nibble(block, byte_index * 2 + 1, (value >> 4) & 0x0F)?;
    }

    Ok(())
}
