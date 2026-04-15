use super::{PackingError, StegoPacker};

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
