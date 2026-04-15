use super::{PackingError, StegoPacker};

pub const NAME: &str = "q6_k";
pub const BLOCK_BYTES: usize = 210;
pub const PAYLOAD_BYTES_PER_BLOCK: usize = 64;
const QL_BYTES: usize = 128;

pub struct Q6KPacker;

impl StegoPacker for Q6KPacker {
    const NAME: &'static str = NAME;
    const STEALABLE_BITS_PER_WEIGHT: usize = 2;
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
        let lo = read_carrier_nibble(block[payload_index * 2]);
        let hi = read_carrier_nibble(block[payload_index * 2 + 1]);
        *slot = lo | (hi << 4);
    }
    Ok(payload)
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
        write_carrier_nibble(&mut block[payload_index * 2], byte & 0x0F);
        write_carrier_nibble(&mut block[payload_index * 2 + 1], (byte >> 4) & 0x0F);
    }
    Ok(())
}

fn read_carrier_nibble(byte: u8) -> u8 {
    (byte & 0x03) | (((byte >> 4) & 0x03) << 2)
}

fn write_carrier_nibble(slot: &mut u8, nibble: u8) {
    let preserved = *slot & 0xCC;
    *slot = preserved | (nibble & 0x03) | (((nibble >> 2) & 0x03) << 4);
}

#[allow(dead_code)]
const _: usize = QL_BYTES;
