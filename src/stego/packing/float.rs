use super::{PackingError, QuantPacker, StegoPacker};

pub const NAME: &str = "float";
pub const F16_BYTES_PER_VALUE: usize = 2;
pub const F32_BYTES_PER_VALUE: usize = 4;

pub struct F16Packer;
pub struct F32Packer;

impl StegoPacker for F16Packer {
    const NAME: &'static str = "f16";
    const STEALABLE_BITS_PER_WEIGHT: usize = 4;
}

impl StegoPacker for F32Packer {
    const NAME: &'static str = "f32";
    const STEALABLE_BITS_PER_WEIGHT: usize = 8;
}

impl QuantPacker for F16Packer {
    fn bits_per_weight(&self) -> u32 {
        4
    }
    fn block_size_bytes(&self) -> usize {
        F16_BYTES_PER_VALUE
    }
    fn weights_per_block(&self) -> usize {
        1
    }
    fn extract(&self, block_bytes: &[u8]) -> Vec<u8> {
        assert_eq!(block_bytes.len(), F16_BYTES_PER_VALUE, "f16 block size");
        vec![block_bytes[0] & 0x0F]
    }
    fn embed(&self, block_bytes: &[u8], data: &[u8]) -> Vec<u8> {
        assert_eq!(block_bytes.len(), F16_BYTES_PER_VALUE, "f16 block size");
        assert_eq!(data.len(), 1, "f16 data nibble");
        vec![
            (block_bytes[0] & 0xF0) | (data[0] & 0x0F),
            block_bytes[1],
        ]
    }
    fn stealable_byte_offsets(&self) -> Vec<usize> {
        vec![0]
    }
}

impl QuantPacker for F32Packer {
    fn bits_per_weight(&self) -> u32 {
        8
    }
    fn block_size_bytes(&self) -> usize {
        F32_BYTES_PER_VALUE
    }
    fn weights_per_block(&self) -> usize {
        1
    }
    fn extract(&self, block_bytes: &[u8]) -> Vec<u8> {
        assert_eq!(block_bytes.len(), F32_BYTES_PER_VALUE, "f32 block size");
        vec![block_bytes[0]]
    }
    fn embed(&self, block_bytes: &[u8], data: &[u8]) -> Vec<u8> {
        assert_eq!(block_bytes.len(), F32_BYTES_PER_VALUE, "f32 block size");
        assert_eq!(data.len(), 1, "f32 data byte");
        vec![data[0], block_bytes[1], block_bytes[2], block_bytes[3]]
    }
    fn stealable_byte_offsets(&self) -> Vec<usize> {
        vec![0]
    }
}

pub fn read_f16_nibble(storage: &[u8], value_index: usize) -> Result<u8, PackingError> {
    validate_value_storage(NAME, storage.len(), F16_BYTES_PER_VALUE)?;

    let byte_offset = checked_value_offset(NAME, storage.len(), value_index, F16_BYTES_PER_VALUE)?;
    let value = u16::from_le_bytes([storage[byte_offset], storage[byte_offset + 1]]);
    Ok((value & 0x000F) as u8)
}

pub fn write_f16_nibble(
    storage: &mut [u8],
    value_index: usize,
    nibble: u8,
) -> Result<(), PackingError> {
    validate_value_storage(NAME, storage.len(), F16_BYTES_PER_VALUE)?;

    let byte_offset = checked_value_offset(NAME, storage.len(), value_index, F16_BYTES_PER_VALUE)?;
    let value = u16::from_le_bytes([storage[byte_offset], storage[byte_offset + 1]]);
    let updated = (value & 0xFFF0) | u16::from(nibble & 0x0F);
    storage[byte_offset..byte_offset + 2].copy_from_slice(&updated.to_le_bytes());
    Ok(())
}

pub fn read_f16_payload(storage: &[u8]) -> Result<Vec<u8>, PackingError> {
    validate_value_storage(NAME, storage.len(), F16_BYTES_PER_VALUE)?;

    let value_count = storage.len() / F16_BYTES_PER_VALUE;
    let mut payload = vec![0_u8; value_count.div_ceil(2)];

    for value_index in 0..value_count {
        let nibble = read_f16_nibble(storage, value_index)?;
        let payload_index = value_index / 2;
        if value_index % 2 == 0 {
            payload[payload_index] = nibble;
        } else {
            payload[payload_index] |= nibble << 4;
        }
    }

    Ok(payload)
}

pub fn write_f16_payload(storage: &mut [u8], payload: &[u8]) -> Result<(), PackingError> {
    validate_value_storage(NAME, storage.len(), F16_BYTES_PER_VALUE)?;

    let value_count = storage.len() / F16_BYTES_PER_VALUE;
    let max_payload_bytes = value_count.div_ceil(2);
    if payload.len() > max_payload_bytes {
        return Err(PackingError::PayloadTooLarge {
            context: NAME,
            max_payload_bytes,
            actual: payload.len(),
        });
    }

    for (payload_index, byte) in payload.iter().copied().enumerate() {
        let even_index = payload_index * 2;
        if even_index < value_count {
            write_f16_nibble(storage, even_index, byte & 0x0F)?;
        }
        let odd_index = even_index + 1;
        if odd_index < value_count {
            write_f16_nibble(storage, odd_index, (byte >> 4) & 0x0F)?;
        }
    }

    Ok(())
}

pub fn read_f32_byte(storage: &[u8], value_index: usize) -> Result<u8, PackingError> {
    validate_value_storage(NAME, storage.len(), F32_BYTES_PER_VALUE)?;

    let byte_offset = checked_value_offset(NAME, storage.len(), value_index, F32_BYTES_PER_VALUE)?;
    let value = u32::from_le_bytes([
        storage[byte_offset],
        storage[byte_offset + 1],
        storage[byte_offset + 2],
        storage[byte_offset + 3],
    ]);
    Ok((value & 0x0000_00FF) as u8)
}

pub fn write_f32_byte(
    storage: &mut [u8],
    value_index: usize,
    byte: u8,
) -> Result<(), PackingError> {
    validate_value_storage(NAME, storage.len(), F32_BYTES_PER_VALUE)?;

    let byte_offset = checked_value_offset(NAME, storage.len(), value_index, F32_BYTES_PER_VALUE)?;
    let value = u32::from_le_bytes([
        storage[byte_offset],
        storage[byte_offset + 1],
        storage[byte_offset + 2],
        storage[byte_offset + 3],
    ]);
    let updated = (value & 0xFFFF_FF00) | u32::from(byte);
    storage[byte_offset..byte_offset + 4].copy_from_slice(&updated.to_le_bytes());
    Ok(())
}

pub fn read_f32_payload(storage: &[u8]) -> Result<Vec<u8>, PackingError> {
    validate_value_storage(NAME, storage.len(), F32_BYTES_PER_VALUE)?;

    let value_count = storage.len() / F32_BYTES_PER_VALUE;
    let mut payload = Vec::with_capacity(value_count);
    for value_index in 0..value_count {
        payload.push(read_f32_byte(storage, value_index)?);
    }
    Ok(payload)
}

pub fn write_f32_payload(storage: &mut [u8], payload: &[u8]) -> Result<(), PackingError> {
    validate_value_storage(NAME, storage.len(), F32_BYTES_PER_VALUE)?;

    let value_count = storage.len() / F32_BYTES_PER_VALUE;
    if payload.len() > value_count {
        return Err(PackingError::PayloadTooLarge {
            context: NAME,
            max_payload_bytes: value_count,
            actual: payload.len(),
        });
    }

    for (value_index, byte) in payload.iter().copied().enumerate() {
        write_f32_byte(storage, value_index, byte)?;
    }

    Ok(())
}

fn validate_value_storage(
    context: &'static str,
    storage_len: usize,
    bytes_per_value: usize,
) -> Result<(), PackingError> {
    if storage_len % bytes_per_value != 0 {
        Err(PackingError::InvalidStorageLength {
            context,
            unit: bytes_per_value,
            actual: storage_len,
        })
    } else {
        Ok(())
    }
}

fn checked_value_offset(
    context: &'static str,
    storage_len: usize,
    value_index: usize,
    bytes_per_value: usize,
) -> Result<usize, PackingError> {
    let value_count = storage_len / bytes_per_value;
    if value_index >= value_count {
        return Err(PackingError::IndexOutOfRange {
            context,
            index: value_index,
            len: value_count,
        });
    }

    Ok(value_index * bytes_per_value)
}
