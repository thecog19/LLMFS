use super::{PackingError, QuantPacker, StegoPacker};

pub const NAME: &str = "float";
pub const F16_BYTES_PER_VALUE: usize = 2;
pub const F32_BYTES_PER_VALUE: usize = 4;

/// Decode IEEE 754 half-precision (fp16) bits to f32. Standard layout:
/// 1 sign bit, 5 exponent bits (bias 15), 10 mantissa bits. Subnormals
/// are promoted; +inf / -inf and NaN preserved. Used by the K-quant
/// decoders (`q4_k`, `q5_k`, `q6_k`) for the per-block fp16 super-scale,
/// by Q8_0 for its per-block fp16 scale, and by the magnitude
/// estimator for raw fp16 weights.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1F;
    let mantissa = bits & 0x3FF;
    if exp == 0 {
        if mantissa == 0 {
            return if sign == 0 { 0.0 } else { -0.0 };
        }
        let sign_f = if sign == 0 { 1.0 } else { -1.0 };
        return sign_f * (mantissa as f32) * 2.0_f32.powi(-24);
    }
    if exp == 31 {
        return if mantissa == 0 {
            if sign == 0 {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            }
        } else {
            f32::NAN
        };
    }
    let f32_bits =
        ((sign as u32) << 31) | (((exp as u32) + (127 - 15)) << 23) | ((mantissa as u32) << 13);
    f32::from_bits(f32_bits)
}

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
        vec![(block_bytes[0] & 0xF0) | (data[0] & 0x0F), block_bytes[1]]
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

/// Touch only the f16 values that back `data` when it lands at
/// `start_in_payload` within the tensor's full payload range. Avoids the
/// `write_f16_payload` full-tensor walk — the optimisation that turned a
/// ~25-minute mkfs into something interactive.
pub fn write_f16_range(
    storage: &mut [u8],
    start_in_payload: usize,
    data: &[u8],
) -> Result<(), PackingError> {
    validate_value_storage(NAME, storage.len(), F16_BYTES_PER_VALUE)?;
    let value_count = storage.len() / F16_BYTES_PER_VALUE;
    for (offset, &byte) in data.iter().enumerate() {
        let payload_idx = start_in_payload + offset;
        let weight_even = payload_idx * 2;
        let weight_odd = weight_even + 1;
        if weight_even < value_count {
            write_f16_nibble(storage, weight_even, byte & 0x0F)?;
        }
        if weight_odd < value_count {
            write_f16_nibble(storage, weight_odd, (byte >> 4) & 0x0F)?;
        }
    }
    Ok(())
}

pub fn read_f16_range(
    storage: &[u8],
    start_in_payload: usize,
    len: usize,
) -> Result<Vec<u8>, PackingError> {
    validate_value_storage(NAME, storage.len(), F16_BYTES_PER_VALUE)?;
    let value_count = storage.len() / F16_BYTES_PER_VALUE;
    let max_payload_bytes = value_count.div_ceil(2);
    if start_in_payload.saturating_add(len) > max_payload_bytes {
        return Err(PackingError::PayloadTooLarge {
            context: NAME,
            max_payload_bytes,
            actual: start_in_payload.saturating_add(len),
        });
    }
    let mut out = vec![0_u8; len];
    for (offset, dst) in out.iter_mut().enumerate() {
        let payload_idx = start_in_payload + offset;
        let weight_even = payload_idx * 2;
        let weight_odd = weight_even + 1;
        let lo = if weight_even < value_count {
            read_f16_nibble(storage, weight_even)?
        } else {
            0
        };
        let hi = if weight_odd < value_count {
            read_f16_nibble(storage, weight_odd)?
        } else {
            0
        };
        *dst = lo | (hi << 4);
    }
    Ok(out)
}

pub fn write_f32_range(
    storage: &mut [u8],
    start_in_payload: usize,
    data: &[u8],
) -> Result<(), PackingError> {
    validate_value_storage(NAME, storage.len(), F32_BYTES_PER_VALUE)?;
    let value_count = storage.len() / F32_BYTES_PER_VALUE;
    if start_in_payload.saturating_add(data.len()) > value_count {
        return Err(PackingError::PayloadTooLarge {
            context: NAME,
            max_payload_bytes: value_count,
            actual: start_in_payload.saturating_add(data.len()),
        });
    }
    for (offset, &byte) in data.iter().enumerate() {
        write_f32_byte(storage, start_in_payload + offset, byte)?;
    }
    Ok(())
}

pub fn read_f32_range(
    storage: &[u8],
    start_in_payload: usize,
    len: usize,
) -> Result<Vec<u8>, PackingError> {
    validate_value_storage(NAME, storage.len(), F32_BYTES_PER_VALUE)?;
    let value_count = storage.len() / F32_BYTES_PER_VALUE;
    if start_in_payload.saturating_add(len) > value_count {
        return Err(PackingError::PayloadTooLarge {
            context: NAME,
            max_payload_bytes: value_count,
            actual: start_in_payload.saturating_add(len),
        });
    }
    let mut out = vec![0_u8; len];
    for (offset, dst) in out.iter_mut().enumerate() {
        *dst = read_f32_byte(storage, start_in_payload + offset)?;
    }
    Ok(out)
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
    if storage_len.is_multiple_of(bytes_per_value) {
        Ok(())
    } else {
        Err(PackingError::InvalidStorageLength {
            context,
            unit: bytes_per_value,
            actual: storage_len,
        })
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
