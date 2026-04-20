pub mod float;
pub mod q3_k;
pub mod q4_k;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;

use thiserror::Error;

use crate::gguf::quant::GgufQuantType;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PackingError {
    #[error("invalid storage length for {context}: expected multiple of {unit}, got {actual}")]
    InvalidStorageLength {
        context: &'static str,
        unit: usize,
        actual: usize,
    },
    #[error("payload too large for {context}: max {max_payload_bytes} bytes, got {actual}")]
    PayloadTooLarge {
        context: &'static str,
        max_payload_bytes: usize,
        actual: usize,
    },
    #[error("index {index} out of range for {context}: len {len}")]
    IndexOutOfRange {
        context: &'static str,
        index: usize,
        len: usize,
    },
}

pub trait StegoPacker {
    const NAME: &'static str;
    const STEALABLE_BITS_PER_WEIGHT: usize;
}

/// Unified packer interface per DESIGN-NEW §4. Object-safe so callers can
/// dispatch through `packer_for(quant_type)` without match statements.
pub trait QuantPacker: Sync {
    fn bits_per_weight(&self) -> u32;
    fn block_size_bytes(&self) -> usize;
    fn weights_per_block(&self) -> usize;
    /// Extract the stolen payload bytes from a single quant block.
    /// Panics for types with `bits_per_weight() == 0`.
    fn extract(&self, block_bytes: &[u8]) -> Vec<u8>;
    /// Splice `data` into `block_bytes` and return the modified block.
    /// Panics for types with `bits_per_weight() == 0`.
    fn embed(&self, block_bytes: &[u8], data: &[u8]) -> Vec<u8>;
    /// Byte offsets within a single block that carry stolen bits.
    /// Empty for types with `bits_per_weight() == 0`.
    fn stealable_byte_offsets(&self) -> Vec<usize>;
}

/// Generic "update only the quant blocks that back this payload range" writer.
/// Avoids the whole-tensor decode-modify-encode cycle that `write_stego_bytes`
/// used to perform — for a 4 KB block write that used to replay every quant
/// block in a 1–2 MB tensor, this only touches the handful of blocks that
/// actually overlap the range. See `src/stego/packing/q8_0.rs` for a caller.
pub fn blockwise_write_range<const P: usize>(
    storage: &mut [u8],
    block_bytes: usize,
    start_in_payload: usize,
    data: &[u8],
    read_block: fn(&[u8]) -> Result<[u8; P], PackingError>,
    write_block: fn(&mut [u8], &[u8]) -> Result<(), PackingError>,
) -> Result<(), PackingError> {
    let end_in_payload = start_in_payload + data.len();
    let first_block = start_in_payload / P;
    let last_block_excl = end_in_payload.div_ceil(P);

    let mut data_cursor = 0_usize;
    for block_idx in first_block..last_block_excl {
        let block_storage_start = block_idx * block_bytes;
        let block = &mut storage[block_storage_start..block_storage_start + block_bytes];

        let block_payload_start = block_idx * P;
        let block_payload_end = block_payload_start + P;
        let overlap_start = start_in_payload.max(block_payload_start);
        let overlap_end = end_in_payload.min(block_payload_end);
        let within_start = overlap_start - block_payload_start;
        let within_end = overlap_end - block_payload_start;
        let copy_len = within_end - within_start;

        if within_start == 0 && within_end == P {
            // Full block overwrite — skip the decode.
            write_block(block, &data[data_cursor..data_cursor + copy_len])?;
        } else {
            // Partial block — read current payload, splice, write back.
            let mut payload = read_block(block)?;
            payload[within_start..within_end]
                .copy_from_slice(&data[data_cursor..data_cursor + copy_len]);
            write_block(block, &payload)?;
        }
        data_cursor += copy_len;
    }
    Ok(())
}

/// Companion reader for `blockwise_write_range`. Decodes only the blocks that
/// overlap the requested payload range.
pub fn blockwise_read_range<const P: usize>(
    storage: &[u8],
    block_bytes: usize,
    start_in_payload: usize,
    len: usize,
    read_block: fn(&[u8]) -> Result<[u8; P], PackingError>,
) -> Result<Vec<u8>, PackingError> {
    let end_in_payload = start_in_payload + len;
    let first_block = start_in_payload / P;
    let last_block_excl = end_in_payload.div_ceil(P);

    let mut out = vec![0_u8; len];
    let mut cursor = 0_usize;
    for block_idx in first_block..last_block_excl {
        let block_storage_start = block_idx * block_bytes;
        let block = &storage[block_storage_start..block_storage_start + block_bytes];
        let payload = read_block(block)?;

        let block_payload_start = block_idx * P;
        let block_payload_end = block_payload_start + P;
        let overlap_start = start_in_payload.max(block_payload_start);
        let overlap_end = end_in_payload.min(block_payload_end);
        let within_start = overlap_start - block_payload_start;
        let within_end = overlap_end - block_payload_start;
        let copy_len = within_end - within_start;

        out[cursor..cursor + copy_len].copy_from_slice(&payload[within_start..within_end]);
        cursor += copy_len;
    }
    Ok(out)
}

pub fn supported_packers() -> &'static [&'static str] {
    &[
        q8_0::NAME,
        q6_k::NAME,
        q5_k::NAME,
        q4_k::NAME,
        q3_k::NAME,
        float::NAME,
    ]
}

pub fn packer_for(quant_type: GgufQuantType) -> &'static dyn QuantPacker {
    match quant_type {
        GgufQuantType::Q8_0 => &q8_0::Q8_0Packer,
        GgufQuantType::Q6K => &q6_k::Q6KPacker,
        GgufQuantType::Q5K => &q5_k::Q5KPacker,
        GgufQuantType::Q4K => &q4_k::Q4KPacker,
        GgufQuantType::Q3K => &q3_k::Q3KPacker,
        GgufQuantType::F16 => &float::F16Packer,
        GgufQuantType::F32 => &float::F32Packer,
        GgufQuantType::Q2K
        | GgufQuantType::Q4_0
        | GgufQuantType::Q4_1
        | GgufQuantType::Q5_0
        | GgufQuantType::Q5_1
        | GgufQuantType::Q8_1
        | GgufQuantType::Q8K => &UnsupportedPacker,
    }
}

/// Placeholder packer for quant types that DESIGN-NEW §4 does not support
/// for stego in V1. `bits_per_weight() == 0`; `extract` and `embed` panic.
pub struct UnsupportedPacker;

impl QuantPacker for UnsupportedPacker {
    fn bits_per_weight(&self) -> u32 {
        0
    }
    fn block_size_bytes(&self) -> usize {
        0
    }
    fn weights_per_block(&self) -> usize {
        0
    }
    fn extract(&self, _block_bytes: &[u8]) -> Vec<u8> {
        unreachable!("UnsupportedPacker::extract called on a non-stego quant type")
    }
    fn embed(&self, _block_bytes: &[u8], _data: &[u8]) -> Vec<u8> {
        unreachable!("UnsupportedPacker::embed called on a non-stego quant type")
    }
    fn stealable_byte_offsets(&self) -> Vec<usize> {
        Vec::new()
    }
}
