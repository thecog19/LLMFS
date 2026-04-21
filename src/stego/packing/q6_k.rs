use super::{PackingError, QuantPacker, StegoPacker};

pub const NAME: &str = "q6_k";
pub const BLOCK_BYTES: usize = 210;
pub const PAYLOAD_BYTES_PER_BLOCK: usize = 64;
pub const WEIGHTS_PER_BLOCK: usize = 256;
const QL_BYTES: usize = 128;

pub struct Q6KPacker;

impl StegoPacker for Q6KPacker {
    const NAME: &'static str = NAME;
    const STEALABLE_BITS_PER_WEIGHT: usize = 2;
}

impl QuantPacker for Q6KPacker {
    fn bits_per_weight(&self) -> u32 {
        2
    }
    fn block_size_bytes(&self) -> usize {
        BLOCK_BYTES
    }
    fn weights_per_block(&self) -> usize {
        WEIGHTS_PER_BLOCK
    }
    fn extract(&self, block_bytes: &[u8]) -> Vec<u8> {
        read_payload_block(block_bytes)
            .expect("q6_k extract on invalid block")
            .to_vec()
    }
    fn embed(&self, block_bytes: &[u8], data: &[u8]) -> Vec<u8> {
        let mut block = block_bytes.to_vec();
        write_payload_block(&mut block, data).expect("q6_k embed on invalid block");
        block
    }
    fn stealable_byte_offsets(&self) -> Vec<usize> {
        (0..QL_BYTES).collect()
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
        let lo = read_carrier_nibble(block[payload_index * 2]);
        let hi = read_carrier_nibble(block[payload_index * 2 + 1]);
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

/// Decode the value of a single Q6_K weight (`weight_index` in
/// `0..256`). Mirrors ggml's `dequantize_row_q6_K`. Block layout:
///
/// ```text
/// 0   .. 128  ql       — 256 low nibbles (4 bits each, packed two per byte)
/// 128 .. 192  qh       — 256 high pairs (2 bits each, packed four per byte)
/// 192 .. 208  scales   — 16 i8 sub-block scales
/// 208 .. 210  d        — fp16 super-block scale
/// ```
///
/// Per weight: `value = d * scales[scale_idx] * ((q6) - 32)` where
/// `q6` is the reconstructed 6-bit quant (4 low + 2 high), bias 32.
///
/// Returns `Err` if the block is the wrong length or `weight_index` is
/// out of range. Caller is responsible for catching the error; the
/// alternative is silently producing nonsense for malformed inputs.
pub fn read_weight_value(block: &[u8], weight_index: usize) -> Result<f32, PackingError> {
    if block.len() != BLOCK_BYTES {
        return Err(PackingError::InvalidStorageLength {
            context: NAME,
            unit: BLOCK_BYTES,
            actual: block.len(),
        });
    }
    if weight_index >= WEIGHTS_PER_BLOCK {
        return Err(PackingError::IndexOutOfRange {
            context: NAME,
            index: weight_index,
            len: WEIGHTS_PER_BLOCK,
        });
    }

    let ql = &block[0..QL_BYTES];
    let qh = &block[QL_BYTES..QL_BYTES + QH_BYTES];
    let scales = &block[QL_BYTES + QH_BYTES..QL_BYTES + QH_BYTES + SCALES_BYTES];
    let d_bits = u16::from_le_bytes([block[208], block[209]]);
    let d = super::float::f16_to_f32(d_bits);

    // Layout decomposes the 256 weights into two 128-element super-halves;
    // each super-half is 4 quadrants of 32 weights each. Quadrants share
    // ql bytes (low/high nibble) and pack 2 bits per quadrant into qh.
    let super_n = weight_index / 128; // 0 or 1
    let within = weight_index % 128;
    let quadrant = within / 32; // 0..4
    let l = within % 32; // 0..32

    let ql_base = super_n * 64;
    let ql_idx = ql_base + l + (quadrant % 2) * 32;
    let ql_byte = ql[ql_idx];
    let ql_nibble = if quadrant < 2 {
        ql_byte & 0x0F
    } else {
        ql_byte >> 4
    };

    let qh_base = super_n * 32;
    let qh_byte = qh[qh_base + l];
    let qh_2bits = (qh_byte >> (quadrant * 2)) & 0x03;

    let q6 = (ql_nibble | (qh_2bits << 4)) as i32 - 32;

    let scale_idx = super_n * 8 + (l / 16) + quadrant * 2;
    let sc = scales[scale_idx] as i8;

    Ok(d * (sc as f32) * (q6 as f32))
}

const QH_BYTES: usize = 64;
const SCALES_BYTES: usize = 16;

#[cfg(test)]
mod read_value_tests {
    use super::*;

    /// Build a synthetic Q6_K block with explicit field values.
    /// `ql_value` and `qh_value` are written as the same byte to every
    /// position in the respective arrays — useful for verifying the
    /// per-weight decode formula in isolation.
    fn block(d_fp16: u16, scale: i8, ql_value: u8, qh_value: u8) -> Vec<u8> {
        let mut b = vec![0_u8; BLOCK_BYTES];
        b[0..QL_BYTES].fill(ql_value);
        b[QL_BYTES..QL_BYTES + QH_BYTES].fill(qh_value);
        b[QL_BYTES + QH_BYTES..QL_BYTES + QH_BYTES + SCALES_BYTES].fill(scale as u8);
        b[208] = (d_fp16 & 0xFF) as u8;
        b[209] = ((d_fp16 >> 8) & 0xFF) as u8;
        b
    }

    #[test]
    fn read_weight_value_zero_quants_yields_minus_thirty_two_times_scale() {
        // d = 1.0, sc = 1, all ql/qh = 0 → q6 = (0 | 0) - 32 = -32 → value = 1*1*-32 = -32.0
        let b = block(0x3C00, 1, 0x00, 0x00);
        assert_eq!(read_weight_value(&b, 0).unwrap(), -32.0);
        assert_eq!(read_weight_value(&b, 100).unwrap(), -32.0);
        assert_eq!(read_weight_value(&b, 255).unwrap(), -32.0);
    }

    #[test]
    fn read_weight_value_picks_low_nibble_for_first_quadrant_high_for_third() {
        // ql byte = 0x65 → low nibble 5, high nibble 6
        // qh byte = 0x00 (no high bits contributed)
        // d = 1.0, sc = 1
        // weight 0 (super_n=0, quadrant=0, l=0): q6 = (5 | 0) - 32 = -27
        // weight 64 (super_n=0, quadrant=2, l=0): q6 = (6 | 0) - 32 = -26
        let b = block(0x3C00, 1, 0x65, 0x00);
        assert_eq!(read_weight_value(&b, 0).unwrap(), -27.0);
        assert_eq!(read_weight_value(&b, 64).unwrap(), -26.0);
    }

    #[test]
    fn read_weight_value_picks_two_high_bits_per_quadrant() {
        // ql = 0x00, qh = 0b00_01_10_11 = 0x1B
        // weight 0 (quadrant=0): qh_2bits = (0x1B >> 0) & 3 = 3 → q6 = (0 | 0x30) - 32 = 16
        // weight 32 (quadrant=1): qh_2bits = (0x1B >> 2) & 3 = 2 → q6 = (0 | 0x20) - 32 = 0
        // weight 64 (quadrant=2): qh_2bits = (0x1B >> 4) & 3 = 1 → q6 = (0 | 0x10) - 32 = -16
        // weight 96 (quadrant=3): qh_2bits = (0x1B >> 6) & 3 = 0 → q6 = (0 | 0x00) - 32 = -32
        let b = block(0x3C00, 1, 0x00, 0x1B);
        assert_eq!(read_weight_value(&b, 0).unwrap(), 16.0);
        assert_eq!(read_weight_value(&b, 32).unwrap(), 0.0);
        assert_eq!(read_weight_value(&b, 64).unwrap(), -16.0);
        assert_eq!(read_weight_value(&b, 96).unwrap(), -32.0);
    }

    #[test]
    fn read_weight_value_applies_super_block_scale_and_subblock_scale() {
        // d = 0.5 (0x3800), sc = 4. Quants pick value 0 (q6 = -32).
        // value = 0.5 * 4 * -32 = -64
        let b = block(0x3800, 4, 0x00, 0x00);
        assert_eq!(read_weight_value(&b, 0).unwrap(), -64.0);
    }

    #[test]
    fn read_weight_value_addresses_second_super_half() {
        // d = 1.0, sc = 1. ql byte 0x05 → low nibble 5; qh byte = 0
        // weight 128 lives in super_n=1, quadrant=0, l=0. Its ql byte is
        // ql[64] (= 5 here). Decode: (5 | 0) - 32 = -27.
        let b = block(0x3C00, 1, 0x05, 0x00);
        assert_eq!(read_weight_value(&b, 128).unwrap(), -27.0);
    }

    #[test]
    fn read_weight_value_index_out_of_range_errors() {
        let b = block(0x3C00, 1, 0x00, 0x00);
        assert!(read_weight_value(&b, 256).is_err());
        assert!(read_weight_value(&b, 1_000_000).is_err());
    }

    #[test]
    fn read_weight_value_wrong_block_length_errors() {
        let too_short = vec![0_u8; 100];
        assert!(read_weight_value(&too_short, 0).is_err());
    }
}
