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

const SCALES_BYTES: usize = 12;
const SCALES_OFFSET: usize = 4; // after d (2) + dmin (2)
const QS_OFFSET: usize = SCALES_OFFSET + SCALES_BYTES;

/// Decode the value of a single Q4_K weight (`weight_index` in
/// `0..256`). Mirrors ggml's `dequantize_row_q4_K`. Block layout:
///
/// ```text
/// 0   ..  2   d        — fp16 super-block scale (for sub-block scales)
/// 2   ..  4   dmin     — fp16 super-block scale (for sub-block mins)
/// 4   .. 16   scales   — 12 bytes packing 8x6-bit scales + 8x6-bit mins
/// 16  ..144   qs       — 256 4-bit quants packed two per byte
/// ```
///
/// Per weight: `value = d * sc * nibble - dmin * m`, where `(sc, m)`
/// are the sub-block scale/min retrieved via `get_scale_min_k4` and
/// `nibble` is the appropriate 4 bits from `qs`. Returns `Err` if
/// `block` is the wrong length or `weight_index` is out of range.
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

    let d_bits = u16::from_le_bytes([block[0], block[1]]);
    let dmin_bits = u16::from_le_bytes([block[2], block[3]]);
    let d = super::float::f16_to_f32(d_bits);
    let dmin = super::float::f16_to_f32(dmin_bits);
    let scales = &block[SCALES_OFFSET..SCALES_OFFSET + SCALES_BYTES];
    let qs = &block[QS_OFFSET..QS_OFFSET + QS_BYTES];

    // 256 weights split into 4 outer iterations of 64; each outer
    // iteration has two halves (32 weights each, low/high nibble of
    // qs). Eight sub-blocks total, indexed by `2*j_outer + half` (0..8).
    let j_outer = weight_index / 64;
    let within = weight_index % 64;
    let half = within / 32;
    let l = within % 32;

    let sub_block_idx = 2 * j_outer + half;
    let (sc, m) = get_scale_min_k4(sub_block_idx, scales);

    let qs_byte = qs[j_outer * 32 + l];
    let nibble = if half == 0 {
        qs_byte & 0x0F
    } else {
        qs_byte >> 4
    };

    let d_pair = d * sc as f32;
    let m_pair = dmin * m as f32;
    Ok(d_pair * nibble as f32 - m_pair)
}

/// Reconstruct the 6-bit `(scale, min)` pair for sub-block `j`
/// (0..8) from the 12-byte packed `scales` array. Mirrors ggml's
/// `get_scale_min_k4`: each pair is 6+6 bits and the upper 2 bits
/// for `j ≥ 4` are split off into a different source byte.
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        let sc = scales[j] & 0x3F;
        let m = scales[j + 4] & 0x3F;
        (sc, m)
    } else {
        let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

#[cfg(test)]
mod read_value_tests {
    use super::*;

    fn block_with(d_fp16: u16, dmin_fp16: u16, scales: [u8; 12], qs_value: u8) -> Vec<u8> {
        let mut b = vec![0_u8; BLOCK_BYTES];
        b[0] = (d_fp16 & 0xFF) as u8;
        b[1] = ((d_fp16 >> 8) & 0xFF) as u8;
        b[2] = (dmin_fp16 & 0xFF) as u8;
        b[3] = ((dmin_fp16 >> 8) & 0xFF) as u8;
        b[SCALES_OFFSET..SCALES_OFFSET + SCALES_BYTES].copy_from_slice(&scales);
        b[QS_OFFSET..QS_OFFSET + QS_BYTES].fill(qs_value);
        b
    }

    #[test]
    fn read_weight_value_zero_quants_yields_negative_min() {
        // d=1.0, dmin=1.0, sub-block 0: sc=1 m=1, qs=0
        // value = 1*1*0 - 1*1 = -1
        let mut scales = [0_u8; 12];
        scales[0] = 1;
        scales[4] = 1;
        let b = block_with(0x3C00, 0x3C00, scales, 0x00);
        assert_eq!(read_weight_value(&b, 0).unwrap(), -1.0);
    }

    #[test]
    fn read_weight_value_picks_low_nibble_for_first_half() {
        // d=1.0, dmin=0, sc=1, qs = 0x35 (low 5, high 3) → weight 0 = 5
        let mut scales = [0_u8; 12];
        scales[0] = 1;
        let b = block_with(0x3C00, 0x0000, scales, 0x35);
        assert_eq!(read_weight_value(&b, 0).unwrap(), 5.0);
    }

    #[test]
    fn read_weight_value_picks_high_nibble_for_second_half() {
        // weight 32 (j_outer=0, half=1) → high nibble of qs[0], scale = scales[1]
        let mut scales = [0_u8; 12];
        scales[1] = 1;
        let b = block_with(0x3C00, 0x0000, scales, 0x35);
        assert_eq!(read_weight_value(&b, 32).unwrap(), 3.0);
    }

    #[test]
    fn get_scale_min_k4_packs_first_four_in_low_six_bits() {
        let mut scales = [0_u8; 12];
        scales[0] = 5;
        scales[3] = 8;
        scales[4] = 9;
        scales[7] = 12;
        assert_eq!(get_scale_min_k4(0, &scales), (5, 9));
        assert_eq!(get_scale_min_k4(3, &scales), (8, 12));
    }

    #[test]
    fn get_scale_min_k4_unpacks_split_for_j_ge_four() {
        // Pack scale = 21 (0b010101), min = 42 (0b101010) for sub-block 4:
        //   scales[8] low nibble = 0b0101 (sc low 4)
        //   scales[8] high nibble = 0b1010 (m low 4)  → 0xA5
        //   scales[0] high 2 bits = 0b01 (sc high 2) → 0x40
        //   scales[4] high 2 bits = 0b10 (m high 2)  → 0x80
        let mut scales = [0_u8; 12];
        scales[0] = 0x40;
        scales[4] = 0x80;
        scales[8] = 0xA5;
        let (sc, m) = get_scale_min_k4(4, &scales);
        assert_eq!(sc, 21);
        assert_eq!(m, 42);
    }

    #[test]
    fn read_weight_value_index_out_of_range_errors() {
        let b = vec![0_u8; BLOCK_BYTES];
        assert!(read_weight_value(&b, 256).is_err());
    }

    #[test]
    fn read_weight_value_wrong_block_length_errors() {
        let too_short = vec![0_u8; 100];
        assert!(read_weight_value(&too_short, 0).is_err());
    }
}
