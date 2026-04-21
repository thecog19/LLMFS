use super::{PackingError, QuantPacker, StegoPacker};

pub const NAME: &str = "q5_k";
pub const BLOCK_BYTES: usize = 176;
pub const PAYLOAD_BYTES_PER_BLOCK: usize = 32;
pub const WEIGHTS_PER_BLOCK: usize = 256;
const QS_BYTES: usize = 128;

pub struct Q5KPacker;

impl StegoPacker for Q5KPacker {
    const NAME: &'static str = NAME;
    const STEALABLE_BITS_PER_WEIGHT: usize = 1;
}

impl QuantPacker for Q5KPacker {
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
            .expect("q5_k extract on invalid block")
            .to_vec()
    }
    fn embed(&self, block_bytes: &[u8], data: &[u8]) -> Vec<u8> {
        let mut block = block_bytes.to_vec();
        write_payload_block(&mut block, data).expect("q5_k embed on invalid block");
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
const QH_BYTES: usize = 32;
const SCALES_OFFSET: usize = 4;
const QH_OFFSET: usize = SCALES_OFFSET + SCALES_BYTES;
const QS_OFFSET: usize = QH_OFFSET + QH_BYTES;

/// Decode the value of a single Q5_K weight (`weight_index` in
/// `0..256`). Same structure as Q4_K plus an extra 32-byte `qh`
/// array supplying one high bit per weight, so each quant is 5 bits
/// instead of 4. Block layout:
///
/// ```text
/// 0   ..  2   d        — fp16 super-block scale (for sub-block scales)
/// 2   ..  4   dmin     — fp16 super-block scale (for sub-block mins)
/// 4   .. 16   scales   — 12 bytes packing 8x6-bit scales + 8x6-bit mins
/// 16  .. 48   qh       — 256 high bits, one per quant
/// 48  ..176   qs       — 256 4-bit lows packed two per byte
/// ```
///
/// Per weight: `value = d * sc * q5 - dmin * m`, where `q5 =
/// (high1 << 4) | nibble` and `(sc, m)` come from `get_scale_min_k4`
/// (same packing as Q4_K).
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
    let qh = &block[QH_OFFSET..QH_OFFSET + QH_BYTES];
    let qs = &block[QS_OFFSET..QS_OFFSET + QS_BYTES];

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

    // qh[l] packs one high bit per sub-block in bit positions 0..8.
    let high_bit = (qh[l] >> sub_block_idx) & 0x01;
    let q5 = ((high_bit << 4) | nibble) as f32;

    let d_pair = d * sc as f32;
    let m_pair = dmin * m as f32;
    Ok(d_pair * q5 - m_pair)
}

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

    fn block_with(
        d_fp16: u16,
        dmin_fp16: u16,
        scales: [u8; 12],
        qh_value: u8,
        qs_value: u8,
    ) -> Vec<u8> {
        let mut b = vec![0_u8; BLOCK_BYTES];
        b[0] = (d_fp16 & 0xFF) as u8;
        b[1] = ((d_fp16 >> 8) & 0xFF) as u8;
        b[2] = (dmin_fp16 & 0xFF) as u8;
        b[3] = ((dmin_fp16 >> 8) & 0xFF) as u8;
        b[SCALES_OFFSET..SCALES_OFFSET + SCALES_BYTES].copy_from_slice(&scales);
        b[QH_OFFSET..QH_OFFSET + QH_BYTES].fill(qh_value);
        b[QS_OFFSET..QS_OFFSET + QS_BYTES].fill(qs_value);
        b
    }

    #[test]
    fn read_weight_value_with_high_bit_zero_matches_q4_k_shape() {
        // d=1.0, dmin=0, sc=1, qh all zero, qs = 0x35 (low nibble 5).
        // value = 1 * 1 * 5 - 0 = 5
        let mut scales = [0_u8; 12];
        scales[0] = 1;
        let b = block_with(0x3C00, 0x0000, scales, 0x00, 0x35);
        assert_eq!(read_weight_value(&b, 0).unwrap(), 5.0);
    }

    #[test]
    fn read_weight_value_high_bit_adds_sixteen_to_quant() {
        // qh = 0x01 sets bit 0 → sub-block 0 weights gain a high bit.
        // Weight 0: q5 = (1 << 4) | 5 = 21. value = 1*1*21 - 0 = 21
        let mut scales = [0_u8; 12];
        scales[0] = 1;
        let b = block_with(0x3C00, 0x0000, scales, 0x01, 0x35);
        assert_eq!(read_weight_value(&b, 0).unwrap(), 21.0);
    }

    #[test]
    fn read_weight_value_high_bit_position_tracks_sub_block() {
        // qh = 0x02 sets bit 1 → only sub-block 1 (weights 32..63)
        // gains the high bit; sub-block 0 doesn't.
        let mut scales = [0_u8; 12];
        scales[0] = 1;
        scales[1] = 1;
        let b = block_with(0x3C00, 0x0000, scales, 0x02, 0x33);
        assert_eq!(read_weight_value(&b, 0).unwrap(), 3.0); // sub 0, no high bit
        assert_eq!(read_weight_value(&b, 32).unwrap(), 19.0); // sub 1, high bit set
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
