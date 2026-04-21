use super::{PackingError, QuantPacker, StegoPacker};

pub const NAME: &str = "q3_k";
pub const BLOCK_BYTES: usize = 110;
pub const PAYLOAD_BYTES_PER_BLOCK: usize = 32;
pub const WEIGHTS_PER_BLOCK: usize = 256;
const QS_BYTES: usize = 64;

pub struct Q3KPacker;

impl StegoPacker for Q3KPacker {
    const NAME: &'static str = NAME;
    const STEALABLE_BITS_PER_WEIGHT: usize = 1;
}

impl QuantPacker for Q3KPacker {
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
            .expect("q3_k extract on invalid block")
            .to_vec()
    }
    fn embed(&self, block_bytes: &[u8], data: &[u8]) -> Vec<u8> {
        let mut block = block_bytes.to_vec();
        write_payload_block(&mut block, data).expect("q3_k embed on invalid block");
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
    (byte & 0x01)
        | (((byte >> 2) & 0x01) << 1)
        | (((byte >> 4) & 0x01) << 2)
        | (((byte >> 6) & 0x01) << 3)
}

fn write_carrier_nibble(slot: &mut u8, nibble: u8) {
    let preserved = *slot & 0xAA;
    *slot = preserved
        | (nibble & 0x01)
        | (((nibble >> 1) & 0x01) << 2)
        | (((nibble >> 2) & 0x01) << 4)
        | (((nibble >> 3) & 0x01) << 6);
}

#[allow(dead_code)]
const _: usize = QS_BYTES;

// --- Dequantization ---------------------------------------------------
//
// Block layout (mirrors ggml's `block_q3_K`):
//
// ```text
//   0  .. 32   hmask   — 32 bytes, one high bit per weight (256 bits)
//  32  .. 96   qs      — 64 bytes, 4 weights per byte, 2 low bits each
//  96  ..108   scales  — 12 bytes, 16 x 6-bit sub-block scales
// 108  ..110   d       — fp16 super-block scale
// ```
//
// Per weight: `value = d * (raw_scale - 32) * (low_2 - (hmask_bit ? 0 : 4))`
//
// - `low_2` is the 2 bits at position `shift` in the qs byte for this
//   weight. `shift = quadrant * 2`, so the 4 quadrants of a super-half
//   share 32 qs bytes.
// - `hmask_bit` is
//   `(hmask[within_quadrant] >> (super_n * 4 + quadrant)) & 1`. ggml
//   doubles the hmask mask `m` inside the `j` loop, so each of the 8
//   `(super_n, quadrant)` pairs gets its own bit position within the
//   hmask byte (256 weights / 32 bytes = 8 bits/byte used, as expected
//   for 3-bit storage).
// - `raw_scale` is a 6-bit unsigned value stored scattered across the
//   12-byte scales array (see `unpack_scale`); effective scale after
//   subtracting the 32-bias is signed in [-32, 31].

pub const HMASK_BYTES: usize = 32;
pub const SCALES_BYTES: usize = 12;
pub const HMASK_OFFSET: usize = 0;
pub const QS_OFFSET: usize = HMASK_OFFSET + HMASK_BYTES;
pub const SCALES_OFFSET: usize = QS_OFFSET + QS_BYTES;
pub const D_OFFSET: usize = SCALES_OFFSET + SCALES_BYTES;

/// Decode the value of a single Q3_K weight (`weight_index` in
/// `0..256`). Mirrors ggml's `dequantize_row_q3_K`. Returns `Err` if
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

    let hmask = &block[HMASK_OFFSET..HMASK_OFFSET + HMASK_BYTES];
    let qs = &block[QS_OFFSET..QS_OFFSET + QS_BYTES];
    let scales_bytes = &block[SCALES_OFFSET..SCALES_OFFSET + SCALES_BYTES];
    let d_bits = u16::from_le_bytes([block[D_OFFSET], block[D_OFFSET + 1]]);
    let d = super::float::f16_to_f32(d_bits);

    let super_n = weight_index / 128;
    let within_super = weight_index % 128;
    let j = within_super / 32;
    let within_quadrant = within_super % 32;
    let group = within_quadrant / 16;

    let qs_idx = super_n * 32 + within_quadrant;
    let hmask_idx = within_quadrant;
    let scale_idx = super_n * 8 + j * 2 + group;
    let shift = j * 2;

    let low_2 = ((qs[qs_idx] >> shift) & 0x03) as i32;
    let hmask_bit_pos = super_n * 4 + j;
    let hmask_set = (hmask[hmask_idx] >> hmask_bit_pos) & 0x01 != 0;
    let q3 = low_2 - if hmask_set { 0 } else { 4 };

    let scale_raw = unpack_scale(scales_bytes, scale_idx) as i32;
    let scale = scale_raw - 32;

    Ok(d * (scale as f32) * (q3 as f32))
}

/// Extract the 6-bit scale at `scale_idx` (0..16) from the 12-byte
/// packed scales array. Mirrors the ggml aux[0..4] bit-shuffle: each
/// scale splits its low 4 bits into one byte and its high 2 bits into
/// a different byte, packed tightly to fit 16 × 6 = 96 bits into 12
/// bytes.
pub fn unpack_scale(scales_bytes: &[u8], scale_idx: usize) -> u8 {
    debug_assert_eq!(scales_bytes.len(), SCALES_BYTES);
    match scale_idx {
        0..=3 => (scales_bytes[scale_idx] & 0x0F) | ((scales_bytes[scale_idx + 8] & 0x03) << 4),
        4..=7 => {
            (scales_bytes[scale_idx] & 0x0F) | (((scales_bytes[scale_idx + 4] >> 2) & 0x03) << 4)
        }
        8..=11 => {
            ((scales_bytes[scale_idx - 8] >> 4) & 0x0F)
                | (((scales_bytes[scale_idx] >> 4) & 0x03) << 4)
        }
        12..=15 => {
            ((scales_bytes[scale_idx - 8] >> 4) & 0x0F)
                | (((scales_bytes[scale_idx - 4] >> 6) & 0x03) << 4)
        }
        _ => panic!("q3_k scale_idx {scale_idx} out of range 0..16"),
    }
}

#[cfg(test)]
mod read_value_tests {
    use super::*;

    fn empty_block() -> Vec<u8> {
        vec![0_u8; BLOCK_BYTES]
    }

    /// Write `d_fp16` into the super-block scale slot and return the
    /// block — the common starting point for the assertions below.
    fn block_with_d(d_fp16: u16) -> Vec<u8> {
        let mut b = empty_block();
        b[D_OFFSET] = (d_fp16 & 0xFF) as u8;
        b[D_OFFSET + 1] = ((d_fp16 >> 8) & 0xFF) as u8;
        b
    }

    #[test]
    fn all_zero_qs_and_hmask_gives_minus_four_times_minus_thirty_two() {
        // d = 1.0, everything else zero. q3 = 0 - 4 = -4 (hmask bit
        // clear). scale_raw = 0, so effective scale = -32. Weight
        // value = 1 * -32 * -4 = 128 for every weight in the block.
        let b = block_with_d(0x3C00);
        assert_eq!(read_weight_value(&b, 0).unwrap(), 128.0);
        assert_eq!(read_weight_value(&b, 100).unwrap(), 128.0);
        assert_eq!(read_weight_value(&b, 255).unwrap(), 128.0);
    }

    #[test]
    fn hmask_bit_set_zeroes_the_bias() {
        // hmask[0] bit 0 set → weight 0 (super_n=0, wq=0) has q3 =
        // low_2 - 0 = 0 (since low_2 = 0). value = 1 * -32 * 0 = 0.
        let mut b = block_with_d(0x3C00);
        b[HMASK_OFFSET] = 0x01;
        assert_eq!(read_weight_value(&b, 0).unwrap(), 0.0);
        // Weight 128 (super_n=1, wq=0) reads hmask[0] bit 1 — clear,
        // so q3 = 0 - 4 = -4, value = 128 (unchanged).
        assert_eq!(read_weight_value(&b, 128).unwrap(), 128.0);
    }

    #[test]
    fn hmask_quadrant_bit_selects_the_right_bit() {
        // hmask bit = super_n * 4 + j (quadrant). hmask[0] = 0x02
        // sets bit 1, which corresponds to (super_n=0, j=1) — i.e.
        // weights 32..48 with wq=0 (weight 32 here). Weights in any
        // other super/quadrant pair see a clear hmask bit.
        let mut b = block_with_d(0x3C00);
        b[HMASK_OFFSET] = 0x02;
        // Weight 0: (super_n=0, j=0) → bit 0, clear → q3 = -4, value = 128.
        assert_eq!(read_weight_value(&b, 0).unwrap(), 128.0);
        // Weight 32: (super_n=0, j=1) → bit 1, set → q3 = 0, value = 0.
        assert_eq!(read_weight_value(&b, 32).unwrap(), 0.0);
        // Weight 128: (super_n=1, j=0) → bit 4, clear → q3 = -4, value = 128.
        assert_eq!(read_weight_value(&b, 128).unwrap(), 128.0);
    }

    #[test]
    fn hmask_super_n_selects_the_right_bit_group() {
        // Verify the super_n → bit mapping: bit 4 set affects only
        // (super_n=1, j=0) weights, leaving super_n=0 untouched.
        let mut b = block_with_d(0x3C00);
        b[HMASK_OFFSET] = 0x10; // bit 4
        assert_eq!(read_weight_value(&b, 0).unwrap(), 128.0);
        assert_eq!(read_weight_value(&b, 32).unwrap(), 128.0);
        // Weight 128: (super_n=1, j=0) → bit 4 set → q3 = 0.
        assert_eq!(read_weight_value(&b, 128).unwrap(), 0.0);
    }

    #[test]
    fn shift_picks_the_right_two_bits_per_quadrant() {
        // qs[0] = 0x0C (bits 2, 3 set → value 3 at shift 2).
        // Weight 0:  shift=0, low_2 = 0, q3 = -4 → 128
        // Weight 32: shift=2, low_2 = 3, q3 = 3 - 4 = -1 → 1*-32*-1 = 32
        // Weight 64: shift=4, low_2 = 0, q3 = -4 → 128
        // Weight 96: shift=6, low_2 = 0, q3 = -4 → 128
        let mut b = block_with_d(0x3C00);
        b[QS_OFFSET] = 0x0C;
        assert_eq!(read_weight_value(&b, 0).unwrap(), 128.0);
        assert_eq!(read_weight_value(&b, 32).unwrap(), 32.0);
        assert_eq!(read_weight_value(&b, 64).unwrap(), 128.0);
        assert_eq!(read_weight_value(&b, 96).unwrap(), 128.0);
    }

    #[test]
    fn unpack_scale_spans_all_four_byte_groups() {
        // Exercise one scale_idx per bucket of `unpack_scale` and
        // verify the low-4 / high-2 reconstruction matches a hand
        // computation. Pack raw value 33 (0b100001) at indices 0, 5,
        // 9, 14.
        let mut s = [0_u8; SCALES_BYTES];
        // scale 0: low 4 = 0001, high 2 = 10
        //   scales[0] low 4 = 1, scales[8] bit 0,1 = 0b10
        s[0] = 0x01;
        s[8] = 0x02;
        // scale 5: low 4 = 0001, high 2 = 10 via (scales[5+4]>>2)&3 = (scales[9]>>2)&3 = 0b10
        //   scales[5] low 4 = 1, scales[9] bits 2,3 = 0b10
        s[5] = 0x01;
        s[9] = 0x08; // bit 3 set → (0x08 >> 2) & 3 = 0b10
        // scale 9: low 4 = (scales[9-8]>>4)&0xF = (scales[1]>>4)&0xF = 1 → scales[1] = 0x10
        //   high 2 = (scales[9]>>4)&3 = 0b10 → scales[9] bits 4,5 = 0b10
        s[1] = 0x10;
        s[9] |= 0x20; // bit 5 set
        // scale 14: low 4 = (scales[14-8]>>4)&0xF = (scales[6]>>4)&0xF = 1 → scales[6] = 0x10
        //   high 2 = (scales[14-4]>>6)&3 = (scales[10]>>6)&3 = 0b10 → scales[10] bit 6,7 = 0b10
        s[6] = 0x10;
        s[10] = 0x80; // bit 7 set

        assert_eq!(unpack_scale(&s, 0), 33);
        assert_eq!(unpack_scale(&s, 5), 33);
        assert_eq!(unpack_scale(&s, 9), 33);
        assert_eq!(unpack_scale(&s, 14), 33);
    }

    #[test]
    fn scale_bias_is_thirty_two() {
        // scale_raw = 33 for scale 0 (encoded above) → effective = 1.
        // All other fields zero: q3 = -4, value = 1 * 1 * -4 = -4.
        let mut b = block_with_d(0x3C00);
        b[SCALES_OFFSET] = 0x01;
        b[SCALES_OFFSET + 8] = 0x02;
        assert_eq!(read_weight_value(&b, 0).unwrap(), -4.0);
    }

    #[test]
    fn super_block_scale_multiplies_everything() {
        // d = 0.5 (0x3800), scale_raw = 33 (effective 1), q3 = -4.
        // value = 0.5 * 1 * -4 = -2.
        let mut b = block_with_d(0x3800);
        b[SCALES_OFFSET] = 0x01;
        b[SCALES_OFFSET + 8] = 0x02;
        assert_eq!(read_weight_value(&b, 0).unwrap(), -2.0);
    }

    #[test]
    fn read_weight_value_index_out_of_range_errors() {
        let b = block_with_d(0x3C00);
        assert!(read_weight_value(&b, 256).is_err());
        assert!(read_weight_value(&b, usize::MAX).is_err());
    }

    #[test]
    fn read_weight_value_wrong_block_length_errors() {
        let too_short = vec![0_u8; 100];
        assert!(read_weight_value(&too_short, 0).is_err());
        let too_long = vec![0_u8; 200];
        assert!(read_weight_value(&too_long, 0).is_err());
    }
}
