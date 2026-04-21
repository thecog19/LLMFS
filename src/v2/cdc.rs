//! FastCDC — content-defined chunking.
//!
//! Partitions a byte stream into variable-length chunks by sliding
//! a rolling hash (GEAR) over it and cutting at positions where the
//! hash matches a bit pattern. Two masks (stricter before
//! `avg_size`, looser after) aim chunk sizes at `avg_size` while
//! bounding them by `min_size` and `max_size`.
//!
//! **Why content-defined chunking.** Fixed-size chunking at offset
//! multiples of `K` means a byte insertion at offset `N` shifts every
//! downstream chunk boundary by 1 — catastrophic for dedup. FastCDC
//! picks cut points based on the local byte content, so an insertion
//! only perturbs chunks near the insertion point; everything after
//! the next boundary stays identical. This is the property the
//! insertion-stability test verifies and what DESIGN-NEW §15.6 leans
//! on for dedup robustness.
//!
//! **Parameters.** `min_size ≤ avg_size ≤ max_size`, and `avg_size`
//! must be a power of two (the mask trick uses `log2(avg_size)`
//! bits). Production default: 1 KB / 4 KB / 16 KB. Tests typically
//! use smaller sizes so each test exercises multiple chunks.
//!
//! **GEAR table.** 256 × u64 values, deterministic via splitmix64
//! from a fixed seed — all V2 implementations and all builds
//! produce identical chunk boundaries, which is load-bearing for
//! dedup and for cross-session consistency.

use std::ops::Range;
use thiserror::Error;

pub const DEFAULT_MIN_SIZE: usize = 1024;
pub const DEFAULT_AVG_SIZE: usize = 4096;
pub const DEFAULT_MAX_SIZE: usize = 16384;

/// FastCDC parameter bundle. Validate with [`Self::validate`] before
/// use; [`chunk_ranges`] assumes validated parameters and can
/// behave oddly (or panic on shift overflow) otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FastCdcParams {
    pub min_size: usize,
    pub avg_size: usize,
    pub max_size: usize,
}

impl Default for FastCdcParams {
    fn default() -> Self {
        Self {
            min_size: DEFAULT_MIN_SIZE,
            avg_size: DEFAULT_AVG_SIZE,
            max_size: DEFAULT_MAX_SIZE,
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum FastCdcError {
    #[error("avg_size {0} is not a power of two; FastCDC's mask trick needs exactly log2(avg) bits")]
    AvgNotPowerOfTwo(usize),

    #[error("size order invalid: require 0 < min_size ≤ avg_size ≤ max_size; got min={min}, avg={avg}, max={max}")]
    InvalidSizeOrder {
        min: usize,
        avg: usize,
        max: usize,
    },

    #[error("avg_size must be ≥ 4 so bits-1 ≥ 1 bit of looseness in the post-avg mask")]
    AvgTooSmall(usize),
}

impl FastCdcParams {
    pub fn validate(&self) -> Result<(), FastCdcError> {
        if !self.avg_size.is_power_of_two() {
            return Err(FastCdcError::AvgNotPowerOfTwo(self.avg_size));
        }
        if self.avg_size < 4 {
            return Err(FastCdcError::AvgTooSmall(self.avg_size));
        }
        if self.min_size == 0 || self.min_size > self.avg_size || self.max_size < self.avg_size {
            return Err(FastCdcError::InvalidSizeOrder {
                min: self.min_size,
                avg: self.avg_size,
                max: self.max_size,
            });
        }
        Ok(())
    }
}

// ------------------------------------------------------------------
// GEAR table
// ------------------------------------------------------------------

const GEAR: [u64; 256] = build_gear_table();

const fn build_gear_table() -> [u64; 256] {
    let mut table = [0u64; 256];
    let mut state: u64 = 0x1234_5678_9ABC_DEF0;
    let mut i = 0;
    while i < 256 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        table[i] = z ^ (z >> 31);
        i += 1;
    }
    table
}

// ------------------------------------------------------------------
// Public API
// ------------------------------------------------------------------

/// Partition `data` into content-defined chunks. Returns the chunks
/// as half-open ranges; concatenating them covers `0..data.len()`
/// exactly. For empty input returns an empty vec.
///
/// Parameters should have been validated via [`FastCdcParams::validate`];
/// passing unvalidated parameters may panic (shift overflow) on
/// non-power-of-two avg_size, or loop oddly on zero min_size.
pub fn chunk_ranges(data: &[u8], params: &FastCdcParams) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < data.len() {
        let len = next_boundary(&data[start..], params);
        out.push(start..start + len);
        start += len;
    }
    out
}

fn next_boundary(data: &[u8], params: &FastCdcParams) -> usize {
    let len = data.len();
    if len <= params.min_size {
        return len;
    }
    let cap = len.min(params.max_size);
    let bits = params.avg_size.trailing_zeros();
    // Strict mask: bits+1 bits set. Looser mask: bits-1 bits set.
    // Match probability 1/2^(bits+1) before avg, 1/2^(bits-1) after.
    let mask_s = (1u64 << (bits + 1)) - 1;
    let mask_l = (1u64 << (bits - 1)) - 1;

    let mut hash = 0u64;
    let mut i = params.min_size;
    while i < cap {
        hash = hash.wrapping_shl(1).wrapping_add(GEAR[data[i] as usize]);
        let mask = if i < params.avg_size { mask_s } else { mask_l };
        if hash & mask == 0 {
            return i;
        }
        i += 1;
    }
    cap
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gear_table_deterministic_spot_check() {
        // Sanity: the build-time PRNG produces non-zero, non-repeating
        // values. Byte 0 should differ from byte 1.
        assert_ne!(GEAR[0], GEAR[1]);
        assert_ne!(GEAR[0], 0);
        assert_ne!(GEAR[255], 0);
    }

    #[test]
    fn default_params_validate() {
        FastCdcParams::default().validate().unwrap();
    }
}
