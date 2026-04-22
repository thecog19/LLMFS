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
    #[error(
        "avg_size {0} is not a power of two; FastCDC's mask trick needs exactly log2(avg) bits"
    )]
    AvgNotPowerOfTwo(usize),

    #[error(
        "size order invalid: require 0 < min_size ≤ avg_size ≤ max_size; got min={min}, avg={avg}, max={max}"
    )]
    InvalidSizeOrder { min: usize, avg: usize, max: usize },

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

// ------------------------------------------------------------------
// Streaming chunker — same boundaries as `chunk_ranges`, byte-at-a-time
// ------------------------------------------------------------------

/// State machine version of [`chunk_ranges`]. Feed bytes one at a time
/// via [`Self::feed`]; receive chunks (as owned `Vec<u8>`) at content-
/// defined boundaries. Call [`Self::flush`] at end-of-stream to emit
/// the final partial chunk.
///
/// Used for the dirty-bitmap persist path, where the input is too
/// large to materialise as a single `&[u8]` (a 280 GB cover gives
/// a 17 GB bitmap). Allocates at most one in-flight chunk buffer
/// (≤ `max_size` bytes).
///
/// Boundary equivalence with [`chunk_ranges`] is enforced by the
/// `streaming_matches_slice` property test below.
#[derive(Debug)]
pub struct FastCdcStream {
    params: FastCdcParams,
    pending: Vec<u8>,
    hash: u64,
    mask_s: u64,
    mask_l: u64,
}

impl FastCdcStream {
    /// Create a stream with the given parameters. Caller is
    /// responsible for having validated the params via
    /// [`FastCdcParams::validate`] — invalid params will silently
    /// produce wrong boundaries (or panic).
    pub fn new(params: FastCdcParams) -> Self {
        let bits = params.avg_size.trailing_zeros();
        Self {
            params,
            pending: Vec::with_capacity(params.max_size),
            hash: 0,
            mask_s: (1u64 << (bits + 1)) - 1,
            mask_l: (1u64 << (bits - 1)) - 1,
        }
    }

    /// Feed one byte. Returns `Some(chunk)` if a boundary closes
    /// here; the byte is either *included* in the returned chunk
    /// (forced max-size cut) or *deferred* into the next chunk
    /// (content-defined cut).
    pub fn feed(&mut self, byte: u8) -> Option<Vec<u8>> {
        // `pos` is the index this byte would occupy in the current
        // chunk. Matches `i` in `next_boundary`.
        let pos = self.pending.len();

        // Boundary check window: positions [min_size, max_size).
        // Mirrors the loop bounds in `next_boundary`.
        if pos >= self.params.min_size && pos < self.params.max_size {
            self.hash = self.hash.wrapping_shl(1).wrapping_add(GEAR[byte as usize]);
            let mask = if pos < self.params.avg_size {
                self.mask_s
            } else {
                self.mask_l
            };
            if self.hash & mask == 0 {
                // Content-defined cut: byte starts the next chunk.
                let chunk = self.take_pending();
                self.pending.push(byte);
                return Some(chunk);
            }
        }

        self.pending.push(byte);

        // Forced cut at max_size: byte is included in this chunk.
        // (`next_boundary` returns `cap = max_size` when the loop
        // exits without a content cut, so the chunk is `data[..cap]`
        // — i.e. exactly `max_size` bytes including position
        // `max_size - 1`.)
        if self.pending.len() == self.params.max_size {
            return Some(self.take_pending());
        }
        None
    }

    /// Emit the trailing partial chunk if any. After this returns
    /// `None`, the stream is fully drained.
    pub fn flush(&mut self) -> Option<Vec<u8>> {
        if self.pending.is_empty() {
            None
        } else {
            Some(self.take_pending())
        }
    }

    fn take_pending(&mut self) -> Vec<u8> {
        self.hash = 0;
        std::mem::replace(&mut self.pending, Vec::with_capacity(self.params.max_size))
    }
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

    /// Stream the whole input through `FastCdcStream`, concatenate the
    /// emitted chunks, and assert the boundaries match what
    /// `chunk_ranges` would have produced. The streaming path is
    /// behavioural equivalent — same GEAR rolling hash, same masks,
    /// same min/max-size handling — just byte-at-a-time.
    fn stream_chunks(data: &[u8], params: &FastCdcParams) -> Vec<Vec<u8>> {
        let mut s = FastCdcStream::new(*params);
        let mut out = Vec::new();
        for &b in data {
            if let Some(c) = s.feed(b) {
                out.push(c);
            }
        }
        if let Some(c) = s.flush() {
            out.push(c);
        }
        out
    }

    fn slice_chunks(data: &[u8], params: &FastCdcParams) -> Vec<Vec<u8>> {
        chunk_ranges(data, params)
            .into_iter()
            .map(|r| data[r].to_vec())
            .collect()
    }

    #[test]
    fn streaming_matches_slice_on_zeros() {
        let data = vec![0u8; 64 * 1024];
        let params = FastCdcParams::default();
        assert_eq!(stream_chunks(&data, &params), slice_chunks(&data, &params));
    }

    #[test]
    fn streaming_matches_slice_on_ascending() {
        let data: Vec<u8> = (0..64 * 1024).map(|i| (i % 251) as u8).collect();
        let params = FastCdcParams::default();
        assert_eq!(stream_chunks(&data, &params), slice_chunks(&data, &params));
    }

    #[test]
    fn streaming_matches_slice_short_inputs() {
        let params = FastCdcParams::default();
        for len in [
            0_usize, 1, 100, 1023, 1024, 1025, 4095, 4096, 4097, 16383, 16384, 16385,
        ] {
            let data: Vec<u8> = (0..len).map(|i| (i * 13 + 7) as u8).collect();
            assert_eq!(
                stream_chunks(&data, &params),
                slice_chunks(&data, &params),
                "mismatch at len={len}"
            );
        }
    }

    #[test]
    fn streaming_matches_slice_small_params() {
        let params = FastCdcParams {
            min_size: 32,
            avg_size: 64,
            max_size: 128,
        };
        params.validate().unwrap();
        let data: Vec<u8> = (0..2048).map(|i| ((i * 31) ^ (i >> 3)) as u8).collect();
        assert_eq!(stream_chunks(&data, &params), slice_chunks(&data, &params));
    }
}
