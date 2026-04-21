//! FastCDC chunker — unit tests for `src/v2/cdc.rs`.
//!
//! Properties under test:
//! 1. Empty input → zero chunks.
//! 2. Input ≤ min_size → one chunk covering the whole input.
//! 3. Determinism — same bytes → same boundary set.
//! 4. Total coverage — chunk ranges concatenate back to `0..data.len()`.
//! 5. Size bounds — interior chunks are ≥ min_size and ≤ max_size.
//! 6. Insertion stability — inserting a byte near the end of a long
//!    input leaves boundaries before the insertion point unchanged
//!    (the core FastCDC win over fixed-size chunking).
//! 7. Parameter validation — bad params reject with a typed error.

use llmdb::v2::cdc::{FastCdcError, FastCdcParams, chunk_ranges};

/// Deterministic pseudo-random bytes so tests are reproducible
/// without a rand crate dependency.
struct Prng(u64);
impl Prng {
    fn new(seed: u64) -> Self {
        let s = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0x1234_5678_DEAD_BEEF);
        Self(if s == 0 { 1 } else { s })
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn pseudorandom(n: usize, seed: u64) -> Vec<u8> {
    let mut p = Prng::new(seed);
    (0..n).map(|_| (p.next() & 0xFF) as u8).collect()
}

fn small_params() -> FastCdcParams {
    // All powers of two with min < avg < max, avg ≥ 4.
    FastCdcParams {
        min_size: 32,
        avg_size: 64,
        max_size: 256,
    }
}

fn medium_params() -> FastCdcParams {
    FastCdcParams {
        min_size: 128,
        avg_size: 512,
        max_size: 2048,
    }
}

// ------------------------------------------------------------------
// Basic shapes
// ------------------------------------------------------------------

#[test]
fn empty_input_yields_no_chunks() {
    let ranges = chunk_ranges(&[], &small_params());
    assert!(ranges.is_empty());
}

#[test]
fn input_smaller_than_min_yields_single_chunk() {
    let data = vec![0x42_u8; 10];
    let ranges = chunk_ranges(&data, &small_params());
    assert_eq!(ranges, vec![0..10]);
}

#[test]
fn input_exactly_min_size_yields_single_chunk() {
    let data = vec![0x42_u8; 32];
    let ranges = chunk_ranges(&data, &small_params());
    assert_eq!(ranges, vec![0..32]);
}

#[test]
fn chunks_concatenate_to_full_input() {
    let data = pseudorandom(10_000, 0xA5A5);
    let ranges = chunk_ranges(&data, &medium_params());
    assert!(!ranges.is_empty());
    let mut cursor = 0;
    for r in &ranges {
        assert_eq!(r.start, cursor);
        cursor = r.end;
    }
    assert_eq!(cursor, data.len());
}

// ------------------------------------------------------------------
// Determinism
// ------------------------------------------------------------------

#[test]
fn determinism_same_input_same_ranges() {
    let data = pseudorandom(8192, 42);
    let p = medium_params();
    let a = chunk_ranges(&data, &p);
    let b = chunk_ranges(&data, &p);
    let c = chunk_ranges(&data, &p);
    assert_eq!(a, b);
    assert_eq!(b, c);
}

// ------------------------------------------------------------------
// Size bounds
// ------------------------------------------------------------------

#[test]
fn interior_chunks_respect_min_and_max_bounds() {
    let data = pseudorandom(100_000, 7);
    let p = medium_params();
    let ranges = chunk_ranges(&data, &p);
    assert!(ranges.len() >= 2, "expected multi-chunk for 100 KB input");

    // Every chunk except the final one must satisfy
    // min_size ≤ len ≤ max_size. The last chunk can be short if the
    // tail of the input doesn't reach min_size — the stream ended.
    for r in &ranges[..ranges.len() - 1] {
        let len = r.end - r.start;
        assert!(
            len >= p.min_size,
            "chunk {:?} shorter than min_size {}",
            r,
            p.min_size
        );
        assert!(
            len <= p.max_size,
            "chunk {:?} longer than max_size {}",
            r,
            p.max_size
        );
    }

    // Last chunk's upper bound is also max_size (it might be shorter).
    let last = ranges.last().unwrap();
    assert!(last.end - last.start <= p.max_size);
}

#[test]
fn runs_of_zeros_still_cut_by_min_max() {
    // Zeros don't mutate the rolling hash much; verify we still
    // get max-size cuts rather than one giant chunk.
    let data = vec![0u8; 20_000];
    let p = small_params();
    let ranges = chunk_ranges(&data, &p);
    assert!(ranges.len() >= 2);
    for r in &ranges[..ranges.len() - 1] {
        let len = r.end - r.start;
        assert!(len >= p.min_size);
        assert!(len <= p.max_size);
    }
}

// ------------------------------------------------------------------
// Insertion stability (the core FastCDC win)
// ------------------------------------------------------------------

#[test]
fn byte_insertion_leaves_prefix_boundaries_identical() {
    let data = pseudorandom(20_000, 99);
    let p = medium_params();
    let original = chunk_ranges(&data, &p);

    // Insert a byte at offset ~15_000 (near the end of the stream).
    let insertion_point = 15_000;
    let mut shifted = data.clone();
    shifted.insert(insertion_point, 0xAB);
    let shifted_ranges = chunk_ranges(&shifted, &p);

    // Every range that ends strictly before the insertion point must
    // appear unchanged in the shifted stream. The boundaries at or
    // after the insertion may shift — content-defined chunking is
    // what we're testing here.
    let unchanged = original
        .iter()
        .take_while(|r| r.end <= insertion_point)
        .count();
    assert!(
        unchanged > 0,
        "expected at least one boundary before {insertion_point}"
    );
    for i in 0..unchanged {
        assert_eq!(
            original[i], shifted_ranges[i],
            "boundary {i} shifted under insertion (CDC not insert-stable)",
        );
    }
}

// ------------------------------------------------------------------
// Parameter validation
// ------------------------------------------------------------------

#[test]
fn validate_rejects_non_power_of_two_avg() {
    let p = FastCdcParams {
        min_size: 64,
        avg_size: 100, // not a power of 2
        max_size: 512,
    };
    match p.validate() {
        Err(FastCdcError::AvgNotPowerOfTwo(100)) => {}
        other => panic!("expected AvgNotPowerOfTwo, got {other:?}"),
    }
}

#[test]
fn validate_rejects_min_greater_than_avg() {
    let p = FastCdcParams {
        min_size: 128,
        avg_size: 64, // avg < min
        max_size: 512,
    };
    assert!(p.validate().is_err());
}

#[test]
fn validate_rejects_max_smaller_than_avg() {
    let p = FastCdcParams {
        min_size: 16,
        avg_size: 64,
        max_size: 32, // max < avg
    };
    assert!(p.validate().is_err());
}

#[test]
fn validate_accepts_sensible_params() {
    let p = FastCdcParams {
        min_size: 32,
        avg_size: 64,
        max_size: 256,
    };
    p.validate().expect("sensible params should validate");
}
