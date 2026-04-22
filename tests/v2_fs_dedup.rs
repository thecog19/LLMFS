//! V2 Filesystem-level dedup — integration tests that exercise the
//! hash index through the public write path.
//!
//! What these verify, via the visible public surface:
//! 1. **Within-write dedup.** An all-zeros payload collapses into
//!    one (or very few) unique data chunks — every interior chunk
//!    has identical content, so the file inode holds the same pointer
//!    multiple times.
//! 2. **Rewrite dedup.** Writing the same data twice in a session
//!    reuses the pre-rewrite chunks — all pointers in the new inode
//!    appeared in the old inode.
//! 3. **Dedup survives unmount.** A remount rebuilds the dedup index
//!    by walking every file in the tree; a subsequent write of
//!    identical data still dedup-hits (pointers unchanged).
//! 4. **Unique content allocates uniquely.** Writing distinct
//!    chunks' worth of data produces distinct pointers — the index
//!    doesn't collapse non-duplicates.

use std::collections::HashSet;

use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};
use llmdb::v2::cdc::FastCdcParams;
use llmdb::v2::fs::Filesystem;
use llmdb::v2::pointer::Pointer;

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp32 = ((bits >> 23) & 0xFF) as i32;
    let mantissa32 = bits & 0x7FFFFF;
    if exp32 == 0 {
        return sign << 15;
    }
    let exp16 = exp32 - 127 + 15;
    if exp16 <= 0 {
        return sign << 15;
    }
    if exp16 >= 31 {
        return (sign << 15) | (0x1F << 10);
    }
    let mantissa16 = (mantissa32 >> 13) as u16;
    (sign << 15) | ((exp16 as u16) << 10) | mantissa16
}

fn make_cover() -> (Vec<u8>, TensorMap) {
    let weight_count = 20_000_u64;
    let values: Vec<f32> = (0..weight_count)
        .map(|i| {
            let sign = if i % 3 == 0 { -1.0 } else { 1.0 };
            sign * ((i + 1) as f32) * 0.00002
        })
        .collect();
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for v in &values {
        bytes.extend_from_slice(&f32_to_f16_bits(*v).to_le_bytes());
    }
    let slot = TensorSlot {
        name: "dedup.f16".to_owned(),
        quant_type: GgufQuantType::F16,
        tier: TensorTier::Tier1,
        data_offset: 0,
        weight_count,
        stealable_bits_per_weight: GgufQuantType::F16.stealable_bits_hint(),
        capacity_bits: weight_count * GgufQuantType::F16.stealable_bits_hint() as u64,
        bit_start: 0,
        bit_end: weight_count * GgufQuantType::F16.stealable_bits_hint() as u64,
    };
    let map = TensorMap {
        slots: vec![slot.clone()],
        total_capacity_bits: slot.capacity_bits,
        total_capacity_bytes: slot.capacity_bits / 8,
    };
    (bytes, map)
}

fn small_cdc() -> FastCdcParams {
    FastCdcParams {
        min_size: 32,
        avg_size: 64,
        max_size: 128,
    }
}

/// Collect every non-null direct-pointer from the `/data` file's
/// inode — the test's single user-visible file.
fn direct_pointers(fs: &Filesystem) -> Vec<Pointer> {
    fs.inode_at("/data")
        .expect("file inode")
        .direct
        .iter()
        .filter(|p| !p.is_null())
        .copied()
        .collect()
}

// ------------------------------------------------------------------
// Within-write dedup
// ------------------------------------------------------------------

#[test]
fn all_zeros_file_collapses_to_a_single_data_chunk() {
    // Interior CDC chunks over an all-zeros stream are byte-
    // identical (the rolling hash evolves the same per position),
    // so dedup should collapse every chunk to a single pointer.
    let (cover, map) = make_cover();
    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");

    // 1 KB zeros → maybe 8–32 chunks at small_cdc params, all but
    // possibly the first / last identical.
    let data = vec![0u8; 1024];
    fs.create_file("/data", &data).expect("write zeros");

    let ptrs = direct_pointers(&fs);
    assert!(!ptrs.is_empty(), "expected at least one chunk");

    // At least one pointer should appear twice — dedup hit among
    // identical interior chunks.
    let unique: HashSet<(u16, u32)> =
        ptrs.iter().map(|p| (p.slot, p.start_weight)).collect();
    assert!(
        unique.len() < ptrs.len(),
        "all-zeros file should dedup some chunks: {} pointers → {} unique",
        ptrs.len(),
        unique.len(),
    );

    // The dedup index tracks every live file chunk — its size should
    // be at least `unique.len()` (it may also include the file's
    // indirect-tier pointers, but the direct-only file we wrote has
    // no indirects).
    assert!(fs.dedup_index().len() >= unique.len());

    // Round-trip the bytes.
    assert_eq!(fs.read_file("/data").expect("read"), data);
}

// ------------------------------------------------------------------
// Rewrite within a session
// ------------------------------------------------------------------

#[test]
fn rewriting_identical_content_reuses_existing_pointers() {
    let (cover, map) = make_cover();
    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");

    let data = b"dedup in action, same content, same pointers";
    fs.create_file("/data", data).expect("first write");
    let first: HashSet<(u16, u32)> = direct_pointers(&fs)
        .iter()
        .map(|p| (p.slot, p.start_weight))
        .collect();

    fs.create_file("/data", data)
        .expect("second write of same content");
    let second: HashSet<(u16, u32)> = direct_pointers(&fs)
        .iter()
        .map(|p| (p.slot, p.start_weight))
        .collect();

    // Every pointer in the new inode must exist in the old inode —
    // dedup should have reused every single data chunk.
    for p in &second {
        assert!(
            first.contains(p),
            "new inode has pointer {p:?} that wasn't in the first write",
        );
    }
    // And the sets match one-for-one (no new chunks allocated).
    assert_eq!(first, second);
}

// ------------------------------------------------------------------
// Dedup survives unmount + remount
// ------------------------------------------------------------------

#[test]
fn dedup_survives_remount() {
    let (cover, map) = make_cover();
    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");

    let data = b"persistent dedup: identical bytes across sessions";
    fs.create_file("/data", data).expect("write 1");
    let original_ptrs: HashSet<(u16, u32)> = direct_pointers(&fs)
        .iter()
        .map(|p| (p.slot, p.start_weight))
        .collect();
    let cover1 = fs.unmount().expect("unmount");

    let mut fs2 =
        Filesystem::mount_with_cdc_params(cover1, map, small_cdc()).expect("mount");
    // After mount the dedup index should have been rebuilt from
    // the current inode tree.
    assert!(fs2.dedup_index().len() >= original_ptrs.len());

    fs2.create_file("/data", data)
        .expect("second write of same content");
    let after_remount: HashSet<(u16, u32)> = direct_pointers(&fs2)
        .iter()
        .map(|p| (p.slot, p.start_weight))
        .collect();
    assert_eq!(original_ptrs, after_remount, "dedup should match pre-remount");
}

// ------------------------------------------------------------------
// Non-duplicates don't collapse
// ------------------------------------------------------------------

#[test]
fn distinct_content_produces_distinct_pointers() {
    // Pseudo-random bytes from xorshift64 — won't cycle within the
    // test's byte budget, so chunks should be byte-distinct and
    // dedup should not collapse them.
    let (cover, map) = make_cover();
    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");

    let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
    let data: Vec<u8> = (0..800_usize)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state & 0xFF) as u8
        })
        .collect();
    fs.create_file("/data", &data).expect("write");

    let ptrs = direct_pointers(&fs);
    let unique: HashSet<(u16, u32)> =
        ptrs.iter().map(|p| (p.slot, p.start_weight)).collect();
    // Every chunk has distinct content → pointers should be all
    // distinct. Equality (not just ≤) confirms no false dedup hits.
    assert_eq!(
        unique.len(),
        ptrs.len(),
        "random chunks shouldn't collide under dedup",
    );

    assert_eq!(fs.read_file("/data").expect("read"), data);
}
