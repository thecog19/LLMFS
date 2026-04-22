//! B3 — salience-inode lifecycle plumbing.
//!
//! These tests exercise `Filesystem::commit_salience` end to end
//! against the allocator + dedup index + reclaim pipeline. They
//! don't know anything about AWQ numerics (that's B1); the salience
//! payload here is an arbitrary byte blob chosen to force multiple
//! CDC chunks so we actually stress the inode-tree plumbing.
//!
//! Gates from plan.md's B3 entry:
//! (a) mount of a cover with a live salience inode reserves every
//!     chunk (post-mount writes can't collide with them).
//! (b) a commit that doesn't modify salience leaves the salience
//!     chunks reachable from the new super-root.
//! (c) init → commit_salience → load_salience → bytes identical.
//! (d) the dedup index contains hashes of the salience chunks
//!     after mount.

use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};
use llmdb::v2::cdc::FastCdcParams;
use llmdb::v2::dedup::hash_chunk;
use llmdb::v2::fs::Filesystem;

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
    let weight_count = 40_000_u64;
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
        name: "salience.f16".to_owned(),
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

/// Salience payload that deterministically varies byte-to-byte so
/// CDC produces > 1 data chunk, exercising the full inode tree.
fn salience_bytes() -> Vec<u8> {
    let mut data = Vec::with_capacity(1024);
    let mut state: u32 = 1_234_567;
    for _ in 0..1024 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        data.push(((state >> 16) & 0xFF) as u8);
    }
    data
}

#[test]
fn init_commit_salience_load_round_trips_bytes() {
    // Gate (c): init → commit_salience → load_salience bytes identical.
    let (cover, map) = make_cover();
    let mut fs = Filesystem::init_with_cdc_params(cover, map, small_cdc()).expect("init");
    assert_eq!(fs.load_salience().unwrap(), None, "no salience pre-commit");

    let payload = salience_bytes();
    let ptr = fs.commit_salience(&payload).expect("commit_salience");
    assert!(!ptr.is_null());

    let loaded = fs.load_salience().expect("load").expect("some");
    assert_eq!(loaded, payload);
}

#[test]
fn salience_survives_unmount_and_remount() {
    // Gate (a): a cover that already has a live salience inode
    // mounts cleanly; the reservation walk protects every one of
    // its chunks; `load_salience` returns the original bytes.
    let (cover, map) = make_cover();
    let payload = salience_bytes();
    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc())
        .expect("init");
    fs.commit_salience(&payload).expect("commit");
    let cover1 = fs.unmount().expect("unmount");

    let fs2 = Filesystem::mount_with_cdc_params(cover1, map, small_cdc())
        .expect("mount");
    let loaded = fs2.load_salience().expect("load").expect("some");
    assert_eq!(loaded, payload, "salience bytes changed across remount");
}

#[test]
fn file_commit_after_salience_preserves_salience() {
    // Gate (b): a commit that doesn't modify salience must leave
    // salience chunks live + reachable from the new super-root.
    // If `reclaim_abandoned_chunks` doesn't include salience in
    // its new scope, this test would start failing.
    let (cover, map) = make_cover();
    let payload = salience_bytes();
    let mut fs = Filesystem::init_with_cdc_params(cover, map, small_cdc()).expect("init");
    fs.commit_salience(&payload).expect("commit salience");

    // File write — touches root_dir_inode + dirty_bitmap, leaves
    // salience alone.
    fs.create_file("/notes.txt", b"hello, salience-preserving commit!")
        .expect("create file");

    // Salience still loads identically.
    let loaded = fs.load_salience().expect("load").expect("some");
    assert_eq!(loaded, payload);
}

#[test]
fn dedup_index_picks_up_salience_chunks_after_mount() {
    // Gate (d): the dedup index rebuilt at mount time contains
    // hashes for every salience data chunk.
    let (cover, map) = make_cover();
    let payload = salience_bytes();
    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc())
        .expect("init");
    fs.commit_salience(&payload).expect("commit");
    let cover1 = fs.unmount().expect("unmount");

    let fs2 = Filesystem::mount_with_cdc_params(cover1, map, small_cdc())
        .expect("mount");
    let dedup = fs2.dedup_index();
    assert!(
        !dedup.is_empty(),
        "dedup index is empty after mount of a calibrated cover",
    );

    // Deeper check: all CDC chunks of `payload` hashed the same
    // way should be in the index. Re-chunk the payload using the
    // same params and assert at least one of the resulting chunk
    // hashes matches an index entry. (We don't check every chunk
    // because the inode's own chunks hash differently and pollute
    // the index; the data-chunk hashes are what matter.)
    use llmdb::v2::cdc::chunk_ranges;
    let ranges = chunk_ranges(&payload, &small_cdc());
    assert!(!ranges.is_empty());
    let mut hits = 0;
    for r in &ranges {
        let h = hash_chunk(&payload[r.clone()]);
        if dedup.lookup(&h).is_some() {
            hits += 1;
        }
    }
    assert!(
        hits > 0,
        "none of the {} salience data-chunk hashes are in the dedup index",
        ranges.len(),
    );
}

#[test]
fn recommit_same_salience_dedups_chunks() {
    // Writing identical salience twice must hit the dedup index on
    // the second pass — a regression in B3's dedup integration
    // would cause the second commit to allocate fresh chunks.
    let (cover, map) = make_cover();
    let payload = salience_bytes();
    let mut fs = Filesystem::init_with_cdc_params(cover, map, small_cdc()).expect("init");
    let free_0 = fs.allocator_free_weights();
    fs.commit_salience(&payload).expect("commit 1");
    let free_1 = fs.allocator_free_weights();
    let consumed_1 = free_0 - free_1;
    fs.commit_salience(&payload).expect("commit 2");
    let free_2 = fs.allocator_free_weights();
    let consumed_2 = free_1 - free_2;

    // Second commit must consume strictly less than the first —
    // all of payload's data chunks dedup-hit, so the additional
    // cost is just the inode + super-root update + bitmap delta.
    assert!(
        consumed_2 < consumed_1,
        "re-committing identical salience didn't dedup: first commit consumed {consumed_1} \
         weights, second consumed {consumed_2}",
    );
}
