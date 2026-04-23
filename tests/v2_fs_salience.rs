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
use llmdb::v2::salience::{PeriodicSlotSalience, SalienceTable};

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

/// A salience table large enough that its encoding produces more
/// than one CDC chunk, so we actually exercise the full inode tree
/// (indirect blocks + multiple data chunks).
fn large_salience_table() -> SalienceTable {
    // 256 values per slot × 4 bytes = 1 KB of f32s per populated
    // slot. Combined with the 8-byte slot header + 12-byte file
    // header, that's ~2 KB — plenty for small_cdc()'s 64-byte
    // average chunk size to produce multiple chunks.
    let mut per_slot = Vec::new();
    let mut state: u32 = 1_234_567;
    for slot_idx in 0..4 {
        // Slots 0, 1, 2, 3: two populated, two None.
        if slot_idx % 2 == 0 {
            let mut values = Vec::with_capacity(256);
            for _ in 0..256 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let v = ((state >> 16) & 0xFF) as f32 * 0.001; // positive f32
                values.push(v);
            }
            per_slot.push(Some(PeriodicSlotSalience::new(256, values).unwrap()));
        } else {
            per_slot.push(None);
        }
    }
    SalienceTable::new(per_slot)
}

#[test]
fn init_commit_salience_load_round_trips_table() {
    // Gate (c): init → commit_salience → load_salience bytes identical.
    let (cover, map) = make_cover();
    let mut fs = Filesystem::init_with_cdc_params(cover, map, small_cdc()).expect("init");
    assert!(
        fs.load_salience().unwrap().is_none(),
        "no salience pre-commit"
    );

    let table = large_salience_table();
    let ptr = fs.commit_salience(&table).expect("commit_salience");
    assert!(!ptr.is_null());

    let loaded = fs.load_salience().expect("load").expect("some");
    assert_eq!(loaded, table);
}

#[test]
fn salience_survives_unmount_and_remount() {
    // Gate (a): a cover that already has a live salience inode
    // mounts cleanly; the reservation walk protects every one of
    // its chunks; `load_salience` returns the original table.
    let (cover, map) = make_cover();
    let table = large_salience_table();
    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");
    fs.commit_salience(&table).expect("commit");
    let cover1 = fs.unmount().expect("unmount");

    let fs2 = Filesystem::mount_with_cdc_params(cover1, map, small_cdc()).expect("mount");
    let loaded = fs2.load_salience().expect("load").expect("some");
    assert_eq!(loaded, table, "salience table changed across remount");
}

#[test]
fn file_commit_after_salience_preserves_salience() {
    // Gate (b): a commit that doesn't modify salience must leave
    // salience chunks live + reachable from the new super-root.
    // If `reclaim_abandoned_chunks` doesn't include salience in
    // its new scope, this test would start failing.
    let (cover, map) = make_cover();
    let table = large_salience_table();
    let mut fs = Filesystem::init_with_cdc_params(cover, map, small_cdc()).expect("init");
    fs.commit_salience(&table).expect("commit salience");

    // File write — touches root_dir_inode + dirty_bitmap, leaves
    // salience alone.
    fs.create_file("/notes.txt", b"hello, salience-preserving commit!")
        .expect("create file");

    // Salience still loads identically.
    let loaded = fs.load_salience().expect("load").expect("some");
    assert_eq!(loaded, table);
}

#[test]
fn dedup_index_picks_up_salience_chunks_after_mount() {
    // Gate (d): the dedup index rebuilt at mount time contains
    // hashes for every salience data chunk.
    let (cover, map) = make_cover();
    let table = large_salience_table();
    let encoded = table.encode();
    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");
    fs.commit_salience(&table).expect("commit");
    let cover1 = fs.unmount().expect("unmount");

    let fs2 = Filesystem::mount_with_cdc_params(cover1, map, small_cdc()).expect("mount");
    let dedup = fs2.dedup_index();
    assert!(
        !dedup.is_empty(),
        "dedup index is empty after mount of a calibrated cover",
    );

    // Deeper check: at least one of the encoded payload's CDC-chunk
    // hashes appears in the rebuilt index.
    use llmdb::v2::cdc::chunk_ranges;
    let ranges = chunk_ranges(&encoded, &small_cdc());
    assert!(!ranges.is_empty());
    let mut hits = 0;
    for r in &ranges {
        let h = hash_chunk(&encoded[r.clone()]);
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

/// Cover with uniform ceiling across the slot — every weight has
/// the same |x|, so the ceiling tiebreaker never decides placement
/// and the salience secondary term becomes observable.
fn make_uniform_ceiling_cover() -> (Vec<u8>, TensorMap) {
    let weight_count = 2_000_000_u64;
    let mut bytes = Vec::with_capacity((weight_count * 2) as usize);
    let fixed = f32_to_f16_bits(0.01).to_le_bytes();
    for _ in 0..weight_count {
        bytes.extend_from_slice(&fixed);
    }
    let slot = TensorSlot {
        name: "uniform.f16".to_owned(),
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

#[test]
fn calibrated_cover_avoids_high_salience_region_for_new_writes() {
    // B4 gate: writes on a calibrated cover avoid the high-salience
    // band that the salience table points to. Mechanism: the
    // compound FitKey `(max_ceiling, max_salience, …)`. Using a
    // uniform-ceiling cover guarantees ceiling ties so salience
    // becomes the deciding factor.
    let (cover, map) = make_uniform_ceiling_cover();

    // Salience: a narrow high-magnitude band in the middle of the
    // slot. Everything else is either zero (in the populated range
    // before/after the band) or implicitly zero (past the populated
    // range, which is every weight index ≥ values.len()).
    let populated_len = 1024_usize;
    let mut values = vec![0.0_f32; populated_len];
    let high_lo = 400;
    let high_hi = 600;
    for v in values.iter_mut().take(high_hi).skip(high_lo) {
        *v = 1.0;
    }
    let table = SalienceTable::new(vec![Some(
        PeriodicSlotSalience::new(populated_len as u64, values).unwrap(),
    )]);

    let mut fs = Filesystem::init(cover, map.clone()).expect("init");
    fs.commit_salience(&table).expect("calibrate");
    let cover1 = fs.unmount().expect("unmount");
    let mut fs = Filesystem::mount(cover1, map).expect("mount");

    let data = b"placement-bias test write";
    fs.create_file("/f.txt", data).expect("write");

    let inode = fs.inode_at("/f.txt").expect("inode");
    let first = inode
        .direct
        .iter()
        .find(|p| !p.is_null())
        .expect("direct ptr");
    // F16's `stealable_bits_per_weight` is 4 (see
    // `GgufQuantType::stealable_bits_hint`) — one chunk of N bits
    // spans ceil(N / 4) weights.
    let bits_per_weight = 4_u32;
    let weights_per_chunk = first.length_in_bits.div_ceil(bits_per_weight);
    let start = first.start_weight;
    let end = start + weights_per_chunk;

    // Assert the chunk does not overlap the high-salience band.
    let overlaps = start < high_hi as u32 && end > high_lo as u32;
    assert!(
        !overlaps,
        "calibrated placement overlaps high-salience band [{high_lo}, {high_hi}): \
         chunk at [{start}, {end})",
    );
}

#[test]
fn recommit_same_salience_dedups_chunks() {
    // Writing identical salience twice must hit the dedup index on
    // the second pass — a regression in B3's dedup integration
    // would cause the second commit to allocate fresh chunks.
    let (cover, map) = make_cover();
    let table = large_salience_table();
    let mut fs = Filesystem::init_with_cdc_params(cover, map, small_cdc()).expect("init");
    let free_0 = fs.allocator_free_weights();
    fs.commit_salience(&table).expect("commit 1");
    let free_1 = fs.allocator_free_weights();
    let consumed_1 = free_0 - free_1;
    fs.commit_salience(&table).expect("commit 2");
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
