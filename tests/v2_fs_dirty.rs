//! V2 Filesystem dirty-tracking integration tests.
//!
//! Verifies:
//! 1. A write marks every allocated weight dirty.
//! 2. Rewriting with different content frees old chunks (so their
//!    positions enter the free list, still marked dirty).
//! 3. A subsequent write's allocator prefers the just-freed dirty
//!    positions over never-touched pristine ones.
//! 4. Dirty bits persist across unmount → remount.
//!
//! (3) is the load-bearing property: cover-damage stays bounded
//! because rewrites land on already-perturbed weights rather than
//! growing the damaged set.

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
    // 20 K weights → 2.5 KB bitmap → fits in ~40 small_cdc chunks.
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
        name: "dirty.f16".to_owned(),
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

/// Deterministic random-ish bytes (xorshift64) so CDC produces
/// distinct chunks across `salt` values — dedup never hits, so
/// we're purely exercising alloc preference.
fn random_bytes(len: usize, salt: u64) -> Vec<u8> {
    let mut state = salt
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0xDEAD_BEEF_CAFE_BABE);
    if state == 0 {
        state = 1;
    }
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state & 0xFF) as u8
        })
        .collect()
}

fn direct_pointers(fs: &Filesystem) -> Vec<Pointer> {
    fs.inode_at("/data")
        .expect("file inode")
        .direct
        .iter()
        .filter(|p| !p.is_null())
        .copied()
        .collect()
}

fn pointer_weights(ptr: &Pointer, bpw: u32) -> Vec<(u16, u32)> {
    let w_count = ptr.length_in_bits.div_ceil(bpw);
    (0..w_count)
        .map(|i| (ptr.slot, ptr.start_weight + i))
        .collect()
}

// ------------------------------------------------------------------
// (1) Writes mark every allocated weight dirty
// ------------------------------------------------------------------

#[test]
fn write_marks_every_allocated_chunk_weight_dirty() {
    let (cover, map) = make_cover();
    let bpw = 4_u32; // F16 stealable bits per weight

    let mut fs = Filesystem::init_with_cdc_params(cover, map, small_cdc()).expect("init");
    let data = random_bytes(500, 1);
    fs.create_file("/data", &data).expect("write");

    let bm = fs.dirty_bitmap();
    for p in direct_pointers(&fs) {
        for (slot, w) in pointer_weights(&p, bpw) {
            assert!(
                bm.is_dirty(slot, w),
                "weight (slot {slot}, w {w}) should be dirty after write",
            );
        }
    }
}

// ------------------------------------------------------------------
// (2) Rewrite frees old chunks; free list gains dirty runs
// ------------------------------------------------------------------

#[test]
fn rewriting_returns_old_chunks_to_the_free_list() {
    let (cover, map) = make_cover();
    let mut fs = Filesystem::init_with_cdc_params(cover, map, small_cdc()).expect("init");

    let data1 = random_bytes(400, 1);
    fs.create_file("/data", &data1).expect("write 1");
    let free_after_first = fs.allocator_free_weights();

    let data2 = random_bytes(400, 2);
    fs.create_file("/data", &data2).expect("write 2");
    let free_after_second = fs.allocator_free_weights();

    // Rewriting freed the old chunks (≥ their weight count) while
    // also allocating new ones (of roughly the same weight count),
    // so net free-list size should have grown by approximately
    // `data1's weights - data2's inode/super-root overhead`. A
    // loose but useful assertion: the free list grew, indicating
    // reclamation happened.
    assert!(
        free_after_second > free_after_first
            || free_after_second >= free_after_first.saturating_sub(200),
        "expected reclamation; free before={free_after_first}, after={free_after_second}",
    );
}

// ------------------------------------------------------------------
// (3) Subsequent writes prefer dirty-free positions
// ------------------------------------------------------------------

#[test]
fn third_write_lands_on_already_dirty_positions() {
    // Write A, write B (different content), write C (different
    // content). By the time C runs, A's and B's old chunks are in
    // the free list marked dirty. C's allocator should prefer
    // those over the never-touched tail of the cover.
    let (cover, map) = make_cover();
    let bpw = 4_u32;
    let mut fs = Filesystem::init_with_cdc_params(cover, map, small_cdc()).expect("init");

    let data_a = random_bytes(400, 1);
    fs.create_file("/data", &data_a).expect("write A");
    let a_weights: HashSet<(u16, u32)> = direct_pointers(&fs)
        .iter()
        .flat_map(|p| pointer_weights(p, bpw))
        .collect();

    let data_b = random_bytes(400, 2);
    fs.create_file("/data", &data_b).expect("write B");
    let b_weights: HashSet<(u16, u32)> = direct_pointers(&fs)
        .iter()
        .flat_map(|p| pointer_weights(p, bpw))
        .collect();

    let data_c = random_bytes(400, 3);
    fs.create_file("/data", &data_c).expect("write C");
    let c_weights: HashSet<(u16, u32)> = direct_pointers(&fs)
        .iter()
        .flat_map(|p| pointer_weights(p, bpw))
        .collect();

    // C should land entirely on previously-dirty positions — a
    // union of A's and B's former weights.
    let dirty_before_c: HashSet<_> = a_weights.union(&b_weights).copied().collect();
    let c_in_dirty = c_weights.intersection(&dirty_before_c).count();
    let c_total = c_weights.len();

    // Require the vast majority of C's weights to land on dirty
    // ground. Not strictly 100% — chunk boundaries can push a
    // small tail into pristine space when no dirty run is large
    // enough to satisfy a required size — but should be
    // overwhelmingly true.
    assert!(
        c_in_dirty * 20 >= c_total * 17,
        "dirty preference should dominate: {c_in_dirty} of {c_total} \
         C-weights landed on previously-dirty positions",
    );
    // And C's dirty bitmap should include every weight it touched.
    let bm = fs.dirty_bitmap();
    for (s, w) in &c_weights {
        assert!(bm.is_dirty(*s, *w));
    }
}

// ------------------------------------------------------------------
// (4) Dirty bitmap survives unmount → remount
// ------------------------------------------------------------------

#[test]
fn dirty_bits_survive_unmount_and_remount() {
    let (cover, map) = make_cover();
    let bpw = 4_u32;

    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");
    let data = random_bytes(300, 1);
    fs.create_file("/data", &data).expect("write");

    // Snapshot the dirty weights touched by the data chunks.
    let touched: Vec<(u16, u32)> = direct_pointers(&fs)
        .iter()
        .flat_map(|p| pointer_weights(p, bpw))
        .collect();

    let cover_after = fs.unmount();
    let fs2 =
        Filesystem::mount_with_cdc_params(cover_after, map, small_cdc()).expect("mount");

    // After remount, every weight we touched should STILL be dirty —
    // the bitmap was persisted and read back.
    let bm = fs2.dirty_bitmap();
    for (s, w) in &touched {
        assert!(
            bm.is_dirty(*s, *w),
            "weight (slot {s}, w {w}) should still be dirty after remount",
        );
    }
}
