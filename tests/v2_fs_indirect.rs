//! V2 Filesystem indirect-pointer round-trips (DESIGN-NEW §15.3
//! direct + single / double / triple indirect layout).
//!
//! With a test chunk size of 64 bytes → 4 pointers per full indirect
//! block (ppb = 4), the tiers cover:
//!
//! | tier              | chunk count           | data bytes        |
//! |-------------------|-----------------------|-------------------|
//! | direct            | 0..=12                | up to 768 B       |
//! | +single indirect  | 13..=16               | up to 1024 B      |
//! | +double indirect  | 17..=32               | up to 2048 B      |
//! | +triple indirect  | 33..=96               | up to 6144 B      |
//! | fails FileTooLarge| > 96                  | > 6144 B          |
//!
//! Tests exercise each tier and confirm the bytes survive
//! unmount → remount.

use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};
use llmdb::v2::fs::{Filesystem, FsError};

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

fn f16_slot(weight_count: u64, data_offset: u64) -> TensorSlot {
    let bits = GgufQuantType::F16.stealable_bits_hint() as u64;
    TensorSlot {
        name: "indirect.f16".to_owned(),
        quant_type: GgufQuantType::F16,
        tier: TensorTier::Tier1,
        data_offset,
        weight_count,
        stealable_bits_per_weight: bits as usize,
        capacity_bits: weight_count * bits,
        bit_start: 0,
        bit_end: weight_count * bits,
    }
}

/// Big cover for all the tests — enough F16 weights that even
/// triple-indirect won't run into OutOfSpace.
fn make_cover() -> (Vec<u8>, TensorMap) {
    let weight_count = 200_000_u64;
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
    let slot = f16_slot(weight_count, 0);
    let map = TensorMap {
        slots: vec![slot.clone()],
        total_capacity_bits: slot.capacity_bits,
        total_capacity_bytes: slot.capacity_bits / 8,
    };
    (bytes, map)
}

/// Deterministic-ish byte pattern of a given length.
fn pattern(len: usize, salt: u32) -> Vec<u8> {
    (0..len as u32)
        .map(|i| (i.wrapping_mul(salt).wrapping_add(17) & 0xFF) as u8)
        .collect()
}

// ------------------------------------------------------------------
// Tier-by-tier round trips
// ------------------------------------------------------------------

#[test]
fn single_indirect_tier_round_trips() {
    // 14 chunks × 64 B = 896 B: 12 direct + 2 in single indirect.
    let (cover, map) = make_cover();
    let data = pattern(14 * 64, 0xA5A5_A5A5);

    let mut fs = Filesystem::init_with_chunk_size(cover, map.clone(), 64).expect("init");
    fs.write(&data).expect("write");
    assert_eq!(fs.read().expect("read"), data);

    let cover_after = fs.unmount();
    let fs2 = Filesystem::mount_with_chunk_size(cover_after, map, 64).expect("mount");
    assert_eq!(fs2.read().expect("read after remount"), data);
    assert_eq!(fs2.file_length(), data.len() as u64);
}

#[test]
fn single_indirect_tier_full_round_trips() {
    // Exactly 16 chunks = direct + full single-indirect.
    let (cover, map) = make_cover();
    let data = pattern(16 * 64, 0x1234_5678);

    let mut fs = Filesystem::init_with_chunk_size(cover, map.clone(), 64).expect("init");
    fs.write(&data).expect("write");
    assert_eq!(fs.read().expect("read"), data);

    let cover_after = fs.unmount();
    let fs2 = Filesystem::mount_with_chunk_size(cover_after, map, 64).expect("mount");
    assert_eq!(fs2.read().expect("read after remount"), data);
}

#[test]
fn double_indirect_tier_round_trips() {
    // 20 chunks × 64 B = 1280 B: 12 direct + 4 single + 4 via double
    // (one single-indirect sub-block under double indirect).
    let (cover, map) = make_cover();
    let data = pattern(20 * 64, 0xDEAD_BEEF);

    let mut fs = Filesystem::init_with_chunk_size(cover, map.clone(), 64).expect("init");
    fs.write(&data).expect("write");
    assert_eq!(fs.read().expect("read"), data);

    let cover_after = fs.unmount();
    let fs2 = Filesystem::mount_with_chunk_size(cover_after, map, 64).expect("mount");
    assert_eq!(fs2.read().expect("read after remount"), data);
}

#[test]
fn double_indirect_tier_full_round_trips() {
    // 32 chunks = direct + full single + full double.
    let (cover, map) = make_cover();
    let data = pattern(32 * 64, 0xCAFE_BABE);

    let mut fs = Filesystem::init_with_chunk_size(cover, map.clone(), 64).expect("init");
    fs.write(&data).expect("write");
    assert_eq!(fs.read().expect("read"), data);

    let cover_after = fs.unmount();
    let fs2 = Filesystem::mount_with_chunk_size(cover_after, map, 64).expect("mount");
    assert_eq!(fs2.read().expect("read after remount"), data);
}

#[test]
fn triple_indirect_tier_round_trips() {
    // 50 chunks × 64 B = 3200 B: 12 direct + 4 single + 16 double + 18 via triple.
    let (cover, map) = make_cover();
    let data = pattern(50 * 64, 0xFEED_FACE);

    let mut fs = Filesystem::init_with_chunk_size(cover, map.clone(), 64).expect("init");
    fs.write(&data).expect("write");
    assert_eq!(fs.read().expect("read"), data);

    let cover_after = fs.unmount();
    let fs2 = Filesystem::mount_with_chunk_size(cover_after, map, 64).expect("mount");
    assert_eq!(fs2.read().expect("read after remount"), data);
    assert_eq!(fs2.file_length(), data.len() as u64);
}

#[test]
fn triple_indirect_tier_full_round_trips() {
    // Exactly 96 chunks = full direct + single + double + triple.
    let (cover, map) = make_cover();
    let data = pattern(96 * 64, 0x9999_9999);

    let mut fs = Filesystem::init_with_chunk_size(cover, map.clone(), 64).expect("init");
    fs.write(&data).expect("write");
    assert_eq!(fs.read().expect("read"), data);

    let cover_after = fs.unmount();
    let fs2 = Filesystem::mount_with_chunk_size(cover_after, map, 64).expect("mount");
    assert_eq!(fs2.read().expect("read after remount"), data);
}

// ------------------------------------------------------------------
// Overflow past triple
// ------------------------------------------------------------------

#[test]
fn file_beyond_triple_indirect_returns_file_too_large() {
    // 97 chunks exceeds the cap: direct(12) + single(4) + double(16) + triple(64) = 96.
    let (cover, map) = make_cover();
    let data = pattern(97 * 64, 0x0000_0001);

    let mut fs = Filesystem::init_with_chunk_size(cover, map, 64).expect("init");
    match fs.write(&data) {
        Err(FsError::FileTooLarge { max_chunks, .. }) => {
            assert_eq!(max_chunks, 12 + 4 + 16 + 64);
        }
        other => panic!("expected FileTooLarge, got {other:?}"),
    }
}

// ------------------------------------------------------------------
// Rewrites across tiers
// ------------------------------------------------------------------

#[test]
fn rewrite_shrinks_file_across_tiers() {
    let (cover, map) = make_cover();
    let mut fs = Filesystem::init_with_chunk_size(cover, map.clone(), 64).expect("init");

    // Large write → triple indirect.
    let large = pattern(50 * 64, 0xAAAA);
    fs.write(&large).expect("write large");
    assert_eq!(fs.read().expect("read"), large);

    // Small write → direct only. Old triple-indirect chunks leak
    // (per step 6/7 policy; reclaimed on next mount).
    let small = pattern(5 * 64, 0xBBBB);
    fs.write(&small).expect("write small");
    assert_eq!(fs.read().expect("read after shrink"), small);

    let cover_after = fs.unmount();
    let fs2 = Filesystem::mount_with_chunk_size(cover_after, map, 64).expect("mount");
    assert_eq!(fs2.read().expect("read after remount"), small);
}

#[test]
fn rewrite_grows_file_across_tiers() {
    let (cover, map) = make_cover();
    let mut fs = Filesystem::init_with_chunk_size(cover, map.clone(), 64).expect("init");

    let small = pattern(3 * 64, 0xCCCC);
    fs.write(&small).expect("write small");
    assert_eq!(fs.read().expect("read"), small);

    let large = pattern(40 * 64, 0xDDDD);
    fs.write(&large).expect("write large");
    assert_eq!(fs.read().expect("read after grow"), large);

    let cover_after = fs.unmount();
    let fs2 = Filesystem::mount_with_chunk_size(cover_after, map, 64).expect("mount");
    assert_eq!(fs2.read().expect("read after remount"), large);
}
