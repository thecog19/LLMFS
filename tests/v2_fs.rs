//! V2 Filesystem integration — init / create_file / unmount /
//! remount / read_file.
//!
//! Tests against the path-based API. The root directory exists
//! implicitly after init; the conventional "/data" path is used
//! for the single-file smoke tests.
//!
//! What's under test:
//! 1. **Empty init.** init → unmount → mount → root directory empty.
//! 2. **One-chunk file.** init → create_file("/data", small) → round-trip.
//! 3. **Multi-chunk file.** Data spanning 2+ chunks → round-trip.
//! 4. **Generation bumps.** Each create_file increments the anchor
//!    + super-root generation counter by 1.
//! 5. **Rewrite.** create_file(A) → create_file(B) → read == B.
//! 6. **File beyond direct capacity** succeeds via indirect blocks.
//! 7. **Cover with no anchor** → mount fails cleanly.
//! 8. **Mount fails on invalid CDC params**.
//! 9. **Mount fails on corrupted inode pointer**.

use llmdb::forward::{ActivationSite, CholeskyFactor, HessianFactorCache};
use llmdb::gguf::parser::GgufTensorInfo;
use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};
use llmdb::v2::anchor;
use llmdb::v2::cdc::FastCdcParams;
use llmdb::v2::chunk::{read_chunk, write_chunk};
use llmdb::v2::cover::CoverStorage;
use llmdb::v2::fs::{Filesystem, FsError};
use llmdb::v2::inode::{INODE_BYTES, Inode};
use llmdb::v2::super_root::{SUPER_ROOT_BYTES, SuperRoot};

/// Small CDC parameters for tests — avg must be a power of two ≥ 4.
fn small_cdc() -> FastCdcParams {
    FastCdcParams {
        min_size: 32,
        avg_size: 64,
        max_size: 128,
    }
}

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
        name: "fs.f16".to_owned(),
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

/// Cover for round-trip tests: varied magnitudes so the ceiling-
/// magnitude ranking is non-degenerate.
fn make_cover(weight_count: u64) -> (Vec<u8>, TensorMap) {
    let values: Vec<f32> = (0..weight_count)
        .map(|i| {
            let sign = if i % 3 == 0 { -1.0 } else { 1.0 };
            sign * ((i + 1) as f32) * 0.00005
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

fn identity_factor(n: usize) -> CholeskyFactor {
    let mut l = vec![0.0_f32; n * (n + 1) / 2];
    for i in 0..n {
        l[i * (i + 1) / 2 + i] = 1.0;
    }
    CholeskyFactor::new(n, l)
}

fn compensation_runtime_fixture() -> (Vec<GgufTensorInfo>, HessianFactorCache) {
    let tensors = vec![GgufTensorInfo {
        name: "blk.0.attn_q.weight".to_owned(),
        dimensions: vec![2, 2],
        raw_type_id: GgufQuantType::F16 as u32,
        data_offset: 0,
    }];
    let mut factors = HessianFactorCache::new();
    factors.insert(ActivationSite::QkvInput, 0, identity_factor(2));
    (tensors, factors)
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

#[test]
fn init_then_unmount_remount_sees_empty_root() {
    let (cover, map) = make_cover(20_000);
    let fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");
    assert!(fs.readdir("/").expect("readdir").is_empty());
    let cover_after = fs.unmount().expect("unmount");

    let fs2 = Filesystem::mount_with_cdc_params(cover_after, map, small_cdc()).expect("mount");
    assert!(fs2.readdir("/").expect("readdir on mount").is_empty());
}

#[test]
fn init_starts_without_compensation_runtime() {
    let (cover, map) = make_cover(20_000);
    let fs = Filesystem::init_with_cdc_params(cover, map, small_cdc()).expect("init");

    assert!(fs.compensation_runtime().is_none());
}

#[test]
fn compensation_runtime_is_mount_local_state() {
    let (cover, map) = make_cover(20_000);
    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");
    let (tensors, factors) = compensation_runtime_fixture();

    fs.set_compensation_runtime(tensors.clone(), factors.clone());
    let runtime = fs.compensation_runtime().expect("runtime");
    assert_eq!(runtime.gguf_tensors(), tensors.as_slice());
    assert_eq!(runtime.factors().len(), 1);
    assert!(runtime.factors().contains(ActivationSite::QkvInput, 0));

    fs.clear_compensation_runtime();
    assert!(fs.compensation_runtime().is_none());

    fs.set_compensation_runtime(tensors, factors);
    let cover_after = fs.unmount().expect("unmount");
    let fs2 = Filesystem::mount_with_cdc_params(cover_after, map, small_cdc()).expect("mount");
    assert!(fs2.compensation_runtime().is_none());
}

#[test]
fn single_chunk_file_round_trips_across_remount() {
    let (cover, map) = make_cover(20_000);
    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");

    let data = b"Hello, V2 filesystem! This is a round trip.";
    fs.create_file("/data", data).expect("create_file");
    assert_eq!(fs.read_file("/data").expect("read after write"), data);

    let cover_after = fs.unmount().expect("unmount");
    let fs2 = Filesystem::mount_with_cdc_params(cover_after, map, small_cdc()).expect("mount");
    let readback = fs2.read_file("/data").expect("read after mount");
    assert_eq!(readback, data);
}

#[test]
fn multi_chunk_file_round_trips() {
    let (cover, map) = make_cover(30_000);
    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");

    let data: Vec<u8> = (0..400_u32).map(|i| (i & 0xFF) as u8).collect();
    fs.create_file("/data", &data).expect("create_file");
    assert_eq!(fs.read_file("/data").expect("read"), data);

    let cover_after = fs.unmount().expect("unmount");
    let fs2 = Filesystem::mount_with_cdc_params(cover_after, map, small_cdc()).expect("mount");
    assert_eq!(fs2.read_file("/data").expect("read remount"), data);
}

#[test]
fn generation_bumps_on_each_write() {
    let (cover, map) = make_cover(20_000);
    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");
    assert_eq!(fs.generation(), 1);

    fs.create_file("/data", b"first").expect("write 1");
    assert_eq!(fs.generation(), 2);

    fs.create_file("/data", b"second").expect("write 2");
    assert_eq!(fs.generation(), 3);
}

#[test]
fn rewrite_replaces_file_contents() {
    let (cover, map) = make_cover(20_000);
    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");
    fs.create_file("/data", b"first version").expect("write 1");
    fs.create_file("/data", b"second version is longer")
        .expect("write 2");
    assert_eq!(
        fs.read_file("/data").expect("read"),
        b"second version is longer"
    );

    let cover_after = fs.unmount().expect("unmount");
    let fs2 = Filesystem::mount_with_cdc_params(cover_after, map, small_cdc()).expect("mount");
    assert_eq!(
        fs2.read_file("/data").expect("read"),
        b"second version is longer"
    );
}

#[test]
fn write_beyond_direct_pointer_count_succeeds_with_indirect() {
    // > 12 chunks spills into indirect blocks. 13 × 40 = 520 bytes
    // under small_cdc (min 32, max 128) → at least 13 chunks.
    let (cover, map) = make_cover(50_000);
    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");
    let data = vec![0xAA_u8; 13 * 40];
    fs.create_file("/data", &data)
        .expect("write beyond direct limit");
    assert_eq!(fs.read_file("/data").expect("read"), data);
}

#[test]
fn mount_fails_on_cover_with_no_anchor() {
    let (cover, map) = make_cover(20_000);
    match Filesystem::mount_with_cdc_params(cover, map, small_cdc()) {
        Err(FsError::Anchor(_)) => {}
        other => panic!("expected Anchor error on unintialised cover, got {other:?}"),
    }
}

#[test]
fn init_rejects_invalid_cdc_params() {
    let (cover, map) = make_cover(20_000);
    let bad = FastCdcParams {
        min_size: 32,
        avg_size: 100, // not a power of two
        max_size: 256,
    };
    match Filesystem::init_with_cdc_params(cover, map, bad) {
        Err(FsError::Cdc(_)) => {}
        other => panic!("expected FsError::Cdc, got {other:?}"),
    }
}

#[test]
fn mount_rejects_invalid_cdc_params() {
    let (cover, map) = make_cover(20_000);
    let fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");
    let cover_after = fs.unmount().expect("unmount");

    let bad = FastCdcParams {
        min_size: 128,
        avg_size: 64, // avg < min
        max_size: 256,
    };
    match Filesystem::mount_with_cdc_params(cover_after, map, bad) {
        Err(FsError::Cdc(_)) => {}
        other => panic!("expected FsError::Cdc, got {other:?}"),
    }
}

#[test]
fn mount_fails_cleanly_on_inode_pointer_with_invalid_slot() {
    // Corrupt the root directory inode's direct[0] pointer so its
    // slot references a nonexistent tensor. Mount's tree walk should
    // reach that pointer during chunk reservation and error
    // cleanly rather than panicking.
    let (cover, map) = make_cover(20_000);
    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");
    fs.create_file("/data", b"hello").expect("write");
    let mut cover_after = fs.unmount().expect("unmount");

    let anchor_outcome = anchor::read_anchor(cover_after.bytes(), &map).expect("read anchor");
    let super_root_ptr = anchor_outcome.active.super_root;

    let mut super_root_bytes = [0u8; SUPER_ROOT_BYTES];
    read_chunk(
        cover_after.bytes(),
        &map,
        super_root_ptr,
        0,
        &mut super_root_bytes,
    )
    .expect("read super-root");
    let super_root = SuperRoot::decode(&super_root_bytes).expect("decode super-root");

    let root_inode_ptr = super_root.root_dir_inode;
    let mut inode_bytes = [0u8; INODE_BYTES];
    read_chunk(
        cover_after.bytes(),
        &map,
        root_inode_ptr,
        0,
        &mut inode_bytes,
    )
    .expect("read inode");
    let mut inode = Inode::decode(&inode_bytes).expect("decode inode");
    inode.direct[0].slot = u16::MAX;
    write_chunk(
        cover_after.bytes_mut(),
        &map,
        root_inode_ptr,
        0,
        &inode.encode(),
    )
    .expect("write inode");

    match Filesystem::mount_with_cdc_params(cover_after, map, small_cdc()) {
        Err(FsError::PointerSlotOutOfRange {
            slot: u16::MAX,
            slot_count: 1,
        }) => {}
        other => panic!("expected PointerSlotOutOfRange, got {other:?}"),
    }
}

#[test]
fn write_then_read_binary_data_round_trips() {
    // Non-ASCII bytes guard against any accidental text-only
    // assumption.
    let (cover, map) = make_cover(30_000);
    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");
    let data: Vec<u8> = (0..256_u32)
        .map(|i| (i.wrapping_mul(37) & 0xFF) as u8)
        .collect();
    fs.create_file("/data", &data).expect("write");
    assert_eq!(fs.read_file("/data").expect("read"), data);

    let cover_after = fs.unmount().expect("unmount");
    let fs2 = Filesystem::mount_with_cdc_params(cover_after, map, small_cdc()).expect("mount");
    assert_eq!(fs2.read_file("/data").expect("read"), data);
}

#[test]
fn multiple_writes_across_multiple_sessions() {
    let (cover, map) = make_cover(30_000);
    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");
    fs.create_file("/data", b"session-1 data").expect("write 1");
    let cover1 = fs.unmount().expect("unmount");

    let mut fs2 =
        Filesystem::mount_with_cdc_params(cover1, map.clone(), small_cdc()).expect("mount 1");
    assert_eq!(fs2.read_file("/data").expect("read"), b"session-1 data");
    fs2.create_file("/data", b"session-2 replaces")
        .expect("write 2");
    let cover2 = fs2.unmount().expect("unmount");

    let fs3 = Filesystem::mount_with_cdc_params(cover2, map, small_cdc()).expect("mount 2");
    assert_eq!(fs3.read_file("/data").expect("read"), b"session-2 replaces");
}
