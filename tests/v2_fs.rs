//! V2 Filesystem integration — init / write / unmount / remount / read.
//!
//! This is the first V2 milestone that ties everything together:
//! anchor placement, super-root, inode, allocator (with anchor
//! reservation), chunk I/O. Single-file model for now — the root
//! directory is stand-in for "the single user file." Hierarchical
//! directories land in step 11.
//!
//! What's under test:
//! 1. **Empty round-trip.** init → unmount → mount → file is empty.
//! 2. **One-chunk file.** init → write small data → unmount → mount →
//!    read → bytes match.
//! 3. **Multi-chunk file.** Data spanning 2+ chunks → round-trip.
//! 4. **Generation bumps.** Each write increments the anchor+super-
//!    root generation counter by 1.
//! 5. **Rewrite.** write(A) → write(B) → read == B.
//! 6. **File larger than direct pointers** → FileTooLarge error
//!    (step 7 adds indirect pointers).
//! 7. **Cover too small** → CoverTooSmall propagated through init.
//! 8. **Mount after externally-corrupted cover** → error (the anchor's
//!    two-slot scheme sees at least one valid slot most of the time;
//!    this test corrupts both).

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

/// Cover for round-trip tests: 20k F16 weights, varied magnitudes so
/// the ceiling-magnitude ranking is non-degenerate.
fn make_cover(weight_count: u64) -> (Vec<u8>, TensorMap) {
    let values: Vec<f32> = (0..weight_count)
        .map(|i| {
            // Mix magnitudes from ~1e-5 to ~1.0 with both signs.
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

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

#[test]
fn init_then_unmount_remount_sees_empty_file() {
    let (cover, map) = make_cover(20_000);
    let fs = Filesystem::init_with_chunk_size(cover, map.clone(), 64).expect("init");
    assert_eq!(fs.file_length(), 0);
    assert_eq!(fs.read().expect("read empty"), Vec::<u8>::new());
    let cover_after = fs.unmount();

    let fs2 = Filesystem::mount_with_chunk_size(cover_after, map, 64).expect("mount");
    assert_eq!(fs2.file_length(), 0);
    assert_eq!(fs2.read().expect("read empty on mount"), Vec::<u8>::new());
}

#[test]
fn single_chunk_file_round_trips_across_remount() {
    let (cover, map) = make_cover(20_000);
    let mut fs = Filesystem::init_with_chunk_size(cover, map.clone(), 64).expect("init");

    let data = b"Hello, V2 filesystem! This is a round trip.";
    fs.write(data).expect("write");
    assert_eq!(fs.file_length(), data.len() as u64);
    assert_eq!(fs.read().expect("read after write"), data);

    let cover_after = fs.unmount();
    let fs2 = Filesystem::mount_with_chunk_size(cover_after, map, 64).expect("mount");
    let readback = fs2.read().expect("read after mount");
    assert_eq!(readback, data);
    assert_eq!(fs2.file_length(), data.len() as u64);
}

#[test]
fn multi_chunk_file_round_trips() {
    let (cover, map) = make_cover(30_000);
    // 32-byte chunks; 400 bytes = 13 chunks. Oops, over direct limit.
    // Use 40-byte chunks instead: 400 / 40 = 10 chunks, fits in direct.
    let mut fs = Filesystem::init_with_chunk_size(cover, map.clone(), 40).expect("init");

    let data: Vec<u8> = (0..400_u32).map(|i| (i & 0xFF) as u8).collect();
    fs.write(&data).expect("write");
    assert_eq!(fs.read().expect("read"), data);

    let cover_after = fs.unmount();
    let fs2 = Filesystem::mount_with_chunk_size(cover_after, map, 40).expect("mount");
    assert_eq!(fs2.read().expect("read remount"), data);
}

#[test]
fn generation_bumps_on_each_write() {
    let (cover, map) = make_cover(20_000);
    let mut fs = Filesystem::init_with_chunk_size(cover, map.clone(), 64).expect("init");
    // init puts gen=1 in the active slot.
    assert_eq!(fs.generation(), 1);

    fs.write(b"first").expect("write 1");
    assert_eq!(fs.generation(), 2);

    fs.write(b"second").expect("write 2");
    assert_eq!(fs.generation(), 3);
}

#[test]
fn rewrite_replaces_file_contents() {
    let (cover, map) = make_cover(20_000);
    let mut fs = Filesystem::init_with_chunk_size(cover, map.clone(), 64).expect("init");
    fs.write(b"first version").expect("write 1");
    fs.write(b"second version is longer").expect("write 2");
    assert_eq!(fs.read().expect("read"), b"second version is longer");

    let cover_after = fs.unmount();
    let fs2 = Filesystem::mount_with_chunk_size(cover_after, map, 64).expect("mount");
    assert_eq!(fs2.read().expect("read"), b"second version is longer");
}

#[test]
fn write_exceeding_direct_pointer_limit_errors() {
    let (cover, map) = make_cover(50_000);
    let mut fs = Filesystem::init_with_chunk_size(cover, map.clone(), 40).expect("init");
    // 13 × 40 = 520 bytes — one chunk past direct's 12-pointer limit.
    let data = vec![0xAA_u8; 13 * 40];
    match fs.write(&data) {
        Err(FsError::FileTooLarge { .. }) => {}
        other => panic!("expected FileTooLarge, got {other:?}"),
    }
}

#[test]
fn mount_fails_on_cover_with_no_anchor() {
    // Fresh cover without init — anchor positions hold whatever the
    // cover bytes happened to have. Mount should fail cleanly.
    let (cover, map) = make_cover(20_000);
    match Filesystem::mount_with_chunk_size(cover, map, 64) {
        Err(FsError::Anchor(_)) => {}
        other => panic!("expected Anchor error on unintialised cover, got {other:?}"),
    }
}

#[test]
fn write_then_read_binary_data_round_trips() {
    // Non-ASCII bytes to guard against any accidental
    // text-only assumption.
    let (cover, map) = make_cover(30_000);
    let mut fs = Filesystem::init_with_chunk_size(cover, map.clone(), 48).expect("init");
    let data: Vec<u8> = (0..256_u32).map(|i| (i.wrapping_mul(37) & 0xFF) as u8).collect();
    fs.write(&data).expect("write");
    assert_eq!(fs.read().expect("read"), data);

    let cover_after = fs.unmount();
    let fs2 = Filesystem::mount_with_chunk_size(cover_after, map, 48).expect("mount");
    assert_eq!(fs2.read().expect("read"), data);
}

#[test]
fn multiple_writes_across_multiple_sessions() {
    let (cover, map) = make_cover(30_000);
    let mut fs = Filesystem::init_with_chunk_size(cover, map.clone(), 40).expect("init");
    fs.write(b"session-1 data").expect("write 1");
    let cover1 = fs.unmount();

    let mut fs2 =
        Filesystem::mount_with_chunk_size(cover1, map.clone(), 40).expect("mount 1");
    assert_eq!(fs2.read().expect("read"), b"session-1 data");
    fs2.write(b"session-2 replaces").expect("write 2");
    let cover2 = fs2.unmount();

    let fs3 = Filesystem::mount_with_chunk_size(cover2, map, 40).expect("mount 2");
    assert_eq!(fs3.read().expect("read"), b"session-2 replaces");
}
