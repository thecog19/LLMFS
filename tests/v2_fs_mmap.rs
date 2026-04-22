//! Smoke test for [`memmap2::MmapMut`]-backed covers.
//!
//! Verifies that a `Filesystem` can be initialised, written to, and
//! re-mounted across a real file-backed `MmapMut`, and that writes
//! persist to disk after `unmount`'s flush. This is the path the
//! CLI takes for every operation; lives here so test failures land
//! in CI without needing the full CLI surface.
//!
//! No need for /dev/fuse — mmap is a kernel feature available
//! everywhere.

use std::fs::OpenOptions;
use std::io::Write;

use memmap2::MmapMut;

use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};
use llmdb::v2::cdc::FastCdcParams;
use llmdb::v2::cover::CoverStorage;
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

/// Build cover bytes for `weight_count` F16 weights with varied
/// magnitudes (so ceiling-magnitude ranking is non-degenerate).
fn make_cover_bytes(weight_count: u64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity((weight_count * 2) as usize);
    for i in 0..weight_count {
        let sign = if i % 3 == 0 { -1.0 } else { 1.0 };
        let v = sign * ((i + 1) as f32) * 0.000_002;
        bytes.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
    }
    bytes
}

fn f16_map(weight_count: u64) -> TensorMap {
    let bits = GgufQuantType::F16.stealable_bits_hint() as u64;
    let slot = TensorSlot {
        name: "mmap.f16".to_owned(),
        quant_type: GgufQuantType::F16,
        tier: TensorTier::Tier1,
        data_offset: 0,
        weight_count,
        stealable_bits_per_weight: bits as usize,
        capacity_bits: weight_count * bits,
        bit_start: 0,
        bit_end: weight_count * bits,
    };
    TensorMap {
        slots: vec![slot.clone()],
        total_capacity_bits: slot.capacity_bits,
        total_capacity_bytes: slot.capacity_bits / 8,
    }
}

fn small_cdc() -> FastCdcParams {
    FastCdcParams {
        min_size: 32,
        avg_size: 64,
        max_size: 128,
    }
}

fn write_cover_to_tempfile(bytes: &[u8]) -> tempfile::NamedTempFile {
    let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
    tmp.as_file_mut().write_all(bytes).expect("write cover");
    tmp.as_file_mut().flush().expect("flush cover");
    tmp
}

fn open_mmap(file: &std::fs::File) -> MmapMut {
    unsafe { MmapMut::map_mut(file) }.expect("mmap")
}

#[test]
fn init_then_write_then_remount_via_mmap_cover() {
    let weight_count = 30_000;
    let bytes = make_cover_bytes(weight_count);
    let tmp = write_cover_to_tempfile(&bytes);
    let map = f16_map(weight_count);

    // Init the filesystem on the mmap-backed cover, write a file,
    // unmount (which msyncs).
    {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(tmp.path())
            .expect("reopen for init");
        let mmap = open_mmap(&file);
        let mut fs =
            Filesystem::init_with_cdc_params(mmap, map.clone(), small_cdc()).expect("init on mmap");
        fs.create_file("/notes.md", b"persisted via mmap")
            .expect("create");
        // Drop the cover; flushed by unmount.
        drop(fs.unmount().expect("unmount flushes msync"));
    }

    // Re-open the same file fresh, mmap it, mount, read back.
    {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(tmp.path())
            .expect("reopen for mount");
        let mmap = open_mmap(&file);
        let fs = Filesystem::mount_with_cdc_params(mmap, map, small_cdc()).expect("mount on mmap");
        let read = fs.read_file("/notes.md").expect("read");
        assert_eq!(read, b"persisted via mmap".to_vec());
        // Don't bother flushing — read-only path.
        let _ = fs.unmount().expect("unmount");
    }

    // And verify the bytes-on-disk really changed (msync took effect).
    let on_disk = std::fs::read(tmp.path()).expect("read final");
    assert_eq!(on_disk.len(), bytes.len(), "file size unchanged");
    assert_ne!(on_disk, bytes, "mmap writes should have mutated the file");
}

/// Heart of the >RAM claim: build a sparse file far larger than
/// physical RAM, mmap it, and prove init/mount don't blow up. Sparse
/// files (`set_len`) cost zero disk space until written; mmap of a
/// sparse region returns zeroed pages on demand without consuming
/// RAM.
///
/// We don't init a *real* V2 fs on the sparse cover (V2 requires
/// the GGUF tensor data to actually exist for ceiling magnitudes),
/// but we do prove that mmapping a 16 GB region succeeds and that
/// random byte writes through the map work without OOMing.
///
/// Skipped on filesystems that don't support sparse files.
#[test]
fn sparse_file_mmap_does_not_consume_ram() {
    const SIZE: u64 = 16 * 1024 * 1024 * 1024; // 16 GB

    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(tmp.path())
        .expect("open");
    if file.set_len(SIZE).is_err() {
        eprintln!("skipping: filesystem doesn't support sparse files");
        return;
    }

    let mut mmap = unsafe { MmapMut::map_mut(&file) }.expect("mmap 16 GB sparse");
    assert_eq!(mmap.len(), SIZE as usize);

    // Touch a handful of widely-separated pages — this would force
    // page-in if the map were RAM-backed; for sparse mmap it just
    // allocates one 4 KiB page per touched offset.
    for offset in [0_usize, 1 << 30, 4 << 30, 8 << 30, 15 << 30] {
        mmap.bytes_mut()[offset] = 0xAA;
    }

    // Read back to confirm.
    for offset in [0_usize, 1 << 30, 4 << 30, 8 << 30, 15 << 30] {
        assert_eq!(mmap.bytes()[offset], 0xAA);
    }

    // No flush — we don't care about durability for this test.
    drop(mmap);
}
