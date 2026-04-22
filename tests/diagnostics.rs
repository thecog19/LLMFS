//! V2 diagnostics — gather + format on a real `v2::Filesystem`.

use llmdb::diagnostics::{format_human, gather};
use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};
use llmdb::v2::cdc::FastCdcParams;
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

fn make_cover(weight_count: u64) -> (Vec<u8>, TensorMap) {
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
    let bits = GgufQuantType::F16.stealable_bits_hint() as u64;
    let slot = TensorSlot {
        name: "diag.f16".to_owned(),
        quant_type: GgufQuantType::F16,
        tier: TensorTier::Tier1,
        data_offset: 0,
        weight_count,
        stealable_bits_per_weight: bits as usize,
        capacity_bits: weight_count * bits,
        bit_start: 0,
        bit_end: weight_count * bits,
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

#[test]
fn fresh_filesystem_reports_no_files_and_no_dirty_bits() {
    let (cover, map) = make_cover(20_000);
    let fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");

    let status = gather(&fs, &map).expect("gather");

    assert_eq!(status.file_count, 0);
    assert_eq!(status.directory_count, 0);
    assert_eq!(status.total_stored_bytes, 0);
    // Init writes the root directory + serialized dirty bitmap as
    // chunks; both go through dedup. The exact count depends on CDC
    // boundaries — we just assert it's non-empty and bounded.
    assert!(status.dedup_entries > 0);
    assert_eq!(status.dirty_bits_total, 20_000);
    // Init perturbs some weights (anchor + the chunks above); not all
    // 20K, but more than zero.
    assert!(status.dirty_bits_set > 0);
    assert!(status.dirty_bits_set < status.dirty_bits_total);
    assert_eq!(status.allocator_total_capacity_weights, 20_000);
    assert!(status.allocator_free_weights <= 20_000);
    assert_eq!(status.quant_profile, vec![GgufQuantType::F16]);
    assert!(status.generation >= 1);
}

#[test]
fn nested_files_and_directories_count_correctly() {
    let (cover, map) = make_cover(30_000);
    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");

    fs.create_file("/alpha.txt", b"first file").expect("alpha");
    fs.mkdir("/docs").expect("mkdir docs");
    fs.create_file("/docs/notes.md", b"important notes")
        .expect("notes");
    fs.mkdir("/docs/sub").expect("mkdir sub");
    fs.create_file("/docs/sub/leaf.txt", b"hi").expect("leaf");

    let status = gather(&fs, &map).expect("gather");
    assert_eq!(status.file_count, 3);
    assert_eq!(status.directory_count, 2); // /docs and /docs/sub
    assert_eq!(
        status.total_stored_bytes,
        ("first file".len() + "important notes".len() + "hi".len()) as u64
    );
}

#[test]
fn generation_advances_across_writes() {
    let (cover, map) = make_cover(20_000);
    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");
    let baseline = gather(&fs, &map).expect("gather").generation;

    fs.create_file("/a", b"hi").expect("create");
    let after = gather(&fs, &map).expect("gather").generation;

    assert!(
        after > baseline,
        "generation must advance after a write (baseline={baseline}, after={after})"
    );
}

#[test]
fn dirty_bits_grow_monotonically_with_writes() {
    let (cover, map) = make_cover(30_000);
    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");
    let baseline = gather(&fs, &map).expect("gather").dirty_bits_set;

    fs.create_file("/a.bin", &vec![0xAA; 4096]).expect("a");
    let after_one = gather(&fs, &map).expect("gather").dirty_bits_set;
    fs.create_file("/b.bin", &vec![0xBB; 4096]).expect("b");
    let after_two = gather(&fs, &map).expect("gather").dirty_bits_set;

    assert!(after_one >= baseline);
    assert!(after_two >= after_one);
}

#[test]
fn allocator_used_weights_grow_with_writes() {
    let (cover, map) = make_cover(30_000);
    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");
    let baseline_used = {
        let s = gather(&fs, &map).expect("gather");
        s.allocator_total_capacity_weights - s.allocator_free_weights
    };

    fs.create_file("/big.bin", &vec![0xCC; 8192]).expect("big");

    let after_used = {
        let s = gather(&fs, &map).expect("gather");
        s.allocator_total_capacity_weights - s.allocator_free_weights
    };
    assert!(
        after_used > baseline_used,
        "writing 8 KiB must use more weights (baseline={baseline_used}, after={after_used})"
    );
}

#[test]
fn format_human_output_names_every_field() {
    let (cover, map) = make_cover(20_000);
    let mut fs = Filesystem::init_with_cdc_params(cover, map.clone(), small_cdc()).expect("init");
    fs.create_file("/greet.txt", b"hello").expect("create");
    let status = gather(&fs, &map).expect("gather");
    let report = format_human(&status);

    for label in [
        "generation:",
        "files:",
        "directories:",
        "stored:",
        "dirty weights:",
        "allocator:",
        "dedup entries:",
        "quant profile:",
    ] {
        assert!(
            report.contains(label),
            "format_human missing `{label}`:\n{report}"
        );
    }
}
