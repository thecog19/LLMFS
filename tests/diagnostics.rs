mod common;

use common::{SyntheticGgufVersion, SyntheticTensorSpec, write_custom_gguf_fixture};
use llmdb::diagnostics::{PerplexityBucket, format_human, gather};
use llmdb::gguf::quant::GGML_TYPE_Q8_0_ID;
use llmdb::stego::device::{DeviceOptions, StegoDevice};
use llmdb::stego::planner::{AllocationMode, TensorTier};

fn q8_tensors(count: usize) -> Vec<SyntheticTensorSpec> {
    (0..count)
        .map(|i| SyntheticTensorSpec {
            name: format!("blk.{}.ffn_down.weight", count - 1 - i),
            dimensions: vec![8192],
            raw_type_id: GGML_TYPE_Q8_0_ID,
            data: vec![0_u8; (8192 / 32) * 34],
        })
        .collect()
}

fn open_device(name: &str, tensor_count: usize) -> (common::FixtureHandle, StegoDevice) {
    let fx = write_custom_gguf_fixture(SyntheticGgufVersion::V3, name, &q8_tensors(tensor_count));
    let device = StegoDevice::initialize_with_options(
        &fx.path,
        AllocationMode::Standard,
        DeviceOptions { verbose: false },
    )
    .expect("init");
    (fx, device)
}

#[test]
fn fresh_device_reports_zero_files_and_negligible_impact() {
    let (_fx, device) = open_device("diag_fresh.gguf", 12);
    let status = gather(&device).expect("gather");

    assert_eq!(status.file_count, 0);
    assert_eq!(status.total_stored_bytes, 0);
    assert!(!status.dirty_on_open);
    assert!(!status.lobotomy);
    assert_eq!(status.used_blocks, 4); // 4 metadata blocks for a 12-tensor fixture
    assert!(status.utilization_pct > 0.0); // metadata occupies some

    // Freshly init'd: only metadata consumed, perplexity impact should
    // land in the low end of the scale.
    assert!(
        matches!(
            status.estimated_perplexity_impact.bucket,
            PerplexityBucket::Negligible | PerplexityBucket::Low
        ),
        "got {:?}",
        status.estimated_perplexity_impact.bucket
    );
}

#[test]
fn store_two_files_updates_file_count_and_stored_bytes() {
    let (_fx, mut device) = open_device("diag_stored.gguf", 16);
    device
        .store_bytes(&vec![0xAA; 1024], "first.bin", 0o644)
        .expect("store first");
    device
        .store_bytes(&vec![0xBB; 3072], "second.bin", 0o644)
        .expect("store second");

    let status = gather(&device).expect("gather");
    assert_eq!(status.file_count, 2);
    assert_eq!(status.total_stored_bytes, 1024 + 3072);
    assert!(status.utilization_pct > 0.0);
}

#[test]
fn impact_score_monotonically_increases_with_utilization() {
    let (_fx, mut device) = open_device("diag_monotonic.gguf", 20);
    let baseline = gather(&device).expect("gather baseline");

    device
        .store_bytes(&vec![1_u8; 4096], "a.bin", 0o644)
        .expect("store a");
    let after_one = gather(&device).expect("gather 1");

    device
        .store_bytes(&vec![2_u8; 4096], "b.bin", 0o644)
        .expect("store b");
    let after_two = gather(&device).expect("gather 2");

    assert!(
        after_one.estimated_perplexity_impact.score >= baseline.estimated_perplexity_impact.score
    );
    assert!(
        after_two.estimated_perplexity_impact.score >= after_one.estimated_perplexity_impact.score
    );
    assert!(after_two.total_stored_bytes > after_one.total_stored_bytes);
}

#[test]
fn per_tier_breakdown_includes_tier1_for_ffn_fixture() {
    let (_fx, device) = open_device("diag_tiers.gguf", 12);
    let status = gather(&device).expect("gather");

    // All ffn_down tensors are Tier1. Lobotomy mode is off so nothing in
    // Tier2 / Lobotomy exists.
    assert!(status.tier_utilization.contains_key(&TensorTier::Tier1));
    let tier1 = &status.tier_utilization[&TensorTier::Tier1];
    assert!(tier1.tensor_count >= 1);
    assert!(tier1.capacity_bytes > 0);
}

#[test]
fn format_human_output_names_every_field() {
    let (_fx, mut device) = open_device("diag_format.gguf", 12);
    device
        .store_bytes(b"hello", "greetings.txt", 0o644)
        .expect("store");
    let status = gather(&device).expect("gather");
    let report = format_human(&status);

    for label in [
        "total:",
        "used:",
        "free:",
        "utilization:",
        "files:",
        "stored:",
        "quant:",
        "lobotomy:",
        "dirty:",
        "per-tier breakdown:",
        "est. perplexity impact:",
    ] {
        assert!(
            report.contains(label),
            "format_human missing `{label}`:\n{report}"
        );
    }
}
