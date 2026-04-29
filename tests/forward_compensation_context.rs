use llmdb::forward::linalg::cholesky;
use llmdb::forward::{
    ActivationSite, AppliedCompensationRegion, CholeskyFactor, CompensatedChunkWriteError,
    CompensationApplyError, CompensationRegionKey, CompensationWriteDeltaRegion,
    CompensationWriteRegion, HessianFactorCache, apply_cached_compensation,
    apply_compensation_to_cover, delta_regions_for_weight_deltas, regions_for_pointer,
    write_chunk_with_cached_compensation,
};
use llmdb::gguf::parser::GgufTensorInfo;
use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::packing::float::f16_to_f32;
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};
use llmdb::v2::chunk::WeightDelta;
use llmdb::v2::pointer::Pointer;

#[test]
fn forward_reexports_pointer_region_api() {
    let name = "blk.0.attn_q.weight";
    let map = TensorMap {
        slots: vec![TensorSlot {
            name: name.to_owned(),
            quant_type: GgufQuantType::F16,
            tier: TensorTier::Tier1,
            data_offset: 0,
            weight_count: 4,
            stealable_bits_per_weight: 4,
            capacity_bits: 16,
            bit_start: 0,
            bit_end: 16,
        }],
        total_capacity_bits: 16,
        total_capacity_bytes: 2,
    };
    let tensors = vec![GgufTensorInfo {
        name: name.to_owned(),
        dimensions: vec![2, 2],
        raw_type_id: GgufQuantType::F16 as u32,
        data_offset: 0,
    }];
    let pointer = Pointer {
        slot: 0,
        start_weight: 0,
        length_in_bits: 8,
        flags: 0,
        reserved: 0,
    };

    let regions = regions_for_pointer(&map, &tensors, pointer).expect("regions");

    assert_eq!(
        regions,
        vec![CompensationWriteRegion {
            key: CompensationRegionKey {
                tensor_name: name.to_owned(),
                site: ActivationSite::QkvInput,
                layer: 0,
                output_channel: 0,
            },
            input_channels: vec![0, 1],
        }]
    );

    let delta_regions = delta_regions_for_weight_deltas(
        &map,
        &tensors,
        &[WeightDelta {
            slot: 0,
            weight_index: 0,
            before: 1.0,
            after: 1.5,
        }],
    )
    .expect("delta regions");

    assert_eq!(
        delta_regions,
        vec![CompensationWriteDeltaRegion {
            key: CompensationRegionKey {
                tensor_name: name.to_owned(),
                site: ActivationSite::QkvInput,
                layer: 0,
                output_channel: 0,
            },
            input_channels: vec![0],
            deltas: vec![0.5],
        }]
    );

    let mut cache = HessianFactorCache::new();
    cache.insert(
        ActivationSite::QkvInput,
        0,
        CholeskyFactor::new(2, cholesky(&[4.0_f32, 2.0, 5.0], 2).unwrap()),
    );

    let applied = apply_cached_compensation(&cache, &delta_regions).expect("applied");

    let expected = AppliedCompensationRegion {
        key: CompensationRegionKey {
            tensor_name: name.to_owned(),
            site: ActivationSite::QkvInput,
            layer: 0,
            output_channel: 0,
        },
        forced_input_channels: vec![0],
        compensation_input_channels: vec![1],
        compensation_deltas: vec![-0.2],
    };
    assert_eq!(applied[0].key, expected.key);
    assert_eq!(
        applied[0].forced_input_channels,
        expected.forced_input_channels
    );
    assert_eq!(
        applied[0].compensation_input_channels,
        expected.compensation_input_channels
    );
    assert!(
        (applied[0].compensation_deltas[0] - expected.compensation_deltas[0]).abs() < 1e-6,
        "compensation delta = {}",
        applied[0].compensation_deltas[0],
    );
}

#[test]
fn apply_compensation_to_cover_writes_nearest_weight_values() {
    let name = "blk.0.attn_q.weight";
    let mut cover = f16_cover_bits(&[0x3C00, 0x4000, 0x4200, 0x4400]);
    let map = TensorMap {
        slots: vec![TensorSlot {
            name: name.to_owned(),
            quant_type: GgufQuantType::F16,
            tier: TensorTier::Tier1,
            data_offset: 0,
            weight_count: 4,
            stealable_bits_per_weight: 4,
            capacity_bits: 16,
            bit_start: 0,
            bit_end: 16,
        }],
        total_capacity_bits: 16,
        total_capacity_bytes: 2,
    };
    let tensors = vec![GgufTensorInfo {
        name: name.to_owned(),
        dimensions: vec![2, 2],
        raw_type_id: GgufQuantType::F16 as u32,
        data_offset: 0,
    }];
    let before = f16_to_f32(0x4200);
    let after = f16_to_f32(0x4205);
    let region = AppliedCompensationRegion {
        key: CompensationRegionKey {
            tensor_name: name.to_owned(),
            site: ActivationSite::QkvInput,
            layer: 0,
            output_channel: 1,
        },
        forced_input_channels: vec![1],
        compensation_input_channels: vec![0],
        compensation_deltas: vec![after - before],
    };

    let deltas = apply_compensation_to_cover(&mut cover, &map, &tensors, &[region])
        .expect("apply compensation");

    assert_eq!(
        deltas,
        vec![WeightDelta {
            slot: 0,
            weight_index: 2,
            before,
            after,
        }]
    );
    assert_eq!(u16::from_le_bytes([cover[4], cover[5]]), 0x4205);
    assert_eq!(u16::from_le_bytes([cover[6], cover[7]]), 0x4400);
}

#[test]
fn apply_compensation_to_cover_rejects_mismatched_region_without_mutation() {
    let name = "blk.0.attn_q.weight";
    let mut cover = f16_cover_bits(&[0x3C00, 0x4000, 0x4200, 0x4400]);
    let before_cover = cover.clone();
    let map = TensorMap {
        slots: vec![TensorSlot {
            name: name.to_owned(),
            quant_type: GgufQuantType::F16,
            tier: TensorTier::Tier1,
            data_offset: 0,
            weight_count: 4,
            stealable_bits_per_weight: 4,
            capacity_bits: 16,
            bit_start: 0,
            bit_end: 16,
        }],
        total_capacity_bits: 16,
        total_capacity_bytes: 2,
    };
    let tensors = vec![GgufTensorInfo {
        name: name.to_owned(),
        dimensions: vec![2, 2],
        raw_type_id: GgufQuantType::F16 as u32,
        data_offset: 0,
    }];
    let region = AppliedCompensationRegion {
        key: CompensationRegionKey {
            tensor_name: name.to_owned(),
            site: ActivationSite::QkvInput,
            layer: 0,
            output_channel: 1,
        },
        forced_input_channels: vec![1],
        compensation_input_channels: vec![0, 1],
        compensation_deltas: vec![0.001],
    };

    let err = apply_compensation_to_cover(&mut cover, &map, &tensors, &[region])
        .expect_err("mismatched compensation region");

    assert!(format!("{err}").contains("2 compensation channels but 1 deltas"));
    assert_eq!(cover, before_cover);
}

#[test]
fn write_chunk_with_cached_compensation_applies_payload_and_compensation() {
    let name = "blk.0.attn_q.weight";
    let mut cover = f16_cover_bits(&[0x3C0F, 0x4000, 0x4200, 0x4400]);
    let map = f16_tensor_map(name, 4);
    let tensors = vec![tensor(name, 2, 2)];
    let mut cache = HessianFactorCache::new();
    cache.insert(
        ActivationSite::QkvInput,
        0,
        CholeskyFactor::new(2, cholesky(&[4.0_f32, 2.0, 5.0], 2).unwrap()),
    );
    let pointer = Pointer {
        slot: 0,
        start_weight: 0,
        length_in_bits: 4,
        flags: 0,
        reserved: 0,
    };

    let result = write_chunk_with_cached_compensation(
        &mut cover,
        &map,
        &tensors,
        &cache,
        pointer,
        0,
        &[0x00],
    )
    .expect("compensated chunk write");

    assert_eq!(
        result.forced_deltas,
        vec![WeightDelta {
            slot: 0,
            weight_index: 0,
            before: f16_to_f32(0x3C0F),
            after: f16_to_f32(0x3C00),
        }]
    );
    assert_eq!(
        result.compensation_deltas,
        vec![WeightDelta {
            slot: 0,
            weight_index: 1,
            before: f16_to_f32(0x4000),
            after: f16_to_f32(0x4003),
        }]
    );
    assert_eq!(u16::from_le_bytes([cover[0], cover[1]]), 0x3C00);
    assert_eq!(u16::from_le_bytes([cover[2], cover[3]]), 0x4003);
}

#[test]
fn write_chunk_with_cached_compensation_preflights_missing_factor_without_mutation() {
    let name = "blk.0.attn_q.weight";
    let mut cover = f16_cover_bits(&[0x3C0F, 0x4000, 0x4200, 0x4400]);
    let before_cover = cover.clone();
    let map = f16_tensor_map(name, 4);
    let tensors = vec![tensor(name, 2, 2)];
    let pointer = Pointer {
        slot: 0,
        start_weight: 0,
        length_in_bits: 4,
        flags: 0,
        reserved: 0,
    };

    let err = write_chunk_with_cached_compensation(
        &mut cover,
        &map,
        &tensors,
        &HessianFactorCache::new(),
        pointer,
        0,
        &[0x00],
    )
    .expect_err("missing factor");

    assert!(matches!(
        err,
        CompensatedChunkWriteError::Apply {
            source: CompensationApplyError::MissingFactor {
                site: ActivationSite::QkvInput,
                layer: 0,
            }
        }
    ));
    assert_eq!(cover, before_cover);
}

fn f16_cover_bits(bits: &[u16]) -> Vec<u8> {
    let mut cover = Vec::with_capacity(bits.len() * 2);
    for value in bits {
        cover.extend_from_slice(&value.to_le_bytes());
    }
    cover
}

fn f16_tensor_map(name: &str, weight_count: u64) -> TensorMap {
    TensorMap {
        slots: vec![TensorSlot {
            name: name.to_owned(),
            quant_type: GgufQuantType::F16,
            tier: TensorTier::Tier1,
            data_offset: 0,
            weight_count,
            stealable_bits_per_weight: 4,
            capacity_bits: weight_count * 4,
            bit_start: 0,
            bit_end: weight_count * 4,
        }],
        total_capacity_bits: weight_count * 4,
        total_capacity_bytes: weight_count / 2,
    }
}

fn tensor(name: &str, input_dim: u64, output_dim: u64) -> GgufTensorInfo {
    GgufTensorInfo {
        name: name.to_owned(),
        dimensions: vec![input_dim, output_dim],
        raw_type_id: GgufQuantType::F16 as u32,
        data_offset: 0,
    }
}
