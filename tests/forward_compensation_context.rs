use llmdb::forward::linalg::cholesky;
use llmdb::forward::{
    ActivationSite, AppliedCompensationRegion, CholeskyFactor, CompensationRegionKey,
    CompensationWriteDeltaRegion, CompensationWriteRegion, HessianFactorCache,
    apply_cached_compensation, delta_regions_for_weight_deltas, regions_for_pointer,
};
use llmdb::gguf::parser::GgufTensorInfo;
use llmdb::gguf::quant::GgufQuantType;
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
