use llmdb::forward::{
    ActivationSite, CompensationRegionKey, CompensationWriteRegion, regions_for_pointer,
};
use llmdb::gguf::parser::GgufTensorInfo;
use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};
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
}
