use llmdb::gguf::parser::GgufTensorInfo;
use llmdb::gguf::quant::{GGML_TYPE_Q5_K_ID, GGML_TYPE_Q6_K_ID, GGML_TYPE_Q8_0_ID, GgufQuantType};
use llmdb::stego::planner::{AllocationMode, TensorTier, build_allocation_plan};
use llmdb::stego::tensor_map::TensorMap;

#[test]
fn tensor_map_builds_contiguous_bit_ranges_from_allocation_plan() {
    let plan = build_allocation_plan(&mapping_tensors(), AllocationMode::Standard);
    let map = TensorMap::from_allocation_plan(&plan);

    assert_eq!(map.total_capacity_bits, 20);
    assert_eq!(map.total_capacity_bytes, 2);
    assert_eq!(map.slots.len(), 3);

    assert_eq!(map.slots[0].name, "blk.7.ffn_down.weight");
    assert_eq!(map.slots[0].bit_start, 0);
    assert_eq!(map.slots[0].bit_end, 8);
    assert_eq!(map.slots[0].quant_type, GgufQuantType::Q8_0);
    assert_eq!(map.slots[0].tier, TensorTier::Tier1);
    assert_eq!(map.slots[1].tier, TensorTier::Tier1);
    assert_eq!(map.slots[2].tier, TensorTier::Tier2);

    assert_eq!(map.slots[1].name, "blk.2.ffn_up.weight");
    assert_eq!(map.slots[1].bit_start, 8);
    assert_eq!(map.slots[1].bit_end, 12);
    assert_eq!(map.slots[1].quant_type, GgufQuantType::Q5K);

    assert_eq!(map.slots[2].name, "blk.1.attn_q.weight");
    assert_eq!(map.slots[2].bit_start, 12);
    assert_eq!(map.slots[2].bit_end, 20);
    assert_eq!(map.slots[2].quant_type, GgufQuantType::Q6K);
}

#[test]
fn tensor_map_splits_logical_bytes_across_tensor_boundaries() {
    let plan = build_allocation_plan(&mapping_tensors(), AllocationMode::Standard);
    let map = TensorMap::from_allocation_plan(&plan);

    let first = map.map_logical_byte(0).expect("byte 0 should map");
    assert_eq!(first.segments.len(), 1);
    assert_eq!(first.segments[0].tensor_name, "blk.7.ffn_down.weight");
    assert_eq!(first.segments[0].bit_offset_in_tensor, 0);
    assert_eq!(first.segments[0].bit_len, 8);

    let second = map.map_logical_byte(1).expect("byte 1 should map");
    assert_eq!(second.segments.len(), 2);
    assert_eq!(second.segments[0].tensor_name, "blk.2.ffn_up.weight");
    assert_eq!(second.segments[0].bit_offset_in_tensor, 0);
    assert_eq!(second.segments[0].bit_len, 4);
    assert_eq!(second.segments[1].tensor_name, "blk.1.attn_q.weight");
    assert_eq!(second.segments[1].bit_offset_in_tensor, 0);
    assert_eq!(second.segments[1].bit_len, 4);

    assert!(map.map_logical_byte(2).is_none());
}

fn mapping_tensors() -> Vec<GgufTensorInfo> {
    vec![
        tensor("blk.2.ffn_up.weight", &[4], GGML_TYPE_Q5_K_ID),
        tensor("blk.7.ffn_down.weight", &[2], GGML_TYPE_Q8_0_ID),
        tensor("blk.1.attn_q.weight", &[4], GGML_TYPE_Q6_K_ID),
    ]
}

fn tensor(name: &str, dimensions: &[u64], raw_type_id: u32) -> GgufTensorInfo {
    GgufTensorInfo {
        name: name.to_owned(),
        dimensions: dimensions.to_vec(),
        raw_type_id,
        data_offset: 0,
    }
}
