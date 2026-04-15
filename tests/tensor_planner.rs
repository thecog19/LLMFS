use llmdb::gguf::parser::GgufTensorInfo;
use llmdb::gguf::quant::{
    GGML_TYPE_F16_ID, GGML_TYPE_F32_ID, GGML_TYPE_Q2_K_ID, GGML_TYPE_Q3_K_ID, GGML_TYPE_Q4_0_ID,
    GGML_TYPE_Q4_1_ID, GGML_TYPE_Q4_K_ID, GGML_TYPE_Q5_0_ID, GGML_TYPE_Q5_1_ID, GGML_TYPE_Q5_K_ID,
    GGML_TYPE_Q6_K_ID, GGML_TYPE_Q8_0_ID, GGML_TYPE_Q8_1_ID, GGML_TYPE_Q8_K_ID, GgufQuantType,
};
use llmdb::stego::planner::{
    AllocationMode, AllocationTier, SkipReason, TensorRole, build_allocation_plan,
    classify_tensor_role, extract_layer_index,
};

#[test]
fn raw_ggml_type_ids_map_to_supported_quant_types() {
    assert_eq!(
        GgufQuantType::from_raw_ggml_type(GGML_TYPE_F32_ID),
        Some(GgufQuantType::F32)
    );
    assert_eq!(
        GgufQuantType::from_raw_ggml_type(GGML_TYPE_F16_ID),
        Some(GgufQuantType::F16)
    );
    assert_eq!(
        GgufQuantType::from_raw_ggml_type(GGML_TYPE_Q8_0_ID),
        Some(GgufQuantType::Q8_0)
    );
    assert_eq!(
        GgufQuantType::from_raw_ggml_type(GGML_TYPE_Q6_K_ID),
        Some(GgufQuantType::Q6K)
    );
    assert_eq!(
        GgufQuantType::from_raw_ggml_type(GGML_TYPE_Q5_K_ID),
        Some(GgufQuantType::Q5K)
    );
    assert_eq!(
        GgufQuantType::from_raw_ggml_type(GGML_TYPE_Q4_K_ID),
        Some(GgufQuantType::Q4K)
    );
    assert_eq!(
        GgufQuantType::from_raw_ggml_type(GGML_TYPE_Q3_K_ID),
        Some(GgufQuantType::Q3K)
    );
    assert_eq!(
        GgufQuantType::from_raw_ggml_type(GGML_TYPE_Q2_K_ID),
        Some(GgufQuantType::Q2K)
    );
    assert_eq!(
        GgufQuantType::from_raw_ggml_type(GGML_TYPE_Q4_0_ID),
        Some(GgufQuantType::Q4_0)
    );
    assert_eq!(
        GgufQuantType::from_raw_ggml_type(GGML_TYPE_Q4_1_ID),
        Some(GgufQuantType::Q4_1)
    );
    assert_eq!(
        GgufQuantType::from_raw_ggml_type(GGML_TYPE_Q5_0_ID),
        Some(GgufQuantType::Q5_0)
    );
    assert_eq!(
        GgufQuantType::from_raw_ggml_type(GGML_TYPE_Q5_1_ID),
        Some(GgufQuantType::Q5_1)
    );
    assert_eq!(
        GgufQuantType::from_raw_ggml_type(GGML_TYPE_Q8_1_ID),
        Some(GgufQuantType::Q8_1)
    );
    assert_eq!(
        GgufQuantType::from_raw_ggml_type(GGML_TYPE_Q8_K_ID),
        Some(GgufQuantType::Q8K)
    );
    assert_eq!(GgufQuantType::from_raw_ggml_type(4), None);
    assert_eq!(GgufQuantType::from_raw_ggml_type(5), None);
    assert_eq!(GgufQuantType::from_raw_ggml_type(999), None);
}

#[test]
fn unsupported_stego_quant_types_report_zero_stealable_bits() {
    for quant in [
        GgufQuantType::Q2K,
        GgufQuantType::Q4_0,
        GgufQuantType::Q4_1,
        GgufQuantType::Q5_0,
        GgufQuantType::Q5_1,
        GgufQuantType::Q8_1,
        GgufQuantType::Q8K,
    ] {
        assert_eq!(
            quant.stealable_bits_hint(),
            0,
            "{quant:?} must not be stego-eligible in V1"
        );
    }
}

#[test]
fn stego_eligible_quant_types_report_nonzero_stealable_bits() {
    for quant in [
        GgufQuantType::Q8_0,
        GgufQuantType::Q6K,
        GgufQuantType::Q5K,
        GgufQuantType::Q4K,
        GgufQuantType::Q3K,
        GgufQuantType::F16,
        GgufQuantType::F32,
    ] {
        assert!(
            quant.stealable_bits_hint() > 0,
            "{quant:?} must remain stego-eligible in V1"
        );
    }
}

#[test]
fn planner_skips_newly_declared_unsupported_variants_as_no_stealable_bits() {
    let tensors = vec![
        tensor("blk.0.ffn_down.weight", &[32], GGML_TYPE_Q4_0_ID),
        tensor("blk.0.ffn_up.weight", &[32], GGML_TYPE_Q4_1_ID),
        tensor("blk.0.ffn_gate.weight", &[32], GGML_TYPE_Q5_0_ID),
        tensor("blk.1.attn_q.weight", &[32], GGML_TYPE_Q5_1_ID),
        tensor("blk.1.attn_k.weight", &[32], GGML_TYPE_Q8_1_ID),
        tensor("blk.1.attn_v.weight", &[32], GGML_TYPE_Q8_K_ID),
    ];

    let plan = build_allocation_plan(&tensors, AllocationMode::Standard);

    assert!(
        plan.tensors.is_empty(),
        "no new variant should be eligible for stego"
    );
    let reasons: Vec<_> = plan
        .skipped
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor.reason))
        .collect();
    assert_eq!(
        reasons,
        vec![
            ("blk.0.ffn_down.weight", SkipReason::NoStealableBits),
            ("blk.0.ffn_up.weight", SkipReason::NoStealableBits),
            ("blk.0.ffn_gate.weight", SkipReason::NoStealableBits),
            ("blk.1.attn_q.weight", SkipReason::NoStealableBits),
            ("blk.1.attn_k.weight", SkipReason::NoStealableBits),
            ("blk.1.attn_v.weight", SkipReason::NoStealableBits),
        ]
    );
}

#[test]
fn classifies_tensor_roles_and_layer_indices() {
    assert_eq!(
        classify_tensor_role("token_embd.weight"),
        TensorRole::Embedding
    );
    assert_eq!(classify_tensor_role("output.weight"), TensorRole::Output);
    assert_eq!(
        classify_tensor_role("blk.7.ffn_norm.weight"),
        TensorRole::LayerNorm
    );
    assert_eq!(
        classify_tensor_role("blk.12.ffn_up.weight"),
        TensorRole::Ffn
    );
    assert_eq!(
        classify_tensor_role("blk.30.attn_output.weight"),
        TensorRole::Attention
    );
    assert_eq!(
        classify_tensor_role("blk.2.weird.weight"),
        TensorRole::Unknown
    );

    assert_eq!(extract_layer_index("blk.31.ffn_down.weight"), Some(31));
    assert_eq!(extract_layer_index("token_embd.weight"), None);
}

#[test]
fn standard_mode_orders_eligible_tensors_by_tier_then_deepest_layer() {
    let plan = build_allocation_plan(&sample_tensors(), AllocationMode::Standard);

    let names: Vec<_> = plan
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect();
    assert_eq!(
        names,
        vec![
            "blk.31.ffn_down.weight",
            "blk.12.ffn_up.weight",
            "blk.1.ffn_down.weight",
            "blk.9.attn_q.weight",
            "blk.3.attn_v.weight",
            "blk.30.attn_output.weight",
            "blk.10.ffn_gate.weight",
            "blk.8.attn_v.weight",
        ]
    );

    let tiers: Vec<_> = plan
        .tensors
        .iter()
        .map(|tensor| tensor.tier.as_u8())
        .collect();
    assert_eq!(tiers, vec![1, 1, 3, 5, 5, 8, 9, 12]);

    assert_eq!(plan.total_capacity_bits, 3696);
    assert_eq!(plan.total_capacity_bytes, 462);

    let first = &plan.tensors[0];
    assert_eq!(first.tier, AllocationTier::FfnQ80);
    assert_eq!(first.capacity_bits, 2048);
    assert_eq!(first.capacity_bytes_floor, 256);
    assert_eq!(first.quant_type, GgufQuantType::Q8_0);

    let skipped: Vec<_> = plan
        .skipped
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor.reason))
        .collect();
    assert_eq!(
        skipped,
        vec![
            ("token_embd.weight", SkipReason::IneligibleInStandardMode),
            ("output.weight", SkipReason::IneligibleInStandardMode),
            (
                "blk.7.ffn_norm.weight",
                SkipReason::IneligibleInStandardMode
            ),
            ("blk.2.misc.weight", SkipReason::UnsupportedTensorRole),
            ("blk.11.ffn_up.weight", SkipReason::NoStealableBits),
        ]
    );
}

#[test]
fn lobotomy_mode_appends_embedding_output_and_layernorm_tiers() {
    let plan = build_allocation_plan(&sample_tensors(), AllocationMode::Lobotomy);

    let trailing: Vec<_> = plan
        .tensors
        .iter()
        .rev()
        .take(3)
        .map(|tensor| (tensor.name.as_str(), tensor.tier.as_u8()))
        .collect();
    assert_eq!(
        trailing,
        vec![
            ("blk.7.ffn_norm.weight", 15),
            ("output.weight", 14),
            ("token_embd.weight", 13),
        ]
    );

    assert_eq!(plan.total_capacity_bits, 4592);
    assert_eq!(plan.total_capacity_bytes, 574);

    let skipped: Vec<_> = plan
        .skipped
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor.reason))
        .collect();
    assert_eq!(
        skipped,
        vec![
            ("blk.2.misc.weight", SkipReason::UnsupportedTensorRole),
            ("blk.11.ffn_up.weight", SkipReason::NoStealableBits),
        ]
    );
}

fn sample_tensors() -> Vec<GgufTensorInfo> {
    vec![
        tensor("blk.31.ffn_down.weight", &[512], GGML_TYPE_Q8_0_ID),
        tensor("blk.12.ffn_up.weight", &[128], GGML_TYPE_Q8_0_ID),
        tensor("blk.1.ffn_down.weight", &[256], GGML_TYPE_Q6_K_ID),
        tensor("blk.9.attn_q.weight", &[64], GGML_TYPE_Q8_0_ID),
        tensor("blk.3.attn_v.weight", &[32], GGML_TYPE_Q8_0_ID),
        tensor("blk.30.attn_output.weight", &[64], GGML_TYPE_Q5_K_ID),
        tensor("blk.10.ffn_gate.weight", &[80], GGML_TYPE_Q4_K_ID),
        tensor("blk.8.attn_v.weight", &[96], GGML_TYPE_Q3_K_ID),
        tensor("token_embd.weight", &[128], GGML_TYPE_F16_ID),
        tensor("output.weight", &[32], GGML_TYPE_F32_ID),
        tensor("blk.7.ffn_norm.weight", &[32], GGML_TYPE_F16_ID),
        tensor("blk.2.misc.weight", &[48], GGML_TYPE_Q8_0_ID),
        tensor("blk.11.ffn_up.weight", &[128], GGML_TYPE_Q2_K_ID),
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
