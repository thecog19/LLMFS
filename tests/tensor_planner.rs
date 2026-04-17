use llmdb::gguf::parser::GgufTensorInfo;
use llmdb::gguf::quant::{
    GGML_TYPE_F16_ID, GGML_TYPE_F32_ID, GGML_TYPE_Q2_K_ID, GGML_TYPE_Q3_K_ID, GGML_TYPE_Q4_0_ID,
    GGML_TYPE_Q4_1_ID, GGML_TYPE_Q4_K_ID, GGML_TYPE_Q5_0_ID, GGML_TYPE_Q5_1_ID, GGML_TYPE_Q5_K_ID,
    GGML_TYPE_Q6_K_ID, GGML_TYPE_Q8_0_ID, GGML_TYPE_Q8_1_ID, GGML_TYPE_Q8_K_ID, GgufQuantType,
};
use llmdb::stego::planner::{
    AllocationMode, SkipReason, TensorTier, build_allocation_plan, classify_tensor,
    extract_layer_index,
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
fn classify_tensor_matches_design_new_pseudocode() {
    // FFN → Tier1 regardless of lobotomy
    for name in [
        "blk.0.ffn_gate.weight",
        "blk.7.ffn_up.weight",
        "blk.31.ffn_down.weight",
    ] {
        assert_eq!(classify_tensor(name, false), TensorTier::Tier1);
        assert_eq!(classify_tensor(name, true), TensorTier::Tier1);
    }

    // Attention → Tier2 regardless of lobotomy
    for name in [
        "blk.0.attn_q.weight",
        "blk.1.attn_k.weight",
        "blk.2.attn_v.weight",
        "blk.3.attn_output.weight",
    ] {
        assert_eq!(classify_tensor(name, false), TensorTier::Tier2);
        assert_eq!(classify_tensor(name, true), TensorTier::Tier2);
    }

    // Embeddings/output → Skip in standard, Lobotomy when enabled
    for name in ["token_embd.weight", "output.weight"] {
        assert_eq!(classify_tensor(name, false), TensorTier::Skip);
        assert_eq!(classify_tensor(name, true), TensorTier::Lobotomy);
    }

    // Norm tensors → Skip in standard, Lobotomy when enabled
    for name in [
        "blk.5.attn_norm.weight",
        "blk.12.ffn_norm.weight",
        "output_norm.weight",
    ] {
        assert_eq!(classify_tensor(name, false), TensorTier::Skip);
        assert_eq!(classify_tensor(name, true), TensorTier::Lobotomy);
    }

    // Anything else → Skip regardless of lobotomy
    for name in ["blk.2.weird.weight", "unknown.weight"] {
        assert_eq!(classify_tensor(name, false), TensorTier::Skip);
        assert_eq!(classify_tensor(name, true), TensorTier::Skip);
    }

    // Layer index extraction unchanged
    assert_eq!(extract_layer_index("blk.31.ffn_down.weight"), Some(31));
    assert_eq!(extract_layer_index("token_embd.weight"), None);
}

#[test]
fn ffn_q4k_lands_in_tier1_not_a_quant_specific_tier() {
    let plan = build_allocation_plan(
        &[tensor("blk.5.ffn_down.weight", &[256], GGML_TYPE_Q4_K_ID)],
        AllocationMode::Standard,
    );
    assert_eq!(plan.tensors.len(), 1);
    assert_eq!(plan.tensors[0].tier, TensorTier::Tier1);
    assert_eq!(plan.tensors[0].quant_type, GgufQuantType::Q4K);
}

#[test]
fn standard_mode_orders_tier1_then_tier2_with_layer_desc_tiebreak() {
    let plan = build_allocation_plan(&sample_tensors(), AllocationMode::Standard);

    let names: Vec<_> = plan
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect();
    assert_eq!(
        names,
        vec![
            // Tier1 (FFN) sorted by layer descending, tie-broken by name
            "blk.31.ffn_down.weight",
            "blk.12.ffn_up.weight",
            "blk.10.ffn_gate.weight",
            "blk.1.ffn_down.weight",
            // Tier2 (attention) sorted by layer descending
            "blk.30.attn_output.weight",
            "blk.9.attn_q.weight",
            "blk.8.attn_v.weight",
            "blk.3.attn_v.weight",
        ]
    );

    let tiers: Vec<_> = plan.tensors.iter().map(|tensor| tensor.tier).collect();
    assert_eq!(
        tiers,
        vec![
            TensorTier::Tier1,
            TensorTier::Tier1,
            TensorTier::Tier1,
            TensorTier::Tier1,
            TensorTier::Tier2,
            TensorTier::Tier2,
            TensorTier::Tier2,
            TensorTier::Tier2,
        ]
    );

    // Standard mode skip list contains embedding, output, norms, unknown, and zero-bits.
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
fn lobotomy_mode_appends_lobotomy_tier_after_tier2() {
    let plan = build_allocation_plan(&sample_tensors(), AllocationMode::Lobotomy);

    // Lobotomy tier sort: layer desc (Some > None), then name asc. ffn_norm
    // has layer Some(7); output and token_embd both have layer None, so they
    // tie-break by name ("output" < "token_embd").
    let tail: Vec<_> = plan
        .tensors
        .iter()
        .rev()
        .take(3)
        .map(|tensor| (tensor.name.as_str(), tensor.tier))
        .collect();
    assert_eq!(
        tail,
        vec![
            ("token_embd.weight", TensorTier::Lobotomy),
            ("output.weight", TensorTier::Lobotomy),
            ("blk.7.ffn_norm.weight", TensorTier::Lobotomy),
        ]
    );

    // Tier1 and Tier2 still appear first, in the same order as standard mode.
    let prefix: Vec<_> = plan
        .tensors
        .iter()
        .take(8)
        .map(|tensor| tensor.tier)
        .collect();
    assert_eq!(
        prefix,
        vec![
            TensorTier::Tier1,
            TensorTier::Tier1,
            TensorTier::Tier1,
            TensorTier::Tier1,
            TensorTier::Tier2,
            TensorTier::Tier2,
            TensorTier::Tier2,
            TensorTier::Tier2,
        ]
    );

    // Lobotomy skip list drops IneligibleInStandardMode reasons.
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
