use crate::gguf::parser::GgufTensorInfo;
use crate::gguf::quant::GgufQuantType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationMode {
    Standard,
    Lobotomy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorRole {
    Ffn,
    Attention,
    Embedding,
    Output,
    LayerNorm,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    IneligibleInStandardMode,
    UnsupportedTensorRole,
    UnsupportedQuantType,
    NoStealableBits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum AllocationTier {
    FfnQ80 = 1,
    FfnFloat = 2,
    FfnQ6K = 3,
    FfnQ5K = 4,
    AttentionQ80 = 5,
    AttentionFloat = 6,
    AttentionQ6K = 7,
    AttentionQ5K = 8,
    FfnQ4K = 9,
    AttentionQ4K = 10,
    FfnQ3K = 11,
    AttentionQ3K = 12,
    Embedding = 13,
    Output = 14,
    LayerNorm = 15,
}

impl AllocationTier {
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedTensor {
    pub name: String,
    pub role: TensorRole,
    pub quant_type: GgufQuantType,
    pub tier: AllocationTier,
    pub layer_index: Option<u32>,
    pub weight_count: u64,
    pub stealable_bits_per_weight: usize,
    pub capacity_bits: u64,
    pub capacity_bytes_floor: u64,
    pub data_offset: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedTensor {
    pub name: String,
    pub role: TensorRole,
    pub raw_type_id: u32,
    pub reason: SkipReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocationPlan {
    pub mode: AllocationMode,
    pub tensors: Vec<PlannedTensor>,
    pub skipped: Vec<SkippedTensor>,
    pub total_capacity_bits: u64,
    pub total_capacity_bytes: u64,
}

pub fn build_allocation_plan(tensors: &[GgufTensorInfo], mode: AllocationMode) -> AllocationPlan {
    let mut planned = Vec::new();
    let mut skipped = Vec::new();
    let mut total_capacity_bits = 0_u64;

    for tensor in tensors {
        let role = classify_tensor_role(&tensor.name);
        let Some(quant_type) = GgufQuantType::from_raw_ggml_type(tensor.raw_type_id) else {
            skipped.push(SkippedTensor {
                name: tensor.name.clone(),
                role,
                raw_type_id: tensor.raw_type_id,
                reason: SkipReason::UnsupportedQuantType,
            });
            continue;
        };

        let stealable_bits_per_weight = quant_type.stealable_bits_hint();
        if stealable_bits_per_weight == 0 {
            skipped.push(SkippedTensor {
                name: tensor.name.clone(),
                role,
                raw_type_id: tensor.raw_type_id,
                reason: SkipReason::NoStealableBits,
            });
            continue;
        }

        let Some(tier) = resolve_tier(role, quant_type, mode) else {
            skipped.push(SkippedTensor {
                name: tensor.name.clone(),
                role,
                raw_type_id: tensor.raw_type_id,
                reason: skip_reason(role, mode),
            });
            continue;
        };

        let weight_count = tensor.element_count();
        let capacity_bits = weight_count.saturating_mul(stealable_bits_per_weight as u64);
        let capacity_bytes_floor = capacity_bits / 8;

        total_capacity_bits = total_capacity_bits.saturating_add(capacity_bits);

        planned.push(PlannedTensor {
            name: tensor.name.clone(),
            role,
            quant_type,
            tier,
            layer_index: extract_layer_index(&tensor.name),
            weight_count,
            stealable_bits_per_weight,
            capacity_bits,
            capacity_bytes_floor,
            data_offset: tensor.data_offset,
        });
    }

    planned.sort_by(|left, right| {
        left.tier
            .cmp(&right.tier)
            .then_with(|| right.layer_index.cmp(&left.layer_index))
            .then_with(|| quant_rank(right.quant_type).cmp(&quant_rank(left.quant_type)))
            .then_with(|| left.name.cmp(&right.name))
    });

    AllocationPlan {
        mode,
        tensors: planned,
        skipped,
        total_capacity_bits,
        total_capacity_bytes: total_capacity_bits / 8,
    }
}

pub fn classify_tensor_role(name: &str) -> TensorRole {
    if name == "token_embd.weight" {
        TensorRole::Embedding
    } else if name == "output.weight" {
        TensorRole::Output
    } else if name.ends_with(".attn_norm.weight") || name.ends_with(".ffn_norm.weight") {
        TensorRole::LayerNorm
    } else if name.ends_with(".ffn_gate.weight")
        || name.ends_with(".ffn_up.weight")
        || name.ends_with(".ffn_down.weight")
    {
        TensorRole::Ffn
    } else if name.ends_with(".attn_q.weight")
        || name.ends_with(".attn_k.weight")
        || name.ends_with(".attn_v.weight")
        || name.ends_with(".attn_output.weight")
    {
        TensorRole::Attention
    } else {
        TensorRole::Unknown
    }
}

pub fn extract_layer_index(name: &str) -> Option<u32> {
    let suffix = name.strip_prefix("blk.")?;
    let digits = suffix.split_once('.')?.0;
    digits.parse().ok()
}

fn resolve_tier(
    role: TensorRole,
    quant_type: GgufQuantType,
    mode: AllocationMode,
) -> Option<AllocationTier> {
    match (role, quant_type, mode) {
        (TensorRole::Ffn, GgufQuantType::Q8_0, _) => Some(AllocationTier::FfnQ80),
        (TensorRole::Ffn, GgufQuantType::F16 | GgufQuantType::F32, _) => {
            Some(AllocationTier::FfnFloat)
        }
        (TensorRole::Ffn, GgufQuantType::Q6K, _) => Some(AllocationTier::FfnQ6K),
        (TensorRole::Ffn, GgufQuantType::Q5K, _) => Some(AllocationTier::FfnQ5K),
        (TensorRole::Attention, GgufQuantType::Q8_0, _) => Some(AllocationTier::AttentionQ80),
        (TensorRole::Attention, GgufQuantType::F16 | GgufQuantType::F32, _) => {
            Some(AllocationTier::AttentionFloat)
        }
        (TensorRole::Attention, GgufQuantType::Q6K, _) => Some(AllocationTier::AttentionQ6K),
        (TensorRole::Attention, GgufQuantType::Q5K, _) => Some(AllocationTier::AttentionQ5K),
        (TensorRole::Ffn, GgufQuantType::Q4K, _) => Some(AllocationTier::FfnQ4K),
        (TensorRole::Attention, GgufQuantType::Q4K, _) => Some(AllocationTier::AttentionQ4K),
        (TensorRole::Ffn, GgufQuantType::Q3K, _) => Some(AllocationTier::FfnQ3K),
        (TensorRole::Attention, GgufQuantType::Q3K, _) => Some(AllocationTier::AttentionQ3K),
        (TensorRole::Embedding, _, AllocationMode::Lobotomy) => Some(AllocationTier::Embedding),
        (TensorRole::Output, _, AllocationMode::Lobotomy) => Some(AllocationTier::Output),
        (TensorRole::LayerNorm, _, AllocationMode::Lobotomy) => Some(AllocationTier::LayerNorm),
        _ => None,
    }
}

fn skip_reason(role: TensorRole, mode: AllocationMode) -> SkipReason {
    match (role, mode) {
        (TensorRole::Unknown, _) => SkipReason::UnsupportedTensorRole,
        (
            TensorRole::Embedding | TensorRole::Output | TensorRole::LayerNorm,
            AllocationMode::Standard,
        ) => SkipReason::IneligibleInStandardMode,
        _ => SkipReason::UnsupportedTensorRole,
    }
}

fn quant_rank(quant_type: GgufQuantType) -> u8 {
    match quant_type {
        GgufQuantType::F32 => 7,
        GgufQuantType::F16 => 6,
        GgufQuantType::Q8_0 => 5,
        GgufQuantType::Q6K => 4,
        GgufQuantType::Q5K => 3,
        GgufQuantType::Q4K => 2,
        GgufQuantType::Q3K => 1,
        GgufQuantType::Q2K
        | GgufQuantType::Q4_0
        | GgufQuantType::Q4_1
        | GgufQuantType::Q5_0
        | GgufQuantType::Q5_1
        | GgufQuantType::Q8_1
        | GgufQuantType::Q8K => 0,
    }
}
