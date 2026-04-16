use crate::gguf::parser::GgufTensorInfo;
use crate::gguf::quant::GgufQuantType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationMode {
    Standard,
    Lobotomy,
}

impl AllocationMode {
    pub fn is_lobotomy(self) -> bool {
        matches!(self, Self::Lobotomy)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    IneligibleInStandardMode,
    UnsupportedTensorRole,
    UnsupportedQuantType,
    NoStealableBits,
}

/// Allocation tier per DESIGN-NEW §5. V1 uses exactly four tiers ordered
/// by sensitivity: Tier1 (FFN, most robust) fills first, then Tier2
/// (attention projections), then Lobotomy (embeddings, norms, LM head —
/// only eligible when lobotomy mode is on). Skip is the catch-all for
/// anything the planner does not write into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum TensorTier {
    Tier1 = 1,
    Tier2 = 2,
    Lobotomy = 3,
    Skip = 255,
}

impl TensorTier {
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedTensor {
    pub name: String,
    pub quant_type: GgufQuantType,
    pub tier: TensorTier,
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
    pub tier: TensorTier,
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
    let lobotomy = mode.is_lobotomy();
    let mut planned = Vec::new();
    let mut skipped = Vec::new();
    let mut total_capacity_bits = 0_u64;

    for tensor in tensors {
        let tier = classify_tensor(&tensor.name, lobotomy);

        let Some(quant_type) = GgufQuantType::from_raw_ggml_type(tensor.raw_type_id) else {
            skipped.push(SkippedTensor {
                name: tensor.name.clone(),
                tier,
                raw_type_id: tensor.raw_type_id,
                reason: SkipReason::UnsupportedQuantType,
            });
            continue;
        };

        let stealable_bits_per_weight = quant_type.stealable_bits_hint();
        if stealable_bits_per_weight == 0 {
            skipped.push(SkippedTensor {
                name: tensor.name.clone(),
                tier,
                raw_type_id: tensor.raw_type_id,
                reason: SkipReason::NoStealableBits,
            });
            continue;
        }

        if matches!(tier, TensorTier::Skip) {
            skipped.push(SkippedTensor {
                name: tensor.name.clone(),
                tier,
                raw_type_id: tensor.raw_type_id,
                reason: skip_reason_for(&tensor.name, mode),
            });
            continue;
        }

        let weight_count = tensor.element_count();
        let capacity_bits = weight_count.saturating_mul(stealable_bits_per_weight as u64);
        let capacity_bytes_floor = capacity_bits / 8;

        total_capacity_bits = total_capacity_bits.saturating_add(capacity_bits);

        planned.push(PlannedTensor {
            name: tensor.name.clone(),
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

/// Classifier per DESIGN-NEW §5. Returns the `TensorTier` that this tensor
/// should land in given the current lobotomy flag.
///
/// The §5 pseudocode checks the LM-head / embedding pattern first, but that
/// ordering is buggy: `attn_output.weight` contains `output.weight` as a
/// substring, so attention output projections would be misclassified as the
/// LM head and skipped. We check attention patterns first to preserve the
/// spec's spirit — FFN to Tier1, attention projections to Tier2, LM head /
/// embeddings / norms to Skip-or-Lobotomy.
pub fn classify_tensor(name: &str, lobotomy: bool) -> TensorTier {
    let lobotomy_or_skip = if lobotomy {
        TensorTier::Lobotomy
    } else {
        TensorTier::Skip
    };

    if name.contains("attn_q")
        || name.contains("attn_k")
        || name.contains("attn_v")
        || name.contains("attn_output")
    {
        TensorTier::Tier2
    } else if name.contains("ffn_gate") || name.contains("ffn_up") || name.contains("ffn_down") {
        TensorTier::Tier1
    } else if name.contains("token_embd")
        || name.contains("output.weight")
        || name.contains("_norm")
    {
        lobotomy_or_skip
    } else {
        TensorTier::Skip
    }
}

pub fn extract_layer_index(name: &str) -> Option<u32> {
    let suffix = name.strip_prefix("blk.")?;
    let digits = suffix.split_once('.')?.0;
    digits.parse().ok()
}

fn skip_reason_for(name: &str, mode: AllocationMode) -> SkipReason {
    // In standard mode, embeddings / norms / output are rejected because of the
    // mode, not because the tensor is unrecognized. Everything else without a
    // matched tier is an unknown tensor pattern.
    if matches!(mode, AllocationMode::Standard)
        && (name.contains("token_embd")
            || name.contains("output.weight")
            || name.contains("_norm"))
    {
        SkipReason::IneligibleInStandardMode
    } else {
        SkipReason::UnsupportedTensorRole
    }
}
