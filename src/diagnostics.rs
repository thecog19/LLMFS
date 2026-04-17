//! `status` reporting: gather a structured view of device state from a
//! `StegoDevice` and format it for humans.
//!
//! Kept deliberately simple for V1 — the per-tier "used bytes" figure
//! assumes sequential allocation (Tier1 fills first, then Tier2, then
//! Lobotomy), which matches the V1 allocator. A more accurate mapping
//! would walk each live file's block chain and look up the backing
//! tensor's tier; that's a Task 15 concern once the quality harness
//! needs the extra fidelity.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::fs::file_ops::FsError;
use crate::gguf::quant::GgufQuantType;
use crate::stego::device::StegoDevice;
use crate::stego::integrity::decode_quant_profile;
use crate::stego::planner::TensorTier;

/// Retained so `bootstrap_smoke` keeps working. Real status info comes
/// from `gather(&StegoDevice)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticsBootstrap {
    pub tracks_tiers: bool,
}

impl Default for DiagnosticsBootstrap {
    fn default() -> Self {
        Self { tracks_tiers: true }
    }
}

#[derive(Debug, Clone)]
pub struct DeviceStatus {
    pub total_blocks: u32,
    pub used_blocks: u32,
    pub free_blocks: u32,
    pub utilization_pct: f32,
    pub file_count: u32,
    pub total_stored_bytes: u64,
    pub tier_utilization: BTreeMap<TensorTier, TierUsage>,
    pub quant_profile: Vec<GgufQuantType>,
    pub lobotomy: bool,
    pub dirty_on_open: bool,
    pub estimated_perplexity_impact: PerplexityImpact,
}

#[derive(Debug, Clone, Default)]
pub struct TierUsage {
    /// Total stego capacity contributed by tensors in this tier.
    pub capacity_bytes: u64,
    /// Approximate bytes currently occupied in this tier — assumes
    /// sequential allocation order Tier1 → Tier2 → Lobotomy.
    pub used_bytes: u64,
    pub tensor_count: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct PerplexityImpact {
    pub score: f32,
    pub bucket: PerplexityBucket,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerplexityBucket {
    Negligible,
    Low,
    Moderate,
    Severe,
}

impl PerplexityBucket {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Negligible => "negligible",
            Self::Low => "low",
            Self::Moderate => "moderate",
            Self::Severe => "severe",
        }
    }
}

pub fn gather(device: &StegoDevice) -> Result<DeviceStatus, FsError> {
    let sb = device.superblock().clone();
    let total_blocks = device.total_blocks();
    let free_blocks = device.free_blocks()?;
    let used_blocks = total_blocks.saturating_sub(free_blocks);
    let utilization_pct = if total_blocks == 0 {
        0.0
    } else {
        (used_blocks as f32 / total_blocks as f32) * 100.0
    };

    let files = device.list_files()?;
    let file_count = files.len() as u32;
    let total_stored_bytes = files.iter().map(|e| e.size_bytes).sum();

    let plan = device.allocation_plan();
    let mut tier_capacity: BTreeMap<TensorTier, u64> = BTreeMap::new();
    let mut tier_count: BTreeMap<TensorTier, u32> = BTreeMap::new();
    for t in &plan.tensors {
        *tier_capacity.entry(t.tier).or_insert(0) += t.capacity_bytes_floor;
        *tier_count.entry(t.tier).or_insert(0) += 1;
    }

    // Assume V1's sequential fill order (Tier1 → Tier2 → Lobotomy) so we
    // can approximate per-tier used_bytes without walking file block
    // chains. When V2's sensitivity-ordered allocator lands this falls
    // back to the same "first-to-be-allocated" distribution.
    let used_bytes = used_blocks as u64 * crate::BLOCK_SIZE as u64;
    let mut tier_utilization = BTreeMap::new();
    let mut remaining = used_bytes;
    for tier in [TensorTier::Tier1, TensorTier::Tier2, TensorTier::Lobotomy] {
        let capacity = tier_capacity.get(&tier).copied().unwrap_or(0);
        let count = tier_count.get(&tier).copied().unwrap_or(0);
        if capacity == 0 && count == 0 {
            continue;
        }
        let used = remaining.min(capacity);
        remaining = remaining.saturating_sub(used);
        tier_utilization.insert(
            tier,
            TierUsage {
                capacity_bytes: capacity,
                used_bytes: used,
                tensor_count: count,
            },
        );
    }

    let estimated_perplexity_impact = score_perplexity_impact(&tier_utilization);

    Ok(DeviceStatus {
        total_blocks,
        used_blocks,
        free_blocks,
        utilization_pct,
        file_count,
        total_stored_bytes,
        tier_utilization,
        quant_profile: decode_quant_profile(sb.fields.quant_profile),
        lobotomy: sb.is_lobotomy(),
        dirty_on_open: device.was_dirty_on_open(),
        estimated_perplexity_impact,
    })
}

fn score_perplexity_impact(tiers: &BTreeMap<TensorTier, TierUsage>) -> PerplexityImpact {
    let weight = |t: TensorTier| match t {
        TensorTier::Tier1 => 0.5,
        TensorTier::Tier2 => 1.0,
        TensorTier::Lobotomy => 5.0,
        TensorTier::Skip => 0.0,
    };
    let mut score = 0.0_f32;
    for (tier, usage) in tiers {
        if usage.capacity_bytes == 0 {
            continue;
        }
        let ratio = usage.used_bytes as f32 / usage.capacity_bytes as f32;
        score += ratio * weight(*tier);
    }
    let bucket = if score < 0.01 {
        PerplexityBucket::Negligible
    } else if score < 0.25 {
        PerplexityBucket::Low
    } else if score < 1.0 {
        PerplexityBucket::Moderate
    } else {
        PerplexityBucket::Severe
    };
    PerplexityImpact { score, bucket }
}

pub fn format_human(status: &DeviceStatus) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "total:       {} blocks", status.total_blocks);
    let _ = writeln!(out, "used:        {} blocks", status.used_blocks);
    let _ = writeln!(out, "free:        {} blocks", status.free_blocks);
    let _ = writeln!(out, "utilization: {:.1}%", status.utilization_pct);
    let _ = writeln!(out, "files:       {}", status.file_count);
    let _ = writeln!(out, "stored:      {} bytes", status.total_stored_bytes);
    let _ = writeln!(out, "quant:       {:?}", status.quant_profile);
    let _ = writeln!(
        out,
        "lobotomy:    {}",
        if status.lobotomy { "yes" } else { "no" }
    );
    let _ = writeln!(
        out,
        "dirty:       {}",
        if status.dirty_on_open {
            "yes (recovered on open)"
        } else {
            "no"
        }
    );
    out.push('\n');
    let _ = writeln!(out, "per-tier breakdown:");
    for (tier, usage) in &status.tier_utilization {
        let pct = if usage.capacity_bytes == 0 {
            0.0
        } else {
            (usage.used_bytes as f32 / usage.capacity_bytes as f32) * 100.0
        };
        let _ = writeln!(
            out,
            "  {:<9}  {:>4} tensors   {:>10} / {:>10} B   ({:.1}%)",
            format!("{:?}", tier),
            usage.tensor_count,
            usage.used_bytes,
            usage.capacity_bytes,
            pct
        );
    }
    out.push('\n');
    let _ = writeln!(
        out,
        "est. perplexity impact: {} (score {:.3})",
        status.estimated_perplexity_impact.bucket.as_str(),
        status.estimated_perplexity_impact.score
    );
    out
}
