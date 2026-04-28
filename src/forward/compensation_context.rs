//! Phase E write-to-compensation mapping.
//!
//! The math layer in [`crate::forward::compensation`] works in
//! activation-Hessian coordinates: `(ActivationSite, layer,
//! input_channel)`. The V2 filesystem writes to cover coordinates:
//! `(slot, weight_index, bit_index)`. This module is the narrow
//! bridge between those coordinate systems. It deliberately does not
//! apply compensation yet; it only identifies which Hessian row/column
//! a concrete written weight belongs to.

use thiserror::Error;

use crate::forward::awq::{ActivationSite, tensor_site_for_name};
use crate::gguf::parser::GgufTensorInfo;
use crate::stego::tensor_map::TensorMap;
use crate::v2::pointer::Pointer;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompensationTarget {
    pub slot_index: u16,
    pub tensor_name: String,
    pub site: ActivationSite,
    pub layer: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub input_channel: usize,
    pub output_channel: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompensationRegionKey {
    pub tensor_name: String,
    pub site: ActivationSite,
    pub layer: usize,
    pub output_channel: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompensationWriteRegion {
    pub key: CompensationRegionKey,
    pub input_channels: Vec<usize>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CompensationTargetError {
    #[error("slot {slot} out of range (map has {slot_count} slots)")]
    SlotOutOfRange { slot: u16, slot_count: usize },

    #[error("weight {weight_index} out of range for slot {slot} with {weight_count} weights")]
    WeightOutOfRange {
        slot: u16,
        weight_index: u64,
        weight_count: u64,
    },

    #[error("pointer targets slot {slot}, which has no stealable bits")]
    NonStealableSlot { slot: u16 },

    #[error(
        "pointer range [{start_weight}, {end_weight}) lies outside slot {slot} with {weight_count} weights"
    )]
    PointerOutOfBounds {
        slot: u16,
        start_weight: u32,
        end_weight: u64,
        weight_count: u64,
    },

    #[error("GGUF tensor info for '{tensor}' is missing")]
    MissingTensorInfo { tensor: String },

    #[error("tensor '{tensor}' has rank {rank}; compensation expects a 2-D matmul weight")]
    UnsupportedTensorRank { tensor: String, rank: usize },

    #[error("tensor '{tensor}' has zero input dimension")]
    ZeroInputDim { tensor: String },

    #[error("tensor '{tensor}' input dimension {value} does not fit usize")]
    InputDimTooLarge { tensor: String, value: u64 },

    #[error("tensor '{tensor}' output dimension {value} does not fit usize")]
    OutputDimTooLarge { tensor: String, value: u64 },

    #[error(
        "tensor '{tensor}' weight_count mismatch between TensorMap ({map_count}) and GGUF ({gguf_count})"
    )]
    WeightCountMismatch {
        tensor: String,
        map_count: u64,
        gguf_count: u64,
    },
}

pub fn target_for_weight(
    map: &TensorMap,
    gguf_tensors: &[GgufTensorInfo],
    slot_index: u16,
    weight_index: u64,
) -> Result<Option<CompensationTarget>, CompensationTargetError> {
    let slot =
        map.slots
            .get(slot_index as usize)
            .ok_or(CompensationTargetError::SlotOutOfRange {
                slot: slot_index,
                slot_count: map.slots.len(),
            })?;
    if weight_index >= slot.weight_count {
        return Err(CompensationTargetError::WeightOutOfRange {
            slot: slot_index,
            weight_index,
            weight_count: slot.weight_count,
        });
    }

    let Some((site, layer)) = tensor_site_for_name(&slot.name) else {
        return Ok(None);
    };

    let tensor = gguf_tensors
        .iter()
        .find(|t| t.name == slot.name)
        .ok_or_else(|| CompensationTargetError::MissingTensorInfo {
            tensor: slot.name.clone(),
        })?;
    if tensor.dimensions.len() != 2 {
        return Err(CompensationTargetError::UnsupportedTensorRank {
            tensor: slot.name.clone(),
            rank: tensor.dimensions.len(),
        });
    }

    let input_dim_raw = tensor.dimensions[0];
    let output_dim_raw = tensor.dimensions[1];
    if input_dim_raw == 0 {
        return Err(CompensationTargetError::ZeroInputDim {
            tensor: slot.name.clone(),
        });
    }
    let input_dim =
        usize::try_from(input_dim_raw).map_err(|_| CompensationTargetError::InputDimTooLarge {
            tensor: slot.name.clone(),
            value: input_dim_raw,
        })?;
    let output_dim = usize::try_from(output_dim_raw).map_err(|_| {
        CompensationTargetError::OutputDimTooLarge {
            tensor: slot.name.clone(),
            value: output_dim_raw,
        }
    })?;

    let gguf_count = tensor.element_count();
    if gguf_count != slot.weight_count {
        return Err(CompensationTargetError::WeightCountMismatch {
            tensor: slot.name.clone(),
            map_count: slot.weight_count,
            gguf_count,
        });
    }

    Ok(Some(CompensationTarget {
        slot_index,
        tensor_name: slot.name.clone(),
        site,
        layer,
        input_dim,
        output_dim,
        input_channel: (weight_index % input_dim_raw) as usize,
        output_channel: (weight_index / input_dim_raw) as usize,
    }))
}

pub fn regions_for_pointer(
    map: &TensorMap,
    gguf_tensors: &[GgufTensorInfo],
    pointer: Pointer,
) -> Result<Vec<CompensationWriteRegion>, CompensationTargetError> {
    if pointer.is_null() {
        return Ok(Vec::new());
    }

    let slot =
        map.slots
            .get(pointer.slot as usize)
            .ok_or(CompensationTargetError::SlotOutOfRange {
                slot: pointer.slot,
                slot_count: map.slots.len(),
            })?;
    let bpw = slot.stealable_bits_per_weight as u32;
    if bpw == 0 {
        return Err(CompensationTargetError::NonStealableSlot { slot: pointer.slot });
    }

    let covered_weights = pointer.length_in_bits.div_ceil(bpw);
    let end_weight = u64::from(pointer.start_weight) + u64::from(covered_weights);
    if end_weight > slot.weight_count {
        return Err(CompensationTargetError::PointerOutOfBounds {
            slot: pointer.slot,
            start_weight: pointer.start_weight,
            end_weight,
            weight_count: slot.weight_count,
        });
    }

    let mut regions: Vec<CompensationWriteRegion> = Vec::new();
    for weight_index in pointer.start_weight..pointer.start_weight + covered_weights {
        let Some(target) =
            target_for_weight(map, gguf_tensors, pointer.slot, u64::from(weight_index))?
        else {
            continue;
        };
        let key = CompensationRegionKey {
            tensor_name: target.tensor_name,
            site: target.site,
            layer: target.layer,
            output_channel: target.output_channel,
        };
        let region = match regions.iter_mut().find(|r| r.key == key) {
            Some(region) => region,
            None => {
                regions.push(CompensationWriteRegion {
                    key,
                    input_channels: Vec::new(),
                });
                regions.last_mut().expect("just pushed region")
            }
        };
        if !region.input_channels.contains(&target.input_channel) {
            region.input_channels.push(target.input_channel);
        }
    }
    Ok(regions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::quant::GgufQuantType;
    use crate::stego::planner::TensorTier;
    use crate::stego::tensor_map::TensorSlot;

    fn slot(name: &str, weight_count: u64) -> TensorSlot {
        TensorSlot {
            name: name.to_owned(),
            quant_type: GgufQuantType::F16,
            tier: TensorTier::Tier1,
            data_offset: 0,
            weight_count,
            stealable_bits_per_weight: 4,
            capacity_bits: weight_count * 4,
            bit_start: 0,
            bit_end: weight_count * 4,
        }
    }

    fn map(slots: Vec<TensorSlot>) -> TensorMap {
        let mut adjusted = Vec::with_capacity(slots.len());
        let mut cursor = 0;
        for mut slot in slots {
            slot.bit_start = cursor;
            cursor += slot.capacity_bits;
            slot.bit_end = cursor;
            adjusted.push(slot);
        }
        TensorMap {
            slots: adjusted,
            total_capacity_bits: cursor,
            total_capacity_bytes: cursor / 8,
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

    #[test]
    fn qkv_weight_maps_to_site_layer_row_and_input_channel() {
        let name = "blk.7.attn_q.weight";
        let map = map(vec![slot(name, 12)]);
        let tensors = vec![tensor(name, 4, 3)];

        let target = target_for_weight(&map, &tensors, 0, 6)
            .expect("target lookup")
            .expect("q tensor is compensatable");

        assert_eq!(target.slot_index, 0);
        assert_eq!(target.tensor_name, name);
        assert_eq!(target.site, ActivationSite::QkvInput);
        assert_eq!(target.layer, 7);
        assert_eq!(target.input_dim, 4);
        assert_eq!(target.output_dim, 3);
        assert_eq!(target.output_channel, 1);
        assert_eq!(target.input_channel, 2);
    }

    #[test]
    fn ffn_gate_up_and_down_use_their_own_activation_sites() {
        let names = [
            ("blk.2.ffn_gate.weight", ActivationSite::FfnGateUpInput),
            ("blk.2.ffn_up.weight", ActivationSite::FfnGateUpInput),
            ("blk.2.ffn_down.weight", ActivationSite::FfnDownInput),
        ];
        for (name, site) in names {
            let map = map(vec![slot(name, 10)]);
            let tensors = vec![tensor(name, 5, 2)];
            let target = target_for_weight(&map, &tensors, 0, 4)
                .expect("target lookup")
                .expect("ffn tensor is compensatable");
            assert_eq!(target.site, site);
            assert_eq!(target.layer, 2);
            assert_eq!(target.input_channel, 4);
            assert_eq!(target.output_channel, 0);
        }
    }

    #[test]
    fn non_transformer_linear_tensor_returns_none() {
        let name = "token_embd.weight";
        let map = map(vec![slot(name, 12)]);
        let tensors = vec![tensor(name, 4, 3)];

        let target = target_for_weight(&map, &tensors, 0, 0).expect("target lookup");

        assert_eq!(target, None);
    }

    #[test]
    fn out_of_range_slot_errors() {
        let map = map(vec![slot("blk.0.attn_q.weight", 12)]);
        let err = target_for_weight(&map, &[], 1, 0).unwrap_err();
        assert_eq!(
            err,
            CompensationTargetError::SlotOutOfRange {
                slot: 1,
                slot_count: 1,
            }
        );
    }

    #[test]
    fn out_of_range_weight_errors() {
        let name = "blk.0.attn_q.weight";
        let map = map(vec![slot(name, 12)]);
        let err = target_for_weight(&map, &[], 0, 12).unwrap_err();
        assert_eq!(
            err,
            CompensationTargetError::WeightOutOfRange {
                slot: 0,
                weight_index: 12,
                weight_count: 12,
            }
        );
    }

    #[test]
    fn missing_gguf_tensor_info_errors_for_compensatable_slot() {
        let name = "blk.0.attn_q.weight";
        let map = map(vec![slot(name, 12)]);
        let err = target_for_weight(&map, &[], 0, 0).unwrap_err();
        assert_eq!(
            err,
            CompensationTargetError::MissingTensorInfo {
                tensor: name.to_owned(),
            }
        );
    }

    #[test]
    fn mismatched_map_and_gguf_weight_counts_error() {
        let name = "blk.0.attn_q.weight";
        let map = map(vec![slot(name, 12)]);
        let tensors = vec![tensor(name, 5, 3)];
        let err = target_for_weight(&map, &tensors, 0, 0).unwrap_err();
        assert_eq!(
            err,
            CompensationTargetError::WeightCountMismatch {
                tensor: name.to_owned(),
                map_count: 12,
                gguf_count: 15,
            }
        );
    }

    #[test]
    fn pointer_groups_weights_by_tensor_site_layer_and_output_channel() {
        let name = "blk.7.attn_q.weight";
        let map = map(vec![slot(name, 12)]);
        let tensors = vec![tensor(name, 4, 3)];
        let ptr = Pointer {
            slot: 0,
            start_weight: 1,
            length_in_bits: 32,
            flags: 0,
            reserved: 0,
        };

        let regions = regions_for_pointer(&map, &tensors, ptr).expect("regions");

        assert_eq!(
            regions,
            vec![
                CompensationWriteRegion {
                    key: CompensationRegionKey {
                        tensor_name: name.to_owned(),
                        site: ActivationSite::QkvInput,
                        layer: 7,
                        output_channel: 0,
                    },
                    input_channels: vec![1, 2, 3],
                },
                CompensationWriteRegion {
                    key: CompensationRegionKey {
                        tensor_name: name.to_owned(),
                        site: ActivationSite::QkvInput,
                        layer: 7,
                        output_channel: 1,
                    },
                    input_channels: vec![0, 1, 2, 3],
                },
                CompensationWriteRegion {
                    key: CompensationRegionKey {
                        tensor_name: name.to_owned(),
                        site: ActivationSite::QkvInput,
                        layer: 7,
                        output_channel: 2,
                    },
                    input_channels: vec![0],
                },
            ]
        );
    }

    #[test]
    fn pointer_partial_final_weight_is_included() {
        let name = "blk.0.ffn_down.weight";
        let map = map(vec![slot(name, 8)]);
        let tensors = vec![tensor(name, 4, 2)];
        let ptr = Pointer {
            slot: 0,
            start_weight: 0,
            length_in_bits: 5,
            flags: 0,
            reserved: 0,
        };

        let regions = regions_for_pointer(&map, &tensors, ptr).expect("regions");

        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].input_channels, vec![0, 1]);
    }

    #[test]
    fn null_pointer_has_no_compensation_regions() {
        let map = map(vec![slot("blk.0.attn_q.weight", 12)]);
        let regions = regions_for_pointer(&map, &[], Pointer::NULL).expect("regions");
        assert!(regions.is_empty());
    }

    #[test]
    fn pointer_to_non_compensatable_tensor_has_no_regions() {
        let name = "token_embd.weight";
        let map = map(vec![slot(name, 12)]);
        let tensors = vec![tensor(name, 4, 3)];
        let ptr = Pointer {
            slot: 0,
            start_weight: 0,
            length_in_bits: 8,
            flags: 0,
            reserved: 0,
        };

        let regions = regions_for_pointer(&map, &tensors, ptr).expect("regions");

        assert!(regions.is_empty());
    }
}
