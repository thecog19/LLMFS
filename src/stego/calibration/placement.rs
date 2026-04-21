//! Metadata-bit placement: turns a salience ranking (currently
//! magnitude; later AWQ / Hessian) into the concrete list of bit
//! positions the metadata writer should occupy.
//!
//! Layer 0's job per DESIGN-NEW §15.2 is to put V2 metadata into the
//! lowest-salience bit positions in the cover. This module produces
//! that placement — the device layer will eventually consume it
//! instead of the V1 "block 0..K" contiguous layout.
//!
//! The placement is deterministic given the cover's static weight
//! values and the requested bit count, so reopen across sessions
//! recovers the same positions without persisting them.

use crate::stego::calibration::magnitude::lowest_magnitude_weights_for_bits;
use crate::stego::calibration::{WeightRef, stealable_bits_for};
use crate::stego::tensor_map::TensorMap;

/// One stealable bit position inside one weight: enough to address a
/// single bit of metadata in the cover.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MetadataBitPos {
    pub slot_index: u32,
    pub weight_index: u64,
    /// 0..stealable_bits_for(slot.quant_type). The interpretation is
    /// packer-defined: bit 0 is the lowest stealable bit of the
    /// weight as the packer numbers them.
    pub bit_index: u8,
}

/// Ordered list of bit positions reserved for metadata. The order is
/// the addressing convention: bit 0 of the metadata region is
/// `positions[0]`, bit 1 is `positions[1]`, etc. Block-level
/// addressing (the V1 BLOCK_SIZE-byte block) is `positions[N*K..]`
/// where `K = BLOCK_SIZE * 8`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataPlacement {
    pub positions: Vec<MetadataBitPos>,
}

impl MetadataPlacement {
    pub fn len_bits(&self) -> u64 {
        self.positions.len() as u64
    }
}

/// Compute the metadata-bit placement for `needed_bits` of metadata
/// against `map` and the cover-bytes `mmap`. The placement contains
/// at least `needed_bits` positions, drawn from the lowest-magnitude
/// weights and their stealable bits in order.
///
/// Determinism: same inputs → same positions, irrespective of host
/// architecture (the magnitude ranking is fixed-point comparable, and
/// `WeightRef::Ord` resolves ties lexicographically).
pub fn compute_metadata_placement(
    mmap: &[u8],
    map: &TensorMap,
    needed_bits: u64,
) -> MetadataPlacement {
    let weights = lowest_magnitude_weights_for_bits(mmap, map, needed_bits);
    let mut positions = Vec::with_capacity(needed_bits.max(1) as usize);

    for r in weights {
        let bits_in_weight = stealable_bits_for(map.slots[r.slot_index as usize].quant_type);
        for bit in 0..bits_in_weight {
            positions.push(MetadataBitPos {
                slot_index: r.slot_index,
                weight_index: r.weight_index,
                bit_index: bit as u8,
            });
            if positions.len() as u64 >= needed_bits {
                return MetadataPlacement { positions };
            }
        }
    }

    // Caller asked for more than the cover can host; return what we
    // got. Higher layers should validate `len_bits() >= needed_bits`
    // before committing.
    MetadataPlacement { positions }
}

/// Produce the reverse lookup: which bit positions in the cover are
/// reserved for metadata. The data-region addressing layer needs
/// this to skip metadata-occupied positions when computing physical
/// data offsets.
///
/// Returned as a hash set of `(slot_index, weight_index, bit_index)`
/// tuples. For larger covers this is `O(metadata_bits)` memory —
/// trivial because metadata is a tiny fraction of the address space.
pub fn metadata_position_set(
    placement: &MetadataPlacement,
) -> std::collections::HashSet<MetadataBitPos> {
    placement.positions.iter().copied().collect()
}

/// Quick reference to the WeightRef of every metadata-occupied
/// weight (collapsing the per-bit positions back to per-weight).
pub fn metadata_weight_refs(placement: &MetadataPlacement) -> std::collections::HashSet<WeightRef> {
    placement
        .positions
        .iter()
        .map(|p| WeightRef {
            slot_index: p.slot_index,
            weight_index: p.weight_index,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::quant::GgufQuantType;
    use crate::stego::planner::TensorTier;
    use crate::stego::tensor_map::TensorSlot;

    fn f16_slot(weight_count: u64) -> TensorSlot {
        let bits = GgufQuantType::F16.stealable_bits_hint() as u64;
        TensorSlot {
            name: "test.f16".to_owned(),
            quant_type: GgufQuantType::F16,
            tier: TensorTier::Tier1,
            data_offset: 0,
            weight_count,
            stealable_bits_per_weight: bits as usize,
            capacity_bits: weight_count * bits,
            bit_start: 0,
            bit_end: weight_count * bits,
        }
    }

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(values.len() * 2);
        for v in values {
            // Trivial fp16 encoding via f32 bit fiddling — same shape as
            // the test helpers in tests/calibration_magnitude.rs.
            let bits = v.to_bits();
            let sign = ((bits >> 31) & 0x1) as u16;
            let exp32 = ((bits >> 23) & 0xFF) as i32;
            let mantissa32 = bits & 0x7FFFFF;
            let f16_bits = if exp32 == 0 {
                sign << 15
            } else {
                let exp16 = exp32 - 127 + 15;
                if exp16 <= 0 {
                    sign << 15
                } else if exp16 >= 31 {
                    (sign << 15) | (0x1F << 10)
                } else {
                    (sign << 15) | ((exp16 as u16) << 10) | ((mantissa32 >> 13) as u16)
                }
            };
            b.extend_from_slice(&f16_bits.to_le_bytes());
        }
        b
    }

    fn map_with(slot: TensorSlot) -> TensorMap {
        let bits = slot.capacity_bits;
        TensorMap {
            slots: vec![slot],
            total_capacity_bits: bits,
            total_capacity_bytes: bits / 8,
        }
    }

    #[test]
    fn placement_returns_at_least_needed_bits() {
        // 8 F16 weights × 4 stealable bits = 32 bits available
        let bytes = f16_bytes(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let map = map_with(f16_slot(8));
        let p = compute_metadata_placement(&bytes, &map, 16);
        assert!(p.len_bits() >= 16);
    }

    #[test]
    fn placement_is_per_weight_then_per_bit() {
        // 4 F16 weights = 16 bits total. Magnitudes ascending [0.1, 0.2, 0.3, 0.4].
        let bytes = f16_bytes(&[0.1, 0.2, 0.3, 0.4]);
        let map = map_with(f16_slot(4));
        let p = compute_metadata_placement(&bytes, &map, 16);
        assert_eq!(p.positions.len(), 16);

        // First 4 positions: weight 0 (smallest |w|), bits 0..4
        for (i, pos) in p.positions[0..4].iter().enumerate() {
            assert_eq!(pos.slot_index, 0);
            assert_eq!(pos.weight_index, 0);
            assert_eq!(pos.bit_index as usize, i);
        }
        // Next 4: weight 1, bits 0..4
        for (i, pos) in p.positions[4..8].iter().enumerate() {
            assert_eq!(pos.weight_index, 1);
            assert_eq!(pos.bit_index as usize, i);
        }
    }

    #[test]
    fn placement_truncates_to_needed_bits() {
        let bytes = f16_bytes(&[0.1, 0.2, 0.3, 0.4]);
        let map = map_with(f16_slot(4));
        let p = compute_metadata_placement(&bytes, &map, 6);
        assert_eq!(p.positions.len(), 6);
        // First 4 from weight 0; last 2 from weight 1 (bits 0 and 1)
        assert_eq!(p.positions[0].weight_index, 0);
        assert_eq!(p.positions[3].weight_index, 0);
        assert_eq!(p.positions[4].weight_index, 1);
        assert_eq!(p.positions[4].bit_index, 0);
        assert_eq!(p.positions[5].bit_index, 1);
    }

    #[test]
    fn placement_zero_bits_returns_empty() {
        let bytes = f16_bytes(&[0.1, 0.2]);
        let map = map_with(f16_slot(2));
        let p = compute_metadata_placement(&bytes, &map, 0);
        assert!(p.positions.is_empty());
    }

    #[test]
    fn placement_is_deterministic_across_repeated_calls() {
        let bytes = f16_bytes(&[1.0, 0.001, 0.5, 0.002, 0.1, 0.0001, 10.0, 0.05]);
        let map = map_with(f16_slot(8));
        let a = compute_metadata_placement(&bytes, &map, 12);
        let b = compute_metadata_placement(&bytes, &map, 12);
        assert_eq!(a, b);
    }

    #[test]
    fn metadata_weight_refs_collapses_per_weight_bits() {
        let bytes = f16_bytes(&[0.1, 0.2, 0.3]);
        let map = map_with(f16_slot(3));
        // 12 bits = 3 F16 weights
        let p = compute_metadata_placement(&bytes, &map, 12);
        let refs = metadata_weight_refs(&p);
        assert_eq!(refs.len(), 3);
    }
}
