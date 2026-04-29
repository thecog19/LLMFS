//! Variable-length chunk I/O over a [`Pointer`].
//!
//! A chunk is a contiguous run of stealable bits in a single tensor
//! slot. Its address — slot, start weight, and length in stealable
//! bits — is a `Pointer`; this module reads and writes bytes into
//! that run via the same `bit_io`-backed byte-packing convention
//! [`byte_io`] uses at the [`MetadataPlacement`] layer, so inter-
//! module consistency is straightforward to check.
//!
//! Bit ordering: byte `n` at `byte_offset + n` occupies chunk-bits
//! `[(byte_offset + n) * 8, (byte_offset + n) * 8 + 8)`, LSB first —
//! byte value `b`, bit `k ∈ 0..8` corresponds to `(b >> k) & 1`. This
//! matches [`byte_io`] and keeps `0x01` → "bit 0 only" obvious.
//!
//! Position within the slot for chunk-bit `i`:
//! ```text
//! weight_index = pointer.start_weight + i / bits_per_weight
//! bit_index    = i % bits_per_weight
//! ```
//! where `bits_per_weight = slot.stealable_bits_per_weight` — set by
//! the planner from [`GgufQuantType::stealable_bits_hint`].
//!
//! Bounds checks are up-front: an out-of-bounds [`write_chunk`]
//! returns `OutOfBounds` without mutating the cover. [`bit_io`] edits
//! in place, so the early check is what prevents partial-write
//! corruption on OOB callers.
//!
//! If `length_in_bits` is not a multiple of 8, the chunk still exposes
//! a final partial byte. Reads zero the unused high bits of that last
//! byte; writes consume only the low live bits and leave the rest of
//! the cover unchanged.
//!
//! [`byte_io`]: crate::stego::calibration::byte_io
//! [`MetadataPlacement`]: crate::stego::calibration::placement::MetadataPlacement
//! [`GgufQuantType::stealable_bits_hint`]: crate::gguf::quant::GgufQuantType::stealable_bits_hint

use thiserror::Error;

use crate::stego::calibration::bit_io::{read_bit, write_bit};
use crate::stego::calibration::magnitude::read_weight_value;
use crate::stego::calibration::placement::MetadataBitPos;
use crate::stego::tensor_map::{TensorMap, TensorSlot};
use crate::v2::pointer::Pointer;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ChunkError {
    #[error(
        "chunk I/O out of bounds: byte_offset={byte_offset} len={len} capacity_bytes={capacity_bytes}"
    )]
    OutOfBounds {
        byte_offset: u64,
        len: u64,
        capacity_bytes: u64,
    },
    #[error("pointer slot {slot} out of range (map has {slot_count} slots)")]
    SlotOutOfRange { slot: u16, slot_count: usize },
    #[error("weight {weight_index} out of range for slot {slot} with {weight_count} weights")]
    WeightOutOfRange {
        slot: u16,
        weight_index: u64,
        weight_count: u64,
    },
    #[error("pointer targets slot {slot}, which has no stealable bits")]
    NonStealableSlot { slot: u16 },
    #[error("target compensation value is not finite")]
    NonFiniteTarget,
    #[error("no finite candidate value for slot {slot} weight {weight_index}")]
    NoFiniteWeightCandidate { slot: u16, weight_index: u64 },
    #[error(
        "pointer range [{start_weight}, {end_weight}) lies outside slot {slot} with {weight_count} weights"
    )]
    PointerOutOfBounds {
        slot: u16,
        start_weight: u32,
        end_weight: u64,
        weight_count: u64,
    },
}

/// Signed change to one model weight caused by a chunk write.
///
/// `before` and `after` are decoded with the same quant-specific
/// reader used by calibration, so `delta()` is the forced model-space
/// perturbation that Phase E compensation consumes for this weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightDelta {
    pub slot: u16,
    pub weight_index: u64,
    pub before: f32,
    pub after: f32,
}

impl WeightDelta {
    pub fn delta(&self) -> f32 {
        self.after - self.before
    }
}

/// Bytes addressable through the chunk — `ceil(length_in_bits / 8)`.
/// A tail below a byte boundary is surfaced as a partial final byte.
pub fn byte_capacity(pointer: Pointer) -> u64 {
    (pointer.length_in_bits as u64).div_ceil(8)
}

/// Read `buf.len()` bytes from the chunk starting at `byte_offset`.
/// Returns [`ChunkError::OutOfBounds`] when
/// `byte_offset + buf.len() > byte_capacity`; the cover is never
/// touched on error.
pub fn read_chunk(
    mmap: &[u8],
    map: &TensorMap,
    pointer: Pointer,
    byte_offset: u64,
    buf: &mut [u8],
) -> Result<(), ChunkError> {
    let slot = resolve_slot(map, pointer)?;
    let bits_per_weight = bits_per_weight(slot, pointer.slot)?;
    validate_pointer_range(slot, pointer, bits_per_weight)?;
    check_bounds(pointer, byte_offset, buf.len() as u64)?;
    let bit_len = pointer.length_in_bits as u64;
    for (i, dst) in buf.iter_mut().enumerate() {
        let chunk_bit_base = (byte_offset + i as u64) * 8;
        let mut v: u8 = 0;
        let live_bits = bit_len.saturating_sub(chunk_bit_base).min(8);
        for k in 0..live_bits {
            let bit_i = chunk_bit_base + k;
            let pos = position_for_bit(pointer, bits_per_weight, bit_i);
            if read_bit(mmap, slot, pos) {
                v |= 1 << k;
            }
        }
        *dst = v;
    }
    Ok(())
}

/// Write `data` to the chunk starting at `byte_offset`. Returns
/// [`ChunkError::OutOfBounds`] when
/// `byte_offset + data.len() > byte_capacity`; the cover is not
/// touched on error.
pub fn write_chunk(
    mmap: &mut [u8],
    map: &TensorMap,
    pointer: Pointer,
    byte_offset: u64,
    data: &[u8],
) -> Result<(), ChunkError> {
    let slot = resolve_slot(map, pointer)?;
    let bits_per_weight = bits_per_weight(slot, pointer.slot)?;
    validate_pointer_range(slot, pointer, bits_per_weight)?;
    check_bounds(pointer, byte_offset, data.len() as u64)?;
    write_chunk_bits(mmap, slot, pointer, bits_per_weight, byte_offset, data);
    Ok(())
}

/// Write `data` with the same semantics as [`write_chunk`], returning
/// one [`WeightDelta`] per weight whose stealable bits were in the
/// live write range.
///
/// The returned vector includes zero-delta entries when a touched
/// weight's payload bits decode back to the same model value. Callers
/// can filter on [`WeightDelta::delta`] when they only need non-zero
/// perturbations.
pub fn write_chunk_with_weight_deltas(
    mmap: &mut [u8],
    map: &TensorMap,
    pointer: Pointer,
    byte_offset: u64,
    data: &[u8],
) -> Result<Vec<WeightDelta>, ChunkError> {
    let slot = resolve_slot(map, pointer)?;
    let bits_per_weight = bits_per_weight(slot, pointer.slot)?;
    validate_pointer_range(slot, pointer, bits_per_weight)?;
    check_bounds(pointer, byte_offset, data.len() as u64)?;

    let mut deltas: Vec<WeightDelta> =
        touched_weight_indices(pointer, bits_per_weight, byte_offset, data.len() as u64)
            .map(|weight_index| {
                let before = read_weight_value(mmap, slot, weight_index);
                WeightDelta {
                    slot: pointer.slot,
                    weight_index,
                    before,
                    after: before,
                }
            })
            .collect();

    write_chunk_bits(mmap, slot, pointer, bits_per_weight, byte_offset, data);

    for delta in &mut deltas {
        delta.after = read_weight_value(mmap, slot, delta.weight_index);
    }
    Ok(deltas)
}

/// Set one weight's stealable bits to the pattern whose decoded
/// model-space value is closest to `target`.
///
/// This is the compensation actuator primitive: it preserves every
/// non-stealable bit, enumerates the quant-type-specific stealable
/// bit patterns for the selected weight, and returns the signed
/// before/after model-space delta caused by the chosen write.
pub fn write_weight_nearest_value(
    mmap: &mut [u8],
    map: &TensorMap,
    slot_index: u16,
    weight_index: u64,
    target: f32,
) -> Result<WeightDelta, ChunkError> {
    if !target.is_finite() {
        return Err(ChunkError::NonFiniteTarget);
    }
    let slot = map
        .slots
        .get(slot_index as usize)
        .ok_or(ChunkError::SlotOutOfRange {
            slot: slot_index,
            slot_count: map.slots.len(),
        })?;
    if weight_index >= slot.weight_count {
        return Err(ChunkError::WeightOutOfRange {
            slot: slot_index,
            weight_index,
            weight_count: slot.weight_count,
        });
    }
    let bits_per_weight = bits_per_weight(slot, slot_index)? as u8;
    let before = read_weight_value(mmap, slot, weight_index);
    let original_pattern =
        read_stealable_pattern(mmap, slot, slot_index, weight_index, bits_per_weight);
    let pattern_count = 1_u32 << bits_per_weight;

    let mut best_pattern = None;
    let mut best_error = f32::INFINITY;
    for pattern in 0..pattern_count {
        write_stealable_pattern(
            mmap,
            slot,
            slot_index,
            weight_index,
            bits_per_weight,
            pattern,
        );
        let candidate = read_weight_value(mmap, slot, weight_index);
        if !candidate.is_finite() {
            continue;
        }
        let error = (candidate - target).abs();
        if error < best_error {
            best_error = error;
            best_pattern = Some(pattern);
        }
    }

    let Some(best_pattern) = best_pattern else {
        write_stealable_pattern(
            mmap,
            slot,
            slot_index,
            weight_index,
            bits_per_weight,
            original_pattern,
        );
        return Err(ChunkError::NoFiniteWeightCandidate {
            slot: slot_index,
            weight_index,
        });
    };
    write_stealable_pattern(
        mmap,
        slot,
        slot_index,
        weight_index,
        bits_per_weight,
        best_pattern,
    );
    let after = read_weight_value(mmap, slot, weight_index);
    Ok(WeightDelta {
        slot: slot_index,
        weight_index,
        before,
        after,
    })
}

fn read_stealable_pattern(
    mmap: &[u8],
    slot: &TensorSlot,
    slot_index: u16,
    weight_index: u64,
    bits_per_weight: u8,
) -> u32 {
    let mut pattern = 0_u32;
    for bit_index in 0..bits_per_weight {
        let pos = MetadataBitPos {
            slot_index: u32::from(slot_index),
            weight_index,
            bit_index,
        };
        if read_bit(mmap, slot, pos) {
            pattern |= 1_u32 << bit_index;
        }
    }
    pattern
}

fn write_stealable_pattern(
    mmap: &mut [u8],
    slot: &TensorSlot,
    slot_index: u16,
    weight_index: u64,
    bits_per_weight: u8,
    pattern: u32,
) {
    for bit_index in 0..bits_per_weight {
        let pos = MetadataBitPos {
            slot_index: u32::from(slot_index),
            weight_index,
            bit_index,
        };
        write_bit(mmap, slot, pos, (pattern >> bit_index) & 1 == 1);
    }
}

fn write_chunk_bits(
    mmap: &mut [u8],
    slot: &TensorSlot,
    pointer: Pointer,
    bits_per_weight: u64,
    byte_offset: u64,
    data: &[u8],
) {
    let bit_len = pointer.length_in_bits as u64;
    for (i, &byte) in data.iter().enumerate() {
        let chunk_bit_base = (byte_offset + i as u64) * 8;
        let live_bits = bit_len.saturating_sub(chunk_bit_base).min(8);
        for k in 0..live_bits {
            let bit_i = chunk_bit_base + k;
            let pos = position_for_bit(pointer, bits_per_weight, bit_i);
            let bit = ((byte >> k) & 1) == 1;
            write_bit(mmap, slot, pos, bit);
        }
    }
}

fn resolve_slot(map: &TensorMap, pointer: Pointer) -> Result<&TensorSlot, ChunkError> {
    map.slots
        .get(pointer.slot as usize)
        .ok_or(ChunkError::SlotOutOfRange {
            slot: pointer.slot,
            slot_count: map.slots.len(),
        })
}

fn bits_per_weight(slot: &TensorSlot, slot_index: u16) -> Result<u64, ChunkError> {
    let bits = slot.stealable_bits_per_weight as u64;
    if bits == 0 {
        Err(ChunkError::NonStealableSlot { slot: slot_index })
    } else {
        Ok(bits)
    }
}

fn validate_pointer_range(
    slot: &TensorSlot,
    pointer: Pointer,
    bits_per_weight: u64,
) -> Result<(), ChunkError> {
    let covered_weights = if pointer.length_in_bits == 0 {
        0
    } else {
        (pointer.length_in_bits as u64).div_ceil(bits_per_weight)
    };
    let end_weight = pointer.start_weight as u64 + covered_weights;
    if end_weight > slot.weight_count {
        return Err(ChunkError::PointerOutOfBounds {
            slot: pointer.slot,
            start_weight: pointer.start_weight,
            end_weight,
            weight_count: slot.weight_count,
        });
    }
    Ok(())
}

fn touched_weight_indices(
    pointer: Pointer,
    bits_per_weight: u64,
    byte_offset: u64,
    len: u64,
) -> impl Iterator<Item = u64> {
    let start_bit = byte_offset * 8;
    let end_bit = (start_bit + len * 8).min(pointer.length_in_bits as u64);
    let start_weight = u64::from(pointer.start_weight);
    let start = start_weight + start_bit / bits_per_weight;
    let end = if start_bit >= end_bit {
        start
    } else {
        start_weight + (end_bit - 1) / bits_per_weight + 1
    };
    start..end
}

fn check_bounds(pointer: Pointer, byte_offset: u64, len: u64) -> Result<(), ChunkError> {
    let cap = byte_capacity(pointer);
    let end = byte_offset
        .checked_add(len)
        .ok_or(ChunkError::OutOfBounds {
            byte_offset,
            len,
            capacity_bytes: cap,
        })?;
    if end > cap {
        return Err(ChunkError::OutOfBounds {
            byte_offset,
            len,
            capacity_bytes: cap,
        });
    }
    Ok(())
}

fn position_for_bit(pointer: Pointer, bits_per_weight: u64, chunk_bit: u64) -> MetadataBitPos {
    let weight_offset = chunk_bit / bits_per_weight;
    let bit_index = (chunk_bit % bits_per_weight) as u8;
    MetadataBitPos {
        slot_index: pointer.slot as u32,
        weight_index: pointer.start_weight as u64 + weight_offset,
        bit_index,
    }
}
