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
use crate::stego::calibration::placement::MetadataBitPos;
use crate::stego::tensor_map::TensorMap;
use crate::v2::pointer::Pointer;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ChunkError {
    #[error("chunk I/O out of bounds: byte_offset={byte_offset} len={len} capacity_bytes={capacity_bytes}")]
    OutOfBounds {
        byte_offset: u64,
        len: u64,
        capacity_bytes: u64,
    },
    #[error("pointer slot {slot} out of range (map has {slot_count} slots)")]
    SlotOutOfRange { slot: u16, slot_count: usize },
    #[error("pointer targets slot {slot}, which has no stealable bits")]
    NonStealableSlot { slot: u16 },
    #[error("pointer range [{start_weight}, {end_weight}) lies outside slot {slot} with {weight_count} weights")]
    PointerOutOfBounds {
        slot: u16,
        start_weight: u32,
        end_weight: u64,
        weight_count: u64,
    },
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
    Ok(())
}

fn resolve_slot(
    map: &TensorMap,
    pointer: Pointer,
) -> Result<&crate::stego::tensor_map::TensorSlot, ChunkError> {
    map.slots
        .get(pointer.slot as usize)
        .ok_or(ChunkError::SlotOutOfRange {
            slot: pointer.slot,
            slot_count: map.slots.len(),
        })
}

fn bits_per_weight(
    slot: &crate::stego::tensor_map::TensorSlot,
    slot_index: u16,
) -> Result<u64, ChunkError> {
    let bits = slot.stealable_bits_per_weight as u64;
    if bits == 0 {
        Err(ChunkError::NonStealableSlot { slot: slot_index })
    } else {
        Ok(bits)
    }
}

fn validate_pointer_range(
    slot: &crate::stego::tensor_map::TensorSlot,
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

fn check_bounds(pointer: Pointer, byte_offset: u64, len: u64) -> Result<(), ChunkError> {
    let cap = byte_capacity(pointer);
    let end = byte_offset.checked_add(len).ok_or(ChunkError::OutOfBounds {
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
