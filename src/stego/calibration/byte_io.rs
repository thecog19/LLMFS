//! Byte-level I/O over a [`MetadataPlacement`]. Sits on top of
//! [`bit_io`](super::bit_io): byte `N` occupies
//! `placement.positions[N*8 .. N*8+8]`, with bit 0 the LSB of the
//! byte.
//!
//! V2's Layer 1 (DESIGN-NEW §15.3) stores byte-oriented structures —
//! a V1-shaped superblock, a sensitivity table, a placement index —
//! inside the low-magnitude region Layer 0 picks out. The V1 metadata
//! writers already pass `&[u8]`; this adapter lets them target a
//! Layer 0 region without caring about the per-quant bit addressing
//! that `bit_io` handles.
//!
//! Tail bits (`placement.positions.len() % 8`) are not reachable via
//! byte I/O. [`byte_capacity`] floors the bit count; a caller that
//! needs a partial byte has to use `bit_io` directly.
//!
//! All bounds checks happen up front: an out-of-bounds [`write_bytes`]
//! returns `OutOfBoundsError` without mutating the cover.
//! `bit_io::write_bit` still touches the cover byte-at-a-time, so the
//! up-front check is the only thing that keeps partial writes from
//! landing in the cover on an OOB caller.
//!
//! Bit-numbering convention:
//! ```text
//! byte b, bit k ∈ 0..8   →   placement.positions[byte_offset*8 + k]
//! value set iff   (b >> k) & 1 == 1
//! ```
//! LSB-first matches little-endian byte ordering used elsewhere in
//! the codebase and keeps `0x01 → position[0]` obvious.

use thiserror::Error;

use crate::stego::calibration::bit_io::{read_bit, write_bit};
use crate::stego::calibration::placement::MetadataPlacement;
use crate::stego::tensor_map::TensorMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("byte I/O out of range: offset={offset} len={len} capacity_bytes={capacity_bytes}")]
pub struct OutOfBoundsError {
    pub offset: u64,
    pub len: u64,
    pub capacity_bytes: u64,
}

/// Number of bytes addressable through byte I/O. Any tail bits
/// (`positions.len() % 8`) are unreachable via this interface.
pub fn byte_capacity(placement: &MetadataPlacement) -> u64 {
    placement.len_bits() / 8
}

/// Read `buf.len()` bytes from the placement starting at
/// `byte_offset`. Returns `OutOfBoundsError` when
/// `byte_offset + buf.len() > byte_capacity`; the cover is never
/// touched in that case.
pub fn read_bytes(
    mmap: &[u8],
    map: &TensorMap,
    placement: &MetadataPlacement,
    byte_offset: u64,
    buf: &mut [u8],
) -> Result<(), OutOfBoundsError> {
    check_bounds(placement, byte_offset, buf.len() as u64)?;
    for (i, slot_byte) in buf.iter_mut().enumerate() {
        let base = ((byte_offset + i as u64) * 8) as usize;
        let mut value: u8 = 0;
        for k in 0..8 {
            let pos = placement.positions[base + k];
            if read_bit(mmap, &map.slots[pos.slot_index as usize], pos) {
                value |= 1 << k;
            }
        }
        *slot_byte = value;
    }
    Ok(())
}

/// Write `data` to the placement starting at `byte_offset`. Returns
/// `OutOfBoundsError` when `byte_offset + data.len() > byte_capacity`;
/// the cover is not touched on error.
pub fn write_bytes(
    mmap: &mut [u8],
    map: &TensorMap,
    placement: &MetadataPlacement,
    byte_offset: u64,
    data: &[u8],
) -> Result<(), OutOfBoundsError> {
    check_bounds(placement, byte_offset, data.len() as u64)?;
    for (i, byte) in data.iter().enumerate() {
        let base = ((byte_offset + i as u64) * 8) as usize;
        for k in 0..8 {
            let pos = placement.positions[base + k];
            let bit = (byte >> k) & 1 == 1;
            write_bit(mmap, &map.slots[pos.slot_index as usize], pos, bit);
        }
    }
    Ok(())
}

fn check_bounds(
    placement: &MetadataPlacement,
    byte_offset: u64,
    len: u64,
) -> Result<(), OutOfBoundsError> {
    let cap = byte_capacity(placement);
    let end = byte_offset.checked_add(len).ok_or(OutOfBoundsError {
        offset: byte_offset,
        len,
        capacity_bytes: cap,
    })?;
    if end > cap {
        return Err(OutOfBoundsError {
            offset: byte_offset,
            len,
            capacity_bytes: cap,
        });
    }
    Ok(())
}
