//! [`Pointer`] — addresses a contiguous run of stealable bits in one
//! tensor slot. Every inode direct-pointer entry and every
//! pointer-list cell in an indirect block stores one of these.
//!
//! Layout (16 bytes, little-endian throughout; see
//! `tests/v2_pointer.rs` for the byte-for-byte contract):
//!
//! ```text
//! offset size  field
//! 0      2     slot             u16
//! 2      4     start_weight     u32
//! 6      4     length_in_bits   u32
//! 10     2     flags            u16
//! 12     4     reserved         u32  (must be zero on decode)
//! ```
//!
//! The null pointer is `length_in_bits == 0`. Any other state is a
//! valid pointer to some run of stealable bits; `start_weight` is the
//! index of the first weight and `length_in_bits` is the extent (in
//! stealable bits, not bytes — the two differ per-quant-type).
//!
//! Reserved must be zero on decode so future format bumps can extend
//! the layout without the current code silently accepting unknown
//! state.

use thiserror::Error;

/// Reference to a contiguous run of stealable-bit positions within a
/// single tensor slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Pointer {
    pub slot: u16,
    pub start_weight: u32,
    pub length_in_bits: u32,
    pub flags: u16,
    pub reserved: u32,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PointerError {
    #[error("pointer codec truncated: need 16 bytes at offset {offset}, buffer has {available}")]
    Truncated { offset: usize, available: usize },
    #[error("pointer reserved field non-zero ({found:#010x}); refusing to decode unknown state")]
    NonZeroReserved { found: u32 },
}

impl Pointer {
    /// Size of the encoded form, in bytes.
    pub const SIZE: usize = 16;

    /// All-zero pointer — represents "no pointer here."
    pub const NULL: Pointer = Pointer {
        slot: 0,
        start_weight: 0,
        length_in_bits: 0,
        flags: 0,
        reserved: 0,
    };

    /// True if this pointer references nothing. `length_in_bits == 0`
    /// is the canonical null signal — a real chunk always has
    /// `length_in_bits > 0`.
    pub fn is_null(&self) -> bool {
        self.length_in_bits == 0
    }

    /// Serialise to the 16-byte wire form.
    pub fn encode(&self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        out[0..2].copy_from_slice(&self.slot.to_le_bytes());
        out[2..6].copy_from_slice(&self.start_weight.to_le_bytes());
        out[6..10].copy_from_slice(&self.length_in_bits.to_le_bytes());
        out[10..12].copy_from_slice(&self.flags.to_le_bytes());
        out[12..16].copy_from_slice(&self.reserved.to_le_bytes());
        out
    }

    /// Deserialise exactly 16 bytes. Returns [`PointerError::Truncated`]
    /// for short inputs and [`PointerError::NonZeroReserved`] if the
    /// reserved field is set.
    pub fn decode(bytes: &[u8]) -> Result<Self, PointerError> {
        if bytes.len() < Self::SIZE {
            return Err(PointerError::Truncated {
                offset: 0,
                available: bytes.len(),
            });
        }
        let slot = u16::from_le_bytes([bytes[0], bytes[1]]);
        let start_weight = u32::from_le_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]);
        let length_in_bits =
            u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]);
        let flags = u16::from_le_bytes([bytes[10], bytes[11]]);
        let reserved =
            u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        if reserved != 0 {
            return Err(PointerError::NonZeroReserved { found: reserved });
        }
        Ok(Self {
            slot,
            start_weight,
            length_in_bits,
            flags,
            reserved,
        })
    }

    /// Read a pointer from `buf` starting at byte `offset`. Returns
    /// [`PointerError::Truncated`] if `buf[offset..offset + 16]` is
    /// out of range.
    pub fn read_at(buf: &[u8], offset: usize) -> Result<Self, PointerError> {
        if buf.len() < offset + Self::SIZE {
            return Err(PointerError::Truncated {
                offset,
                available: buf.len(),
            });
        }
        Self::decode(&buf[offset..offset + Self::SIZE])
    }

    /// Write this pointer to `buf` starting at byte `offset`. Same
    /// bounds rules as [`Self::read_at`].
    pub fn write_at(&self, buf: &mut [u8], offset: usize) -> Result<(), PointerError> {
        if buf.len() < offset + Self::SIZE {
            return Err(PointerError::Truncated {
                offset,
                available: buf.len(),
            });
        }
        buf[offset..offset + Self::SIZE].copy_from_slice(&self.encode());
        Ok(())
    }
}
