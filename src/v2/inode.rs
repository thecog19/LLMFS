//! V2 inode — addressing unit for any "file" in the filesystem
//! (user files, directories, internal metadata structures).
//!
//! Layout (248 bytes, little-endian throughout):
//!
//! ```text
//! offset size  field
//! 0      8     length u64
//! 8      192   direct[12]       12 × 16-byte Pointer
//! 200    16    single_indirect  Pointer (→ chunk of pointers)
//! 216    16    double_indirect  Pointer (→ chunk of chunks of pointers)
//! 232    16    triple_indirect  Pointer (→ chunk of chunks of chunks of pointers)
//! 248
//! ```
//!
//! This milestone (DESIGN-NEW §15.3) defines the codec and
//! constants. Direct-pointer traversal for reads and writes lands
//! in a later milestone; the single / double / triple indirect
//! forms are placeholders until then. The full 248-byte layout is
//! established immediately so neither of those later milestones
//! needs to bump the on-disk format.

use thiserror::Error;

use crate::v2::pointer::{Pointer, PointerError};

/// Number of direct pointers packed into the inode proper.
pub const NUM_DIRECT: usize = 12;

/// Total encoded size: `8 (length) + 15 pointers × 16 = 248`.
pub const INODE_BYTES: usize = 8 + (NUM_DIRECT + 3) * Pointer::SIZE;

const LENGTH_END: usize = 8;
const DIRECT_END: usize = LENGTH_END + NUM_DIRECT * Pointer::SIZE;
const SINGLE_END: usize = DIRECT_END + Pointer::SIZE;
const DOUBLE_END: usize = SINGLE_END + Pointer::SIZE;
const TRIPLE_END: usize = DOUBLE_END + Pointer::SIZE;

/// V2 inode. Represents a file (user or internal) via a tree of
/// `Pointer`s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inode {
    pub length: u64,
    pub direct: [Pointer; NUM_DIRECT],
    pub single_indirect: Pointer,
    pub double_indirect: Pointer,
    pub triple_indirect: Pointer,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum InodeError {
    #[error("inode codec truncated: need {INODE_BYTES} bytes, got {got}")]
    Truncated { got: usize },

    #[error("pointer decode failed in inode: {source}")]
    PointerDecode {
        #[from]
        source: PointerError,
    },
}

impl Inode {
    /// Canonical empty inode: zero-length, all pointers null.
    pub const EMPTY: Inode = Inode {
        length: 0,
        direct: [Pointer::NULL; NUM_DIRECT],
        single_indirect: Pointer::NULL,
        double_indirect: Pointer::NULL,
        triple_indirect: Pointer::NULL,
    };

    /// Serialise to the 248-byte wire form.
    pub fn encode(&self) -> [u8; INODE_BYTES] {
        let mut out = [0u8; INODE_BYTES];
        out[0..LENGTH_END].copy_from_slice(&self.length.to_le_bytes());
        for (i, ptr) in self.direct.iter().enumerate() {
            let off = LENGTH_END + i * Pointer::SIZE;
            out[off..off + Pointer::SIZE].copy_from_slice(&ptr.encode());
        }
        out[DIRECT_END..SINGLE_END].copy_from_slice(&self.single_indirect.encode());
        out[SINGLE_END..DOUBLE_END].copy_from_slice(&self.double_indirect.encode());
        out[DOUBLE_END..TRIPLE_END].copy_from_slice(&self.triple_indirect.encode());
        out
    }

    /// Deserialise exactly 248 bytes. Returns [`InodeError::Truncated`]
    /// for short inputs and [`InodeError::PointerDecode`] if any
    /// embedded pointer fails to decode (e.g. non-zero reserved).
    pub fn decode(bytes: &[u8]) -> Result<Self, InodeError> {
        if bytes.len() < INODE_BYTES {
            return Err(InodeError::Truncated { got: bytes.len() });
        }
        let length = u64::from_le_bytes(bytes[0..LENGTH_END].try_into().unwrap());
        let mut direct = [Pointer::NULL; NUM_DIRECT];
        for (i, ptr) in direct.iter_mut().enumerate() {
            let off = LENGTH_END + i * Pointer::SIZE;
            *ptr = Pointer::decode(&bytes[off..off + Pointer::SIZE])?;
        }
        let single_indirect = Pointer::decode(&bytes[DIRECT_END..SINGLE_END])?;
        let double_indirect = Pointer::decode(&bytes[SINGLE_END..DOUBLE_END])?;
        let triple_indirect = Pointer::decode(&bytes[DOUBLE_END..TRIPLE_END])?;
        Ok(Self {
            length,
            direct,
            single_indirect,
            double_indirect,
            triple_indirect,
        })
    }
}
