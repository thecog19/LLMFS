//! V2 super-root record.
//!
//! The super-root is the inode-like structure the anchor points at
//! after init / every commit. It carries pointers to every other V2
//! metadata structure — root directory inode, dedup index, dirty
//! bitmap, free-run state, ceiling-magnitude bucket summary — plus
//! a generation counter that mirrors the anchor's.
//!
//! Layout (100 bytes, little-endian throughout):
//!
//! ```text
//! offset size  field
//! 0      4     magic = b"V2SR"
//! 4      1     version = 1
//! 5      3     reserved (zero)
//! 8      16    root_dir_inode        Pointer
//! 24     16    dedup_index_inode     Pointer
//! 40     16    dirty_bitmap_inode    Pointer
//! 56     16    free_run_state_inode  Pointer
//! 72     16    ceiling_summary_inode Pointer
//! 88     8     generation u64
//! 96     4     crc32 (covers bytes 0..96)
//! 100
//! ```
//!
//! Super-root chunks are allocated + written via the V2 chunk layer;
//! the anchor's two slots point at chunks holding these records. The
//! CRC guards against corruption inside the 96 content bytes
//! (pointer updates, generation bumps) — the anchor's own per-slot
//! CRC only covers the anchor pointer to the super-root, not the
//! super-root's contents.

use thiserror::Error;

use crate::v2::pointer::{Pointer, PointerError};

pub const MAGIC: &[u8; 4] = b"V2SR";
pub const VERSION: u8 = 1;
/// Encoded size in bytes: 4 magic + 1 version + 3 reserved + 5×16 pointers + 8 generation + 4 crc = 100.
pub const SUPER_ROOT_BYTES: usize = 4 + 1 + 3 + 5 * Pointer::SIZE + 8 + 4;

const HEADER_END: usize = 8;
const ROOT_DIR_END: usize = HEADER_END + Pointer::SIZE; // 24
const DEDUP_END: usize = ROOT_DIR_END + Pointer::SIZE; // 40
const DIRTY_END: usize = DEDUP_END + Pointer::SIZE; // 56
const FREELIST_END: usize = DIRTY_END + Pointer::SIZE; // 72
const CEILING_END: usize = FREELIST_END + Pointer::SIZE; // 88
const GENERATION_END: usize = CEILING_END + 8; // 96
const CRC_END: usize = GENERATION_END + 4; // 100

const _: () = {
    // Compile-time assert the layout calculation.
    assert!(CRC_END == SUPER_ROOT_BYTES);
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuperRoot {
    pub root_dir_inode: Pointer,
    pub dedup_index_inode: Pointer,
    pub dirty_bitmap_inode: Pointer,
    pub free_run_state_inode: Pointer,
    pub ceiling_summary_inode: Pointer,
    pub generation: u64,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SuperRootError {
    #[error("super-root codec truncated: need {SUPER_ROOT_BYTES} bytes, got {got}")]
    Truncated { got: usize },

    #[error("bad super-root magic: expected b\"V2SR\", got {found:?}")]
    BadMagic { found: [u8; 4] },

    #[error("unsupported super-root version: {0}")]
    UnsupportedVersion(u8),

    #[error("super-root CRC mismatch: computed {computed:#010x}, stored {stored:#010x}")]
    BadChecksum { computed: u32, stored: u32 },

    #[error("pointer decode failed in super-root: {source}")]
    PointerDecode {
        #[from]
        source: PointerError,
    },
}

impl SuperRoot {
    /// Canonical empty super-root: all pointers null, generation zero.
    pub const EMPTY: SuperRoot = SuperRoot {
        root_dir_inode: Pointer::NULL,
        dedup_index_inode: Pointer::NULL,
        dirty_bitmap_inode: Pointer::NULL,
        free_run_state_inode: Pointer::NULL,
        ceiling_summary_inode: Pointer::NULL,
        generation: 0,
    };

    /// Serialise to the 100-byte wire form with trailing CRC32.
    pub fn encode(&self) -> [u8; SUPER_ROOT_BYTES] {
        let mut out = [0u8; SUPER_ROOT_BYTES];
        out[0..4].copy_from_slice(MAGIC);
        out[4] = VERSION;
        // out[5..8] stays zero (reserved).
        out[HEADER_END..ROOT_DIR_END].copy_from_slice(&self.root_dir_inode.encode());
        out[ROOT_DIR_END..DEDUP_END].copy_from_slice(&self.dedup_index_inode.encode());
        out[DEDUP_END..DIRTY_END].copy_from_slice(&self.dirty_bitmap_inode.encode());
        out[DIRTY_END..FREELIST_END].copy_from_slice(&self.free_run_state_inode.encode());
        out[FREELIST_END..CEILING_END].copy_from_slice(&self.ceiling_summary_inode.encode());
        out[CEILING_END..GENERATION_END].copy_from_slice(&self.generation.to_le_bytes());
        let crc = crc_of(&out[0..GENERATION_END]);
        out[GENERATION_END..CRC_END].copy_from_slice(&crc.to_le_bytes());
        out
    }

    /// Deserialise exactly [`SUPER_ROOT_BYTES`] bytes. Returns a
    /// specific [`SuperRootError`] for each failure mode — magic,
    /// version, CRC mismatch, or embedded pointer error.
    pub fn decode(bytes: &[u8]) -> Result<Self, SuperRootError> {
        if bytes.len() < SUPER_ROOT_BYTES {
            return Err(SuperRootError::Truncated { got: bytes.len() });
        }
        let magic: [u8; 4] = bytes[0..4].try_into().unwrap();
        if &magic != MAGIC {
            return Err(SuperRootError::BadMagic { found: magic });
        }
        let version = bytes[4];
        if version != VERSION {
            return Err(SuperRootError::UnsupportedVersion(version));
        }

        let stored_crc = u32::from_le_bytes(bytes[GENERATION_END..CRC_END].try_into().unwrap());
        let computed_crc = crc_of(&bytes[0..GENERATION_END]);
        if computed_crc != stored_crc {
            return Err(SuperRootError::BadChecksum {
                computed: computed_crc,
                stored: stored_crc,
            });
        }

        let root_dir_inode = Pointer::decode(&bytes[HEADER_END..ROOT_DIR_END])?;
        let dedup_index_inode = Pointer::decode(&bytes[ROOT_DIR_END..DEDUP_END])?;
        let dirty_bitmap_inode = Pointer::decode(&bytes[DEDUP_END..DIRTY_END])?;
        let free_run_state_inode = Pointer::decode(&bytes[DIRTY_END..FREELIST_END])?;
        let ceiling_summary_inode = Pointer::decode(&bytes[FREELIST_END..CEILING_END])?;
        let generation = u64::from_le_bytes(bytes[CEILING_END..GENERATION_END].try_into().unwrap());

        Ok(Self {
            root_dir_inode,
            dedup_index_inode,
            dirty_bitmap_inode,
            free_run_state_inode,
            ceiling_summary_inode,
            generation,
        })
    }
}

fn crc_of(bytes: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(bytes);
    h.finalize()
}
