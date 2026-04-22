//! V2 super-root record.
//!
//! The super-root is the inode-like structure the anchor points at
//! after init / every commit. It carries pointers to every other V2
//! metadata structure — root directory inode, dedup index, dirty
//! bitmap, free-run state, ceiling-magnitude bucket summary,
//! salience inode (B1 calibration) — plus a generation counter
//! that mirrors the anchor's.
//!
//! ## Layout (v2, 116 bytes, little-endian throughout)
//!
//! ```text
//! offset size  field
//! 0      4     magic = b"V2SR"
//! 4      1     version = 2
//! 5      3     reserved (zero)
//! 8      16    root_dir_inode        Pointer
//! 24     16    dedup_index_inode     Pointer
//! 40     16    dirty_bitmap_inode    Pointer
//! 56     16    free_run_state_inode  Pointer
//! 72     16    ceiling_summary_inode Pointer
//! 88     16    salience_inode        Pointer     (new in v2; NULL on pre-B1 covers)
//! 104    8     generation u64
//! 112    4     crc32 (covers bytes 0..112)
//! 116
//! ```
//!
//! ## Legacy v1 layout (100 bytes)
//!
//! Prior to B2, the super-root was 100 bytes and had no
//! `salience_inode` slot. `SuperRoot::decode` still reads those
//! records — the version byte discriminates — populating
//! `salience_inode` with `Pointer::NULL`. The first commit on such
//! a cover re-serializes the super-root in v2 format.
//!
//! Super-root chunks are allocated + written via the V2 chunk layer;
//! the anchor's two slots point at chunks holding these records. The
//! CRC guards against corruption inside the content bytes (pointer
//! updates, generation bumps) — the anchor's own per-slot CRC only
//! covers the anchor pointer to the super-root, not the super-root's
//! contents.

use thiserror::Error;

use crate::v2::pointer::{Pointer, PointerError};

pub const MAGIC: &[u8; 4] = b"V2SR";

/// Current super-root format version. Incremented to 2 in B2 when
/// `salience_inode` was added; see [`SuperRoot::decode`] for the
/// v1 → v2 migration path.
pub const VERSION: u8 = 2;

/// Legacy (pre-B2) version. Kept for the backward-compat decode
/// branch; never written.
pub const V1_VERSION: u8 = 1;

/// v1 encoded size: 4 magic + 1 version + 3 reserved + 5×16 pointers + 8 generation + 4 crc = 100.
pub const SUPER_ROOT_V1_BYTES: usize = 4 + 1 + 3 + 5 * Pointer::SIZE + 8 + 4;

/// v2 encoded size: v1 + one additional 16-byte pointer slot for
/// `salience_inode`. 116 bytes.
pub const SUPER_ROOT_BYTES: usize = SUPER_ROOT_V1_BYTES + Pointer::SIZE;

const HEADER_END: usize = 8;
const ROOT_DIR_END: usize = HEADER_END + Pointer::SIZE; // 24
const DEDUP_END: usize = ROOT_DIR_END + Pointer::SIZE; // 40
const DIRTY_END: usize = DEDUP_END + Pointer::SIZE; // 56
const FREELIST_END: usize = DIRTY_END + Pointer::SIZE; // 72
const CEILING_END: usize = FREELIST_END + Pointer::SIZE; // 88

// v1 layout: generation + crc follow immediately after CEILING_END.
const V1_GENERATION_END: usize = CEILING_END + 8; // 96
const V1_CRC_END: usize = V1_GENERATION_END + 4; // 100

// v2 layout: salience pointer sits between CEILING_END and the
// generation slot, so everything past CEILING_END shifts by 16 bytes.
const SALIENCE_END: usize = CEILING_END + Pointer::SIZE; // 104
const GENERATION_END: usize = SALIENCE_END + 8; // 112
const CRC_END: usize = GENERATION_END + 4; // 116

const _: () = {
    // Compile-time assert the layout calculations.
    assert!(V1_CRC_END == SUPER_ROOT_V1_BYTES);
    assert!(CRC_END == SUPER_ROOT_BYTES);
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuperRoot {
    pub root_dir_inode: Pointer,
    pub dedup_index_inode: Pointer,
    pub dirty_bitmap_inode: Pointer,
    pub free_run_state_inode: Pointer,
    pub ceiling_summary_inode: Pointer,
    /// Added in v2 (B1 calibration). `Pointer::NULL` on covers that
    /// haven't been calibrated yet, and on any v1 cover decoded via
    /// the backward-compat path.
    pub salience_inode: Pointer,
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
        salience_inode: Pointer::NULL,
        generation: 0,
    };

    /// Serialise to the 116-byte v2 wire form with trailing CRC32.
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
        out[CEILING_END..SALIENCE_END].copy_from_slice(&self.salience_inode.encode());
        out[SALIENCE_END..GENERATION_END].copy_from_slice(&self.generation.to_le_bytes());
        let crc = crc_of(&out[0..GENERATION_END]);
        out[GENERATION_END..CRC_END].copy_from_slice(&crc.to_le_bytes());
        out
    }

    /// Deserialise a super-root record. Accepts either the v1
    /// (100-byte) or v2 (116-byte) layout, discriminated by the
    /// version byte. v1 records always decode with `salience_inode
    /// = Pointer::NULL`; the caller is expected to re-serialize in
    /// v2 on its next commit, which will grow the chunk.
    pub fn decode(bytes: &[u8]) -> Result<Self, SuperRootError> {
        if bytes.len() < 5 {
            return Err(SuperRootError::Truncated { got: bytes.len() });
        }
        let magic: [u8; 4] = bytes[0..4].try_into().unwrap();
        if &magic != MAGIC {
            return Err(SuperRootError::BadMagic { found: magic });
        }
        match bytes[4] {
            V1_VERSION => decode_v1(bytes),
            VERSION => decode_v2(bytes),
            other => Err(SuperRootError::UnsupportedVersion(other)),
        }
    }
}

fn decode_v1(bytes: &[u8]) -> Result<SuperRoot, SuperRootError> {
    if bytes.len() < SUPER_ROOT_V1_BYTES {
        return Err(SuperRootError::Truncated { got: bytes.len() });
    }
    let stored_crc = u32::from_le_bytes(
        bytes[V1_GENERATION_END..V1_CRC_END].try_into().unwrap(),
    );
    let computed_crc = crc_of(&bytes[0..V1_GENERATION_END]);
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
    let generation = u64::from_le_bytes(
        bytes[CEILING_END..V1_GENERATION_END].try_into().unwrap(),
    );
    Ok(SuperRoot {
        root_dir_inode,
        dedup_index_inode,
        dirty_bitmap_inode,
        free_run_state_inode,
        ceiling_summary_inode,
        salience_inode: Pointer::NULL,
        generation,
    })
}

fn decode_v2(bytes: &[u8]) -> Result<SuperRoot, SuperRootError> {
    if bytes.len() < SUPER_ROOT_BYTES {
        return Err(SuperRootError::Truncated { got: bytes.len() });
    }
    let stored_crc =
        u32::from_le_bytes(bytes[GENERATION_END..CRC_END].try_into().unwrap());
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
    let salience_inode = Pointer::decode(&bytes[CEILING_END..SALIENCE_END])?;
    let generation =
        u64::from_le_bytes(bytes[SALIENCE_END..GENERATION_END].try_into().unwrap());
    Ok(SuperRoot {
        root_dir_inode,
        dedup_index_inode,
        dirty_bitmap_inode,
        free_run_state_inode,
        ceiling_summary_inode,
        salience_inode,
        generation,
    })
}

fn crc_of(bytes: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(bytes);
    h.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_v2_super_root() -> SuperRoot {
        SuperRoot {
            root_dir_inode: Pointer {
                slot: 1,
                start_weight: 100,
                length_in_bits: 800,
                ..Pointer::NULL
            },
            dedup_index_inode: Pointer {
                slot: 2,
                start_weight: 200,
                length_in_bits: 1600,
                ..Pointer::NULL
            },
            dirty_bitmap_inode: Pointer {
                slot: 3,
                start_weight: 300,
                length_in_bits: 3200,
                ..Pointer::NULL
            },
            free_run_state_inode: Pointer::NULL,
            ceiling_summary_inode: Pointer {
                slot: 5,
                start_weight: 500,
                length_in_bits: 400,
                ..Pointer::NULL
            },
            salience_inode: Pointer {
                slot: 7,
                start_weight: 777,
                length_in_bits: 1400,
                ..Pointer::NULL
            },
            generation: 42,
        }
    }

    /// Hand-construct a v1 super-root record and verify it decodes
    /// into a `SuperRoot` with `salience_inode = NULL`. This is the
    /// backward-compat gate: any cover initialized before B2 must
    /// still mount.
    #[test]
    fn v1_record_decodes_with_null_salience() {
        let mut bytes = [0u8; SUPER_ROOT_V1_BYTES];
        bytes[0..4].copy_from_slice(MAGIC);
        bytes[4] = V1_VERSION;

        let root_dir = Pointer {
            slot: 1,
            start_weight: 100,
            length_in_bits: 800,
            ..Pointer::NULL
        };
        let dedup = Pointer {
            slot: 2,
            start_weight: 200,
            length_in_bits: 1600,
            ..Pointer::NULL
        };
        let dirty = Pointer {
            slot: 3,
            start_weight: 300,
            length_in_bits: 3200,
            ..Pointer::NULL
        };
        let ceiling = Pointer {
            slot: 5,
            start_weight: 500,
            length_in_bits: 400,
            ..Pointer::NULL
        };

        bytes[HEADER_END..ROOT_DIR_END].copy_from_slice(&root_dir.encode());
        bytes[ROOT_DIR_END..DEDUP_END].copy_from_slice(&dedup.encode());
        bytes[DEDUP_END..DIRTY_END].copy_from_slice(&dirty.encode());
        bytes[DIRTY_END..FREELIST_END].copy_from_slice(&Pointer::NULL.encode());
        bytes[FREELIST_END..CEILING_END].copy_from_slice(&ceiling.encode());
        // v1: generation sits right after CEILING_END.
        bytes[CEILING_END..V1_GENERATION_END].copy_from_slice(&42_u64.to_le_bytes());
        let crc = crc_of(&bytes[0..V1_GENERATION_END]);
        bytes[V1_GENERATION_END..V1_CRC_END].copy_from_slice(&crc.to_le_bytes());

        let sr = SuperRoot::decode(&bytes).expect("v1 decode");
        assert_eq!(sr.root_dir_inode, root_dir);
        assert_eq!(sr.dedup_index_inode, dedup);
        assert_eq!(sr.dirty_bitmap_inode, dirty);
        assert_eq!(sr.free_run_state_inode, Pointer::NULL);
        assert_eq!(sr.ceiling_summary_inode, ceiling);
        assert_eq!(
            sr.salience_inode,
            Pointer::NULL,
            "v1 decode must populate salience_inode with NULL",
        );
        assert_eq!(sr.generation, 42);
    }

    /// After round-tripping a v1-decoded `SuperRoot` through
    /// `encode`, decoding the v2 bytes should return a `SuperRoot`
    /// byte-identical in every field to the original *except* the
    /// now-populated salience slot (which was NULL on v1 anyway).
    #[test]
    fn v1_decode_encode_roundtrips_as_v2() {
        // Build a v1 record (via the same code as above).
        let mut bytes = [0u8; SUPER_ROOT_V1_BYTES];
        bytes[0..4].copy_from_slice(MAGIC);
        bytes[4] = V1_VERSION;
        let ptr = Pointer {
            slot: 9,
            start_weight: 12345,
            length_in_bits: 800,
            ..Pointer::NULL
        };
        bytes[HEADER_END..ROOT_DIR_END].copy_from_slice(&ptr.encode());
        bytes[ROOT_DIR_END..DEDUP_END].copy_from_slice(&Pointer::NULL.encode());
        bytes[DEDUP_END..DIRTY_END].copy_from_slice(&Pointer::NULL.encode());
        bytes[DIRTY_END..FREELIST_END].copy_from_slice(&Pointer::NULL.encode());
        bytes[FREELIST_END..CEILING_END].copy_from_slice(&Pointer::NULL.encode());
        bytes[CEILING_END..V1_GENERATION_END].copy_from_slice(&7_u64.to_le_bytes());
        let crc = crc_of(&bytes[0..V1_GENERATION_END]);
        bytes[V1_GENERATION_END..V1_CRC_END].copy_from_slice(&crc.to_le_bytes());

        let sr = SuperRoot::decode(&bytes).expect("v1 decode");
        let encoded_v2 = sr.encode();
        assert_eq!(encoded_v2.len(), SUPER_ROOT_BYTES);
        assert_eq!(encoded_v2[4], VERSION, "re-encode should emit v2");

        let round = SuperRoot::decode(&encoded_v2).expect("v2 decode");
        assert_eq!(round, sr, "v1→decode→encode→decode should be identity");
    }

    /// End-to-end v2 round trip for a non-null salience pointer.
    #[test]
    fn v2_roundtrip_preserves_salience_pointer() {
        let sr = sample_v2_super_root();
        let bytes = sr.encode();
        assert_eq!(bytes.len(), SUPER_ROOT_BYTES);
        assert_eq!(bytes[4], VERSION);
        let decoded = SuperRoot::decode(&bytes).expect("v2 decode");
        assert_eq!(decoded, sr);
        assert!(!decoded.salience_inode.is_null());
    }

    #[test]
    fn bad_magic_rejected() {
        let mut bytes = [0u8; SUPER_ROOT_BYTES];
        bytes[0..4].copy_from_slice(b"XXXX");
        bytes[4] = VERSION;
        assert!(matches!(
            SuperRoot::decode(&bytes),
            Err(SuperRootError::BadMagic { .. }),
        ));
    }

    #[test]
    fn unknown_version_rejected() {
        let mut bytes = [0u8; SUPER_ROOT_BYTES];
        bytes[0..4].copy_from_slice(MAGIC);
        bytes[4] = 99;
        assert!(matches!(
            SuperRoot::decode(&bytes),
            Err(SuperRootError::UnsupportedVersion(99)),
        ));
    }

    #[test]
    fn v2_with_corrupt_crc_rejected() {
        let mut bytes = sample_v2_super_root().encode();
        bytes[GENERATION_END] ^= 0xAB;
        assert!(matches!(
            SuperRoot::decode(&bytes),
            Err(SuperRootError::BadChecksum { .. }),
        ));
    }

    #[test]
    fn v1_with_corrupt_crc_rejected() {
        // Build a v1 record, corrupt its CRC, and verify decode
        // path v1 still runs its own CRC check.
        let mut bytes = [0u8; SUPER_ROOT_V1_BYTES];
        bytes[0..4].copy_from_slice(MAGIC);
        bytes[4] = V1_VERSION;
        bytes[HEADER_END..ROOT_DIR_END].copy_from_slice(&Pointer::NULL.encode());
        bytes[ROOT_DIR_END..DEDUP_END].copy_from_slice(&Pointer::NULL.encode());
        bytes[DEDUP_END..DIRTY_END].copy_from_slice(&Pointer::NULL.encode());
        bytes[DIRTY_END..FREELIST_END].copy_from_slice(&Pointer::NULL.encode());
        bytes[FREELIST_END..CEILING_END].copy_from_slice(&Pointer::NULL.encode());
        bytes[CEILING_END..V1_GENERATION_END].copy_from_slice(&0_u64.to_le_bytes());
        bytes[V1_GENERATION_END..V1_CRC_END]
            .copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        assert!(matches!(
            SuperRoot::decode(&bytes),
            Err(SuperRootError::BadChecksum { .. }),
        ));
    }

    #[test]
    fn truncated_below_header_rejected() {
        let bytes = [0u8; 3];
        assert!(matches!(
            SuperRoot::decode(&bytes),
            Err(SuperRootError::Truncated { got: 3 }),
        ));
    }

    #[test]
    fn truncated_v2_buffer_rejected() {
        // 100-byte buffer with v2 magic + version byte — only long
        // enough for a v1 record, but the version byte says v2.
        let mut bytes = [0u8; SUPER_ROOT_V1_BYTES];
        bytes[0..4].copy_from_slice(MAGIC);
        bytes[4] = VERSION;
        assert!(matches!(
            SuperRoot::decode(&bytes),
            Err(SuperRootError::Truncated { got: 100 }),
        ));
    }
}
