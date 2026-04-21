//! V2 anchor — rule-derived bootstrap region (DESIGN-NEW §15.2).
//!
//! The anchor is the only piece of V2 metadata whose location isn't
//! stored somewhere else; it's **derived from the cover's structure**
//! via a write-invariant rule:
//!
//! > The anchor occupies the N lowest-ceiling-magnitude stealable-bit
//! > positions of the cover, sorted by `(slot_index, weight_index)`
//! > for canonical order. N is the anchor payload in bits = 512.
//!
//! Ceiling magnitude depends only on the non-stealable bits of each
//! weight (see `read_weight_ceiling_abs` in
//! [`crate::stego::calibration::magnitude`]), so the ranking is
//! invariant under any write V2 makes. That's the property
//! findability leans on.
//!
//! # Record layout (64 bytes / 512 bits)
//!
//! ```text
//! offset  size  field
//! 0       4     magic = b"LLMV"
//! 4       1     version = 2
//! 5       3     reserved (zero)
//! 8       28    slot 0: (generation u64 | super_root 16 bytes | crc32 u32)
//! 36      28    slot 1: (same shape)
//! ```
//!
//! The two-slot generational scheme is V2's atomic-commit primitive:
//! readers pick the slot with higher valid generation, writers always
//! overwrite the *inactive* slot, so a reader always sees a committed
//! (pre- or post-) state regardless of when a concurrent writer
//! crashed.

use thiserror::Error;

use crate::stego::calibration::byte_io::{OutOfBoundsError, read_bytes, write_bytes};
use crate::stego::calibration::magnitude::read_weight_ceiling_abs;
use crate::stego::calibration::placement::{MetadataBitPos, MetadataPlacement};
use crate::stego::calibration::{WeightRef, stealable_bits_for};
use crate::stego::tensor_map::TensorMap;
use crate::v2::pointer::{Pointer, PointerError};

/// Anchor header magic bytes.
pub const MAGIC: &[u8; 4] = b"LLMV";
/// Anchor format version.
pub const VERSION: u8 = 2;
/// Bytes per slot record: `u64 generation + Pointer(16) + u32 crc`.
pub const SLOT_BYTES: usize = 8 + Pointer::SIZE + 4;
/// Header bytes: magic + version + reserved = 8.
pub const HEADER_BYTES: usize = 4 + 1 + 3;
/// Full anchor record size: 64 bytes = 512 bits.
pub const ANCHOR_BYTES: usize = HEADER_BYTES + 2 * SLOT_BYTES;
/// Full anchor record size in stealable bits.
pub const ANCHOR_BITS: u64 = (ANCHOR_BYTES * 8) as u64;

/// Which of the two root slots is active / target of a commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SlotIndex {
    Slot0,
    Slot1,
}

impl SlotIndex {
    /// The other slot — a commit writes here.
    pub fn other(self) -> Self {
        match self {
            Self::Slot0 => Self::Slot1,
            Self::Slot1 => Self::Slot0,
        }
    }

    fn byte_offset(self) -> u64 {
        (HEADER_BYTES
            + match self {
                Self::Slot0 => 0,
                Self::Slot1 => SLOT_BYTES,
            }) as u64
    }
}

/// One of the two generational root records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnchorSlot {
    pub generation: u64,
    pub super_root: Pointer,
    pub crc32: u32,
}

impl AnchorSlot {
    /// Build a slot with a freshly-computed CRC for the given
    /// generation + pointer.
    pub fn with_pointer(generation: u64, super_root: Pointer) -> Self {
        let crc32 = compute_crc(generation, &super_root);
        Self {
            generation,
            super_root,
            crc32,
        }
    }

    /// CRC matches the generation + super_root pair.
    pub fn is_valid(&self) -> bool {
        self.crc32 == compute_crc(self.generation, &self.super_root)
    }

    fn encode(&self) -> [u8; SLOT_BYTES] {
        let mut out = [0u8; SLOT_BYTES];
        out[0..8].copy_from_slice(&self.generation.to_le_bytes());
        out[8..8 + Pointer::SIZE].copy_from_slice(&self.super_root.encode());
        out[8 + Pointer::SIZE..SLOT_BYTES].copy_from_slice(&self.crc32.to_le_bytes());
        out
    }

    fn decode(bytes: &[u8; SLOT_BYTES]) -> Result<Self, AnchorError> {
        let generation = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let super_root = Pointer::decode(&bytes[8..8 + Pointer::SIZE])?;
        let crc32 =
            u32::from_le_bytes(bytes[8 + Pointer::SIZE..SLOT_BYTES].try_into().unwrap());
        Ok(Self {
            generation,
            super_root,
            crc32,
        })
    }
}

/// Result of reading the anchor — the active slot (higher generation,
/// valid CRC) plus which slot it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnchorReadOutcome {
    pub active_slot: SlotIndex,
    pub active: AnchorSlot,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum AnchorError {
    #[error("bad magic: expected b\"LLMV\", got {found:?}")]
    BadMagic { found: [u8; 4] },

    #[error("unsupported version: {0}")]
    UnsupportedVersion(u8),

    #[error("neither anchor slot is valid (both failed CRC)")]
    NoValidAnchor,

    #[error("cover too small for anchor placement (needs {needed} bits, has {available})")]
    CoverTooSmall { needed: u64, available: u64 },

    #[error("pointer decode failed in anchor slot: {source}")]
    PointerDecode {
        #[from]
        source: PointerError,
    },

    #[error("commit generation mismatch: caller expected {expected}, on-disk is {found}")]
    GenerationMismatch { expected: u64, found: u64 },

    #[error("byte_io error: {source}")]
    ByteIo {
        #[from]
        source: OutOfBoundsError,
    },
}

fn compute_crc(generation: u64, super_root: &Pointer) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&generation.to_le_bytes());
    hasher.update(&super_root.encode());
    hasher.finalize()
}

/// Compute the anchor's bit positions in the cover. The positions are
/// determined by:
///
/// 1. Rank every eligible weight by its ceiling magnitude (ascending).
/// 2. Select weights until their combined stealable-bit capacity
///    covers [`ANCHOR_BITS`].
/// 3. Sort the selected set by `WeightRef` (canonical order —
///    invariant under magnitude shifts within the set).
/// 4. Enumerate stealable bit positions in order: for each selected
///    weight, bits `0..stealable` in ascending order.
///
/// Cost: `O(total_eligible_weights)` for the ceiling-magnitude scan
/// plus `O(n log n)` for the sort. Anchor-finding runs at `llmdb
/// init` and once per mount; not a hot path.
pub fn find_anchor_placement(mmap: &[u8], map: &TensorMap) -> MetadataPlacement {
    find_placement_for_bits(mmap, map, ANCHOR_BITS)
}

fn find_placement_for_bits(
    mmap: &[u8],
    map: &TensorMap,
    needed_bits: u64,
) -> MetadataPlacement {
    let mut candidates: Vec<(f32, WeightRef)> = Vec::new();
    for (slot_idx, slot) in map.slots.iter().enumerate() {
        if stealable_bits_for(slot.quant_type) == 0 {
            continue;
        }
        for weight_index in 0..slot.weight_count {
            let c = read_weight_ceiling_abs(mmap, slot, weight_index);
            candidates.push((
                c,
                WeightRef {
                    slot_index: slot_idx as u32,
                    weight_index,
                },
            ));
        }
    }

    candidates.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });

    let mut selected: Vec<WeightRef> = Vec::new();
    let mut accumulated_bits: u64 = 0;
    for (_, wref) in candidates {
        let slot = &map.slots[wref.slot_index as usize];
        let bits = stealable_bits_for(slot.quant_type) as u64;
        accumulated_bits = accumulated_bits.saturating_add(bits);
        selected.push(wref);
        if accumulated_bits >= needed_bits {
            break;
        }
    }

    // Canonical order within the selection — invariant under
    // magnitude perturbations (bits sort within each weight, weights
    // sort by (slot_index, weight_index)).
    selected.sort_unstable();

    let mut positions: Vec<MetadataBitPos> = Vec::with_capacity(accumulated_bits as usize);
    for wref in selected {
        let slot = &map.slots[wref.slot_index as usize];
        let bits = stealable_bits_for(slot.quant_type);
        for bit_index in 0..bits {
            positions.push(MetadataBitPos {
                slot_index: wref.slot_index,
                weight_index: wref.weight_index,
                bit_index: bit_index as u8,
            });
        }
    }

    MetadataPlacement { positions }
}

/// Write the initial anchor record on a fresh cover. Both slots are
/// populated — slot 0 with `generation = 0`, slot 1 with `generation
/// = 1` — so readers see a valid gen-1 active slot immediately.
pub fn init_anchor(
    mmap: &mut [u8],
    map: &TensorMap,
    super_root: Pointer,
) -> Result<(), AnchorError> {
    let placement = find_anchor_placement(mmap, map);
    if (placement.positions.len() as u64) < ANCHOR_BITS {
        return Err(AnchorError::CoverTooSmall {
            needed: ANCHOR_BITS,
            available: placement.positions.len() as u64,
        });
    }

    let mut buf = [0u8; ANCHOR_BYTES];
    buf[0..4].copy_from_slice(MAGIC);
    buf[4] = VERSION;
    // buf[5..8] stays zero (reserved).

    let slot0 = AnchorSlot::with_pointer(0, super_root);
    let slot1 = AnchorSlot::with_pointer(1, super_root);
    buf[HEADER_BYTES..HEADER_BYTES + SLOT_BYTES].copy_from_slice(&slot0.encode());
    buf[HEADER_BYTES + SLOT_BYTES..HEADER_BYTES + 2 * SLOT_BYTES]
        .copy_from_slice(&slot1.encode());

    write_bytes(mmap, map, &placement, 0, &buf)?;
    Ok(())
}

/// Read the anchor and return its active slot. Errors if neither slot
/// has a valid CRC (or if the header magic / version is wrong).
pub fn read_anchor(
    mmap: &[u8],
    map: &TensorMap,
) -> Result<AnchorReadOutcome, AnchorError> {
    let placement = find_anchor_placement(mmap, map);
    if (placement.positions.len() as u64) < ANCHOR_BITS {
        return Err(AnchorError::CoverTooSmall {
            needed: ANCHOR_BITS,
            available: placement.positions.len() as u64,
        });
    }

    let mut buf = [0u8; ANCHOR_BYTES];
    read_bytes(mmap, map, &placement, 0, &mut buf)?;

    let magic: [u8; 4] = buf[0..4].try_into().unwrap();
    if &magic != MAGIC {
        return Err(AnchorError::BadMagic { found: magic });
    }
    let version = buf[4];
    if version != VERSION {
        return Err(AnchorError::UnsupportedVersion(version));
    }

    let slot0_bytes: [u8; SLOT_BYTES] = buf[HEADER_BYTES..HEADER_BYTES + SLOT_BYTES]
        .try_into()
        .unwrap();
    let slot1_bytes: [u8; SLOT_BYTES] = buf
        [HEADER_BYTES + SLOT_BYTES..HEADER_BYTES + 2 * SLOT_BYTES]
        .try_into()
        .unwrap();
    let slot0 = AnchorSlot::decode(&slot0_bytes)?;
    let slot1 = AnchorSlot::decode(&slot1_bytes)?;

    match (slot0.is_valid(), slot1.is_valid()) {
        (true, true) => {
            if slot0.generation >= slot1.generation {
                Ok(AnchorReadOutcome {
                    active_slot: SlotIndex::Slot0,
                    active: slot0,
                })
            } else {
                Ok(AnchorReadOutcome {
                    active_slot: SlotIndex::Slot1,
                    active: slot1,
                })
            }
        }
        (true, false) => Ok(AnchorReadOutcome {
            active_slot: SlotIndex::Slot0,
            active: slot0,
        }),
        (false, true) => Ok(AnchorReadOutcome {
            active_slot: SlotIndex::Slot1,
            active: slot1,
        }),
        (false, false) => Err(AnchorError::NoValidAnchor),
    }
}

/// Commit a new super-root pointer. Writes to the *inactive* slot
/// (the one with the lower current generation) with
/// `generation = prev_generation + 1`. Returns the new generation on
/// success. Returns [`AnchorError::GenerationMismatch`] if the caller
/// believed a stale generation was active.
pub fn commit_anchor(
    mmap: &mut [u8],
    map: &TensorMap,
    super_root: Pointer,
    prev_generation: u64,
) -> Result<u64, AnchorError> {
    let outcome = read_anchor(mmap, map)?;
    if outcome.active.generation != prev_generation {
        return Err(AnchorError::GenerationMismatch {
            expected: prev_generation,
            found: outcome.active.generation,
        });
    }
    let new_gen = prev_generation + 1;
    let target_slot = outcome.active_slot.other();
    let new_slot = AnchorSlot::with_pointer(new_gen, super_root);

    // Write the slot bytes at their offset within the anchor record.
    let placement = find_anchor_placement(mmap, map);
    write_bytes(
        mmap,
        map,
        &placement,
        target_slot.byte_offset(),
        &new_slot.encode(),
    )?;
    Ok(new_gen)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_bytes_is_sixty_four() {
        assert_eq!(ANCHOR_BYTES, 64);
        assert_eq!(ANCHOR_BITS, 512);
    }

    #[test]
    fn slot_index_other_is_involutory() {
        assert_eq!(SlotIndex::Slot0.other(), SlotIndex::Slot1);
        assert_eq!(SlotIndex::Slot1.other(), SlotIndex::Slot0);
        assert_eq!(SlotIndex::Slot0.other().other(), SlotIndex::Slot0);
    }

    #[test]
    fn slot_byte_offsets() {
        assert_eq!(SlotIndex::Slot0.byte_offset(), HEADER_BYTES as u64);
        assert_eq!(
            SlotIndex::Slot1.byte_offset(),
            (HEADER_BYTES + SLOT_BYTES) as u64
        );
    }

    #[test]
    fn slot_encode_decode_round_trips() {
        let slot = AnchorSlot::with_pointer(
            42,
            Pointer {
                slot: 3,
                start_weight: 0x12345678,
                length_in_bits: 1024,
                flags: 0,
                reserved: 0,
            },
        );
        let bytes = slot.encode();
        let decoded = AnchorSlot::decode(&bytes).expect("decode");
        assert_eq!(decoded, slot);
        assert!(decoded.is_valid());
    }

    #[test]
    fn empty_slot_is_not_valid() {
        // An all-zero slot has crc=0 but the crc of gen=0 + NULL
        // pointer isn't zero → invalid by CRC.
        let bytes = [0u8; SLOT_BYTES];
        let decoded = AnchorSlot::decode(&bytes).expect("decode zero");
        assert!(!decoded.is_valid());
    }
}
