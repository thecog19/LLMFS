use thiserror::Error;

use crate::stego::integrity::NO_BLOCK;

pub const ENTRIES_PER_BLOCK: usize = 1022;
const HEADER_BYTES: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedirectionBootstrap {
    pub entries_per_block: usize,
}

impl Default for RedirectionBootstrap {
    fn default() -> Self {
        Self {
            entries_per_block: ENTRIES_PER_BLOCK,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedirectionTableBlock {
    pub first_logical_block: u32,
    pub entry_count: u32,
    pub entries: Vec<u32>,
}

impl RedirectionTableBlock {
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = vec![0_u8; crate::BLOCK_SIZE];
        bytes[0..4].copy_from_slice(&self.first_logical_block.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.entry_count.to_le_bytes());
        for (index, entry) in self.entries.iter().copied().enumerate() {
            let offset = HEADER_BYTES + index * 4;
            bytes[offset..offset + 4].copy_from_slice(&entry.to_le_bytes());
        }
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, RedirectionError> {
        if bytes.len() != crate::BLOCK_SIZE {
            return Err(RedirectionError::InvalidBlockLength {
                expected: crate::BLOCK_SIZE,
                actual: bytes.len(),
            });
        }

        let first_logical_block = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let entry_count = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let entry_len = entry_count as usize;

        if entry_len > ENTRIES_PER_BLOCK {
            return Err(RedirectionError::EntryCountOverflow {
                max: ENTRIES_PER_BLOCK,
                actual: entry_len,
            });
        }

        let mut entries = Vec::with_capacity(entry_len);
        for index in 0..entry_len {
            let offset = HEADER_BYTES + index * 4;
            entries.push(u32::from_le_bytes(
                bytes[offset..offset + 4].try_into().unwrap(),
            ));
        }

        Ok(Self {
            first_logical_block,
            entry_count,
            entries,
        })
    }
}

/// Maps logical block indices to physical block indices, with `NO_BLOCK`
/// (= `u32::MAX`) as the sentinel for "logical not currently bound to any
/// physical." Logical and physical address spaces are intentionally
/// separate: a redirection-table index is *only* a logical key, never an
/// implicit physical address. The free list and any direct-addressing
/// client (NBD) operate on physicals; the redirection layer is the
/// translation in between.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedirectionTable {
    blocks: Vec<RedirectionTableBlock>,
    total_entries: u32,
}

impl RedirectionTable {
    /// Build an empty redirection table sized for `total_blocks` logical
    /// entries — every entry is `NO_BLOCK` (unmapped). Replaces the older
    /// "identity" initializer that aliased logical L to physical L by
    /// default; that aliasing is what made shadow-copy collide with
    /// direct-addressing workloads.
    pub fn empty(total_blocks: u32) -> Self {
        if total_blocks == 0 {
            return Self {
                blocks: Vec::new(),
                total_entries: 0,
            };
        }

        let block_count = (total_blocks as usize).div_ceil(ENTRIES_PER_BLOCK);
        let mut blocks = Vec::with_capacity(block_count);
        let mut remaining = total_blocks;
        let mut cursor = 0_u32;

        for _ in 0..block_count {
            let count = remaining.min(ENTRIES_PER_BLOCK as u32);
            let entries: Vec<u32> = vec![NO_BLOCK; count as usize];
            blocks.push(RedirectionTableBlock {
                first_logical_block: cursor,
                entry_count: count,
                entries,
            });
            cursor += count;
            remaining -= count;
        }

        Self {
            blocks,
            total_entries: total_blocks,
        }
    }

    /// Returns the physical block holding `logical`'s data, or `None` when
    /// `logical` is unmapped (never written, or freed). Reads of unmapped
    /// logicals at the device layer present as zeros.
    pub fn logical_to_physical(&self, logical: u32) -> Option<u32> {
        let raw = self.raw_entry(logical)?;
        if raw == NO_BLOCK { None } else { Some(raw) }
    }

    /// True iff `logical` currently maps to a physical (i.e., has user data
    /// somewhere). The complement of "unmapped".
    pub fn is_mapped(&self, logical: u32) -> bool {
        self.logical_to_physical(logical).is_some()
    }

    fn raw_entry(&self, logical: u32) -> Option<u32> {
        if logical >= self.total_entries {
            return None;
        }
        let block_index = logical as usize / ENTRIES_PER_BLOCK;
        let entry_index = logical as usize % ENTRIES_PER_BLOCK;
        self.blocks
            .get(block_index)
            .and_then(|block| block.entries.get(entry_index))
            .copied()
    }

    /// Bind `logical` to `physical`. `physical` must be a valid physical
    /// block index (not `NO_BLOCK`). To unbind use `clear`.
    pub fn set_mapping(&mut self, logical: u32, physical: u32) {
        debug_assert!(
            physical != NO_BLOCK,
            "set_mapping called with NO_BLOCK; use clear() to unmap"
        );
        let block_index = logical as usize / ENTRIES_PER_BLOCK;
        let entry_index = logical as usize % ENTRIES_PER_BLOCK;
        if let Some(block) = self.blocks.get_mut(block_index)
            && entry_index < block.entries.len()
        {
            block.entries[entry_index] = physical;
        }
    }

    /// Mark `logical` as unmapped — `logical_to_physical` will return `None`
    /// afterwards. Used by `free_block` and shadow-copy phase 3 cleanup.
    pub fn clear(&mut self, logical: u32) {
        let block_index = logical as usize / ENTRIES_PER_BLOCK;
        let entry_index = logical as usize % ENTRIES_PER_BLOCK;
        if let Some(block) = self.blocks.get_mut(block_index)
            && entry_index < block.entries.len()
        {
            block.entries[entry_index] = NO_BLOCK;
        }
    }

    pub fn encode(&self) -> Vec<Vec<u8>> {
        self.blocks.iter().map(|block| block.encode()).collect()
    }

    pub fn decode(raw_blocks: &[Vec<u8>]) -> Result<Self, RedirectionError> {
        let mut blocks = Vec::with_capacity(raw_blocks.len());
        let mut total_entries = 0_u32;

        for raw in raw_blocks {
            let block = RedirectionTableBlock::decode(raw)?;
            total_entries += block.entry_count;
            blocks.push(block);
        }

        Ok(Self {
            blocks,
            total_entries,
        })
    }

    pub fn total_entries(&self) -> u32 {
        self.total_entries
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RedirectionError {
    #[error("invalid redirection block length: expected {expected}, got {actual}")]
    InvalidBlockLength { expected: usize, actual: usize },
    #[error("redirection entry count {actual} exceeds max {max}")]
    EntryCountOverflow { max: usize, actual: usize },
}
