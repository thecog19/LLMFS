use thiserror::Error;

pub const ENTRIES_PER_BLOCK: usize = 1022;
const HEADER_BYTES: usize = 8;

/// High bit of a redirection entry marks "logical has been written at least
/// once." Entries stored as `physical | WRITTEN_BIT` when written; a bare
/// identity entry (`entries[L] == L`, high bit clear) means "never written".
/// Physical indices are capped at 2^31 − 1 blocks (~8 TB at 4 KB).
pub const WRITTEN_BIT: u32 = 0x8000_0000;
pub const PHYSICAL_MASK: u32 = 0x7FFF_FFFF;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedirectionTable {
    blocks: Vec<RedirectionTableBlock>,
    total_entries: u32,
}

impl RedirectionTable {
    pub fn identity(total_blocks: u32) -> Self {
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
            let entries: Vec<u32> = (cursor..cursor + count).collect();
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

    /// Returns the physical block holding `logical`'s data, stripping the
    /// written flag. For an identity entry this is `logical` itself.
    pub fn logical_to_physical(&self, logical: u32) -> Option<u32> {
        self.raw_entry(logical)
            .map(|entry| entry & PHYSICAL_MASK)
    }

    /// True iff the entry has been explicitly set via `set_mapping` (either
    /// marking a direct first write or recording a shadow redirect). Used by
    /// the device to distinguish "never written" identity from "written at
    /// identity" for crash recovery and to decide direct vs shadow on write.
    pub fn is_written(&self, logical: u32) -> bool {
        self.raw_entry(logical)
            .map(|entry| entry & WRITTEN_BIT != 0)
            .unwrap_or(false)
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

    /// Record that `logical` maps to `physical`, marking it as written.
    /// Physical values with the high bit set are rejected — the high bit is
    /// reserved for the written flag.
    pub fn set_mapping(&mut self, logical: u32, physical: u32) {
        debug_assert!(
            physical & WRITTEN_BIT == 0,
            "physical index {physical:#x} exceeds 31-bit capacity"
        );
        let block_index = logical as usize / ENTRIES_PER_BLOCK;
        let entry_index = logical as usize % ENTRIES_PER_BLOCK;
        if let Some(block) = self.blocks.get_mut(block_index) {
            if entry_index < block.entries.len() {
                block.entries[entry_index] = (physical & PHYSICAL_MASK) | WRITTEN_BIT;
            }
        }
    }

    /// Reset `logical` to unwritten identity (`entries[logical] == logical`,
    /// no written flag). Used on `free_block` so the slot can be re-allocated
    /// cleanly.
    pub fn clear(&mut self, logical: u32) {
        let block_index = logical as usize / ENTRIES_PER_BLOCK;
        let entry_index = logical as usize % ENTRIES_PER_BLOCK;
        if let Some(block) = self.blocks.get_mut(block_index) {
            if entry_index < block.entries.len() {
                block.entries[entry_index] = logical;
            }
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
