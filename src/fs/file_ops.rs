use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crc32fast::Hasher;
use thiserror::Error;

use crate::fs::file_table::{
    ENTRIES_PER_BLOCK, FLAG_DELETED, FileEntry, FileEntryType, FileTableBlock, FileTableError,
    MAX_FILENAME_BYTES, MAX_INLINE_BLOCKS, OVERFLOW_ENTRIES_PER_BLOCK, OverflowBlock,
};
use crate::stego::device::{DeviceError, StegoDevice};
use crate::stego::integrity::NO_BLOCK;

#[derive(Debug, Error)]
pub enum FsError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("device error: {0}")]
    Device(#[from] DeviceError),
    #[error("file table error: {0}")]
    Table(#[from] FileTableError),
    #[error("file not found: {0}")]
    FileNotFound(String),
    #[error("file too large: need {needed} blocks, only {available} free")]
    FileTooLarge { needed: u32, available: u32 },
    #[error("duplicate file name: {0}")]
    DuplicateName(String),
    #[error("invalid file name: {reason}")]
    InvalidFilename { reason: &'static str },
    #[error("file table is full (capacity {capacity})")]
    TableFull { capacity: usize },
    #[error("file crc32 mismatch for {name}: expected {expected:#010x}, got {actual:#010x}")]
    Crc32Mismatch {
        name: String,
        expected: u32,
        actual: u32,
    },
}

/// Location of a file-table entry on disk: `(file_table_block_index, slot_within_block)`.
#[derive(Debug, Clone, Copy)]
struct EntryLocation {
    block_index: u32,
    slot: usize,
}

impl StegoDevice {
    pub fn store_file(
        &mut self,
        host_path: &Path,
        stego_name: &str,
        mode: u16,
    ) -> Result<FileEntry, FsError> {
        validate_filename(stego_name)?;

        let data = std::fs::read(host_path)?;
        self.store_bytes(&data, stego_name, mode)
    }

    /// Variant used by tests and internal callers that already have the
    /// payload in memory.
    pub fn store_bytes(
        &mut self,
        data: &[u8],
        stego_name: &str,
        mode: u16,
    ) -> Result<FileEntry, FsError> {
        validate_filename(stego_name)?;

        if self.find_entry_by_name(stego_name)?.is_some() {
            return Err(FsError::DuplicateName(stego_name.to_owned()));
        }

        let block_count = blocks_for_size(data.len() as u64);
        let overflow_count = overflow_blocks_for_count(block_count);
        let needed = block_count.saturating_add(overflow_count);

        let free = self.free_blocks()?;
        if free < needed {
            return Err(FsError::FileTooLarge {
                needed,
                available: free,
            });
        }

        let slot = self.find_free_slot()?.ok_or(FsError::TableFull {
            capacity: self.file_table_capacity(),
        })?;

        let data_blocks = self.alloc_blocks(block_count)?;
        let overflow_blocks = match self.alloc_blocks(overflow_count) {
            Ok(v) => v,
            Err(err) => {
                // Pre-check should have prevented this, but be defensive:
                // roll back any data block allocations so a failed store is
                // atomic.
                for b in &data_blocks {
                    let _ = self.free_block(*b);
                }
                return Err(err);
            }
        };

        let mut crc = Hasher::new();
        crc.update(data);
        let crc32 = crc.finalize();

        // Write file data to allocated blocks. Any tail bytes shorter than a
        // full block are zero-padded inside the block (§6: every data block
        // is a full 4096-byte page).
        for (i, &logical) in data_blocks.iter().enumerate() {
            let chunk_start = i * crate::BLOCK_SIZE;
            let chunk_end = (chunk_start + crate::BLOCK_SIZE).min(data.len());
            let mut buf = vec![0_u8; crate::BLOCK_SIZE];
            if chunk_start < chunk_end {
                buf[..chunk_end - chunk_start].copy_from_slice(&data[chunk_start..chunk_end]);
            }
            self.write_block(logical, &buf)?;
        }

        // Fill overflow blocks with the tail of the block map.
        if !overflow_blocks.is_empty() {
            let tail = &data_blocks[MAX_INLINE_BLOCKS..];
            self.write_overflow_chain(&overflow_blocks, tail)?;
        }

        let inline_len = (block_count as usize).min(MAX_INLINE_BLOCKS);
        let inline_blocks = data_blocks[..inline_len].to_vec();
        let overflow_block = overflow_blocks.first().copied().unwrap_or(NO_BLOCK);

        let now = unix_now();
        let entry = FileEntry {
            entry_type: FileEntryType::Regular,
            flags: 0,
            mode,
            uid: 0,
            gid: 0,
            size_bytes: data.len() as u64,
            created: now,
            modified: now,
            crc32,
            block_count,
            overflow_block,
            filename: stego_name.to_owned(),
            inline_blocks,
        };

        // Commit point: the file-table entry write. Crash before this and
        // orphan scan reclaims every block we allocated.
        self.write_entry_at(slot, &entry)?;

        Ok(entry)
    }

    pub fn get_file(&self, stego_name: &str, host_path: &Path) -> Result<(), FsError> {
        let data = self.read_file_bytes(stego_name)?;
        std::fs::write(host_path, data)?;
        Ok(())
    }

    pub fn read_file_bytes(&self, stego_name: &str) -> Result<Vec<u8>, FsError> {
        let (_, entry) = self
            .find_live_entry(stego_name)?
            .ok_or_else(|| FsError::FileNotFound(stego_name.to_owned()))?;

        let block_indices = self.collect_block_indices(&entry)?;
        let mut out = Vec::with_capacity((entry.block_count as usize) * crate::BLOCK_SIZE);
        for logical in &block_indices {
            let block = self.read_block(*logical)?;
            out.extend_from_slice(&block);
        }
        out.truncate(entry.size_bytes as usize);

        let mut hasher = Hasher::new();
        hasher.update(&out);
        let actual = hasher.finalize();
        if actual != entry.crc32 {
            return Err(FsError::Crc32Mismatch {
                name: entry.filename,
                expected: entry.crc32,
                actual,
            });
        }

        Ok(out)
    }

    pub fn list_files(&self) -> Result<Vec<FileEntry>, FsError> {
        let mut out = Vec::new();
        let length = self.file_table_length_blocks();
        let start = self.file_table_start_block();
        for offset in 0..length {
            let block = self.read_file_table_block(start + offset)?;
            for entry in block.entries {
                if entry.is_live() {
                    out.push(entry);
                }
            }
        }
        Ok(out)
    }

    pub fn delete_file(&mut self, stego_name: &str) -> Result<(), FsError> {
        let (slot, entry) = self
            .find_live_entry(stego_name)?
            .ok_or_else(|| FsError::FileNotFound(stego_name.to_owned()))?;

        // Commit: write the tombstone with zeroed block references first, so
        // a crash after commit leaves no dangling block list. The actual
        // blocks become orphans that recovery reclaims. Before commit, no
        // device state has changed.
        let mut tombstoned = entry.clone();
        tombstoned.flags |= FLAG_DELETED;
        tombstoned.block_count = 0;
        tombstoned.overflow_block = NO_BLOCK;
        tombstoned.inline_blocks.clear();
        self.write_entry_at(slot, &tombstoned)?;

        // Cleanup: free the data and overflow blocks. If we crash mid-free,
        // recovery's orphan scan will reclaim the stragglers on next open.
        let data_blocks = self.collect_block_indices(&entry)?;
        for logical in data_blocks {
            self.free_block(logical)?;
        }
        let mut overflow_cursor = entry.overflow_block;
        while overflow_cursor != NO_BLOCK {
            let bytes = self.read_block(overflow_cursor)?;
            // Peek next before freeing so we can walk the chain.
            let next_raw = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
            self.free_block(overflow_cursor)?;
            overflow_cursor = next_raw;
        }

        Ok(())
    }

    // -- Helpers below are private to this module --

    fn alloc_blocks(&mut self, n: u32) -> Result<Vec<u32>, FsError> {
        let mut blocks = Vec::with_capacity(n as usize);
        for _ in 0..n {
            match self.alloc_block() {
                Ok(b) => blocks.push(b),
                Err(err) => {
                    for b in &blocks {
                        let _ = self.free_block(*b);
                    }
                    return Err(err.into());
                }
            }
        }
        Ok(blocks)
    }

    fn write_overflow_chain(
        &mut self,
        overflow_blocks: &[u32],
        remaining_indices: &[u32],
    ) -> Result<(), FsError> {
        let mut cursor = 0;
        for (i, &ov_block) in overflow_blocks.iter().enumerate() {
            let chunk_end = (cursor + OVERFLOW_ENTRIES_PER_BLOCK).min(remaining_indices.len());
            let entries = remaining_indices[cursor..chunk_end].to_vec();
            let next = overflow_blocks.get(i + 1).copied().unwrap_or(NO_BLOCK);
            let block = OverflowBlock::new(next, entries)?;
            let bytes = block.encode()?;
            self.write_block(ov_block, &bytes)?;
            cursor = chunk_end;
        }
        Ok(())
    }

    fn collect_block_indices(&self, entry: &FileEntry) -> Result<Vec<u32>, FsError> {
        let mut out = entry.inline_blocks.clone();
        let mut remaining = (entry.block_count as usize).saturating_sub(out.len());
        let mut cursor = entry.overflow_block;
        while remaining > 0 {
            if cursor == NO_BLOCK {
                break;
            }
            let bytes = self.read_block(cursor)?;
            let live = remaining.min(OVERFLOW_ENTRIES_PER_BLOCK);
            let ov = OverflowBlock::decode(&bytes, live)?;
            out.extend_from_slice(&ov.entries);
            remaining = remaining.saturating_sub(live);
            cursor = ov.next;
        }
        Ok(out)
    }

    fn find_entry_by_name(
        &self,
        name: &str,
    ) -> Result<Option<(EntryLocation, FileEntry)>, FsError> {
        let length = self.file_table_length_blocks();
        let start = self.file_table_start_block();
        for offset in 0..length {
            let block_index = start + offset;
            let block = self.read_file_table_block(block_index)?;
            for (slot, entry) in block.entries.into_iter().enumerate() {
                if !entry.is_free() && entry.filename == name {
                    return Ok(Some((EntryLocation { block_index, slot }, entry)));
                }
            }
        }
        Ok(None)
    }

    fn find_live_entry(&self, name: &str) -> Result<Option<(EntryLocation, FileEntry)>, FsError> {
        match self.find_entry_by_name(name)? {
            Some((loc, entry)) if entry.is_live() => Ok(Some((loc, entry))),
            _ => Ok(None),
        }
    }

    fn find_free_slot(&self) -> Result<Option<EntryLocation>, FsError> {
        let length = self.file_table_length_blocks();
        let start = self.file_table_start_block();
        for offset in 0..length {
            let block_index = start + offset;
            let block = self.read_file_table_block(block_index)?;
            for (slot, entry) in block.entries.iter().enumerate() {
                if entry.is_free() {
                    return Ok(Some(EntryLocation { block_index, slot }));
                }
            }
        }
        Ok(None)
    }

    fn write_entry_at(&mut self, loc: EntryLocation, entry: &FileEntry) -> Result<(), FsError> {
        let mut block = self.read_file_table_block(loc.block_index)?;
        block.entries[loc.slot] = entry.clone();
        let bytes = block.encode()?;
        self.write_physical_block_raw(loc.block_index, &bytes)?;
        self.flush()?;
        Ok(())
    }

    fn read_file_table_block(&self, block_index: u32) -> Result<FileTableBlock, FsError> {
        let bytes = self.read_physical_block_raw(block_index)?;
        Ok(FileTableBlock::decode(&bytes)?)
    }

    fn file_table_start_block(&self) -> u32 {
        self.superblock().fields.file_table_start
    }

    fn file_table_length_blocks(&self) -> u32 {
        self.superblock().fields.file_table_length
    }

    fn file_table_capacity(&self) -> usize {
        (self.file_table_length_blocks() as usize) * ENTRIES_PER_BLOCK
    }
}

fn validate_filename(name: &str) -> Result<(), FsError> {
    if name.is_empty() {
        return Err(FsError::InvalidFilename {
            reason: "empty filename",
        });
    }
    if name.len() > MAX_FILENAME_BYTES {
        return Err(FsError::InvalidFilename {
            reason: "filename exceeds 95 bytes",
        });
    }
    if name.contains('\0') {
        return Err(FsError::InvalidFilename {
            reason: "filename contains null byte",
        });
    }
    if name.contains('/') {
        return Err(FsError::InvalidFilename {
            reason: "filename contains '/' (directories not supported in V1)",
        });
    }
    Ok(())
}

fn blocks_for_size(size: u64) -> u32 {
    if size == 0 {
        return 0;
    }
    size.div_ceil(crate::BLOCK_SIZE as u64) as u32
}

fn overflow_blocks_for_count(block_count: u32) -> u32 {
    if (block_count as usize) <= MAX_INLINE_BLOCKS {
        return 0;
    }
    let remaining = block_count as usize - MAX_INLINE_BLOCKS;
    remaining.div_ceil(OVERFLOW_ENTRIES_PER_BLOCK) as u32
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
