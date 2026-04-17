use thiserror::Error;

use crate::stego::integrity::NO_BLOCK;

/// Bootstrap handle for the `bootstrap_smoke` test — exposes the static
/// per-block entry count so downstream crates can size caches / validation
/// without depending on the internal constant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileTableBootstrap {
    pub entries_per_block: usize,
}

impl Default for FileTableBootstrap {
    fn default() -> Self {
        Self {
            entries_per_block: ENTRIES_PER_BLOCK,
        }
    }
}

/// On-disk size of one file-table entry in bytes, per DESIGN-NEW §6.
pub const ENTRY_BYTES: usize = 256;

/// Number of entries in one file-table block (4096 / 256).
pub const ENTRIES_PER_BLOCK: usize = 16;

/// Maximum inline block indices carried in a file-table entry. Files larger
/// than `MAX_INLINE_BLOCKS * BLOCK_SIZE` chain extra indices through overflow
/// blocks.
pub const MAX_INLINE_BLOCKS: usize = 28;

/// Maximum filename payload in bytes. The field is 96 bytes on disk with one
/// byte reserved for a null terminator.
pub const MAX_FILENAME_BYTES: usize = 95;
const FILENAME_FIELD_BYTES: usize = 96;

/// Bit 0 of the flags byte marks a tombstoned entry.
pub const FLAG_DELETED: u8 = 0x01;

const OFFSET_ENTRY_TYPE: usize = 0x00;
const OFFSET_FLAGS: usize = 0x01;
const OFFSET_MODE: usize = 0x02;
const OFFSET_UID: usize = 0x04;
const OFFSET_GID: usize = 0x08;
const OFFSET_SIZE: usize = 0x0C;
const OFFSET_CREATED: usize = 0x14;
const OFFSET_MODIFIED: usize = 0x1C;
const OFFSET_CRC32: usize = 0x24;
const OFFSET_BLOCK_COUNT: usize = 0x28;
const OFFSET_OVERFLOW: usize = 0x2C;
const OFFSET_FILENAME: usize = 0x30;
const OFFSET_INLINE_MAP: usize = 0x90;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileEntryType {
    Free = 0,
    Regular = 1,
    Directory = 2,
    Symlink = 3,
}

impl FileEntryType {
    pub fn from_u8(byte: u8) -> Result<Self, FileTableError> {
        match byte {
            0 => Ok(Self::Free),
            1 => Ok(Self::Regular),
            2 => Ok(Self::Directory),
            3 => Ok(Self::Symlink),
            other => Err(FileTableError::InvalidEntryType(other)),
        }
    }

    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    pub entry_type: FileEntryType,
    pub flags: u8,
    pub mode: u16,
    pub uid: u32,
    pub gid: u32,
    pub size_bytes: u64,
    pub created: u64,
    pub modified: u64,
    pub crc32: u32,
    pub block_count: u32,
    pub overflow_block: u32,
    pub filename: String,
    pub inline_blocks: Vec<u32>,
}

impl FileEntry {
    pub fn free() -> Self {
        Self {
            entry_type: FileEntryType::Free,
            flags: 0,
            mode: 0,
            uid: 0,
            gid: 0,
            size_bytes: 0,
            created: 0,
            modified: 0,
            crc32: 0,
            block_count: 0,
            overflow_block: 0,
            filename: String::new(),
            inline_blocks: Vec::new(),
        }
    }

    pub fn is_free(&self) -> bool {
        matches!(self.entry_type, FileEntryType::Free)
    }

    pub fn is_deleted(&self) -> bool {
        self.flags & FLAG_DELETED != 0
    }

    /// An entry is "live" iff it is a non-free, non-tombstoned regular file
    /// (or directory/symlink, once V1 adds those).
    pub fn is_live(&self) -> bool {
        !self.is_free() && !self.is_deleted()
    }

    pub fn encode(&self) -> Result<[u8; ENTRY_BYTES], FileTableError> {
        let mut bytes = [0_u8; ENTRY_BYTES];
        if self.is_free() {
            return Ok(bytes);
        }
        self.validate()?;

        bytes[OFFSET_ENTRY_TYPE] = self.entry_type.as_u8();
        bytes[OFFSET_FLAGS] = self.flags;
        bytes[OFFSET_MODE..OFFSET_MODE + 2].copy_from_slice(&self.mode.to_le_bytes());
        bytes[OFFSET_UID..OFFSET_UID + 4].copy_from_slice(&self.uid.to_le_bytes());
        bytes[OFFSET_GID..OFFSET_GID + 4].copy_from_slice(&self.gid.to_le_bytes());
        bytes[OFFSET_SIZE..OFFSET_SIZE + 8].copy_from_slice(&self.size_bytes.to_le_bytes());
        bytes[OFFSET_CREATED..OFFSET_CREATED + 8].copy_from_slice(&self.created.to_le_bytes());
        bytes[OFFSET_MODIFIED..OFFSET_MODIFIED + 8].copy_from_slice(&self.modified.to_le_bytes());
        bytes[OFFSET_CRC32..OFFSET_CRC32 + 4].copy_from_slice(&self.crc32.to_le_bytes());
        bytes[OFFSET_BLOCK_COUNT..OFFSET_BLOCK_COUNT + 4]
            .copy_from_slice(&self.block_count.to_le_bytes());
        bytes[OFFSET_OVERFLOW..OFFSET_OVERFLOW + 4]
            .copy_from_slice(&self.overflow_block.to_le_bytes());

        let name_bytes = self.filename.as_bytes();
        bytes[OFFSET_FILENAME..OFFSET_FILENAME + name_bytes.len()].copy_from_slice(name_bytes);

        for (i, block_index) in self.inline_blocks.iter().copied().enumerate() {
            let offset = OFFSET_INLINE_MAP + i * 4;
            bytes[offset..offset + 4].copy_from_slice(&block_index.to_le_bytes());
        }

        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, FileTableError> {
        if bytes.len() != ENTRY_BYTES {
            return Err(FileTableError::InvalidEntryLength {
                expected: ENTRY_BYTES,
                actual: bytes.len(),
            });
        }

        let entry_type = FileEntryType::from_u8(bytes[OFFSET_ENTRY_TYPE])?;
        if matches!(entry_type, FileEntryType::Free) {
            return Ok(Self::free());
        }

        let flags = bytes[OFFSET_FLAGS];
        let mode = u16::from_le_bytes(bytes[OFFSET_MODE..OFFSET_MODE + 2].try_into().unwrap());
        let uid = u32::from_le_bytes(bytes[OFFSET_UID..OFFSET_UID + 4].try_into().unwrap());
        let gid = u32::from_le_bytes(bytes[OFFSET_GID..OFFSET_GID + 4].try_into().unwrap());
        let size_bytes =
            u64::from_le_bytes(bytes[OFFSET_SIZE..OFFSET_SIZE + 8].try_into().unwrap());
        let created =
            u64::from_le_bytes(bytes[OFFSET_CREATED..OFFSET_CREATED + 8].try_into().unwrap());
        let modified =
            u64::from_le_bytes(bytes[OFFSET_MODIFIED..OFFSET_MODIFIED + 8].try_into().unwrap());
        let crc32 =
            u32::from_le_bytes(bytes[OFFSET_CRC32..OFFSET_CRC32 + 4].try_into().unwrap());
        let block_count = u32::from_le_bytes(
            bytes[OFFSET_BLOCK_COUNT..OFFSET_BLOCK_COUNT + 4]
                .try_into()
                .unwrap(),
        );
        let overflow_block = u32::from_le_bytes(
            bytes[OFFSET_OVERFLOW..OFFSET_OVERFLOW + 4]
                .try_into()
                .unwrap(),
        );

        let filename = decode_filename(
            &bytes[OFFSET_FILENAME..OFFSET_FILENAME + FILENAME_FIELD_BYTES],
        )?;

        let inline_len = (block_count as usize).min(MAX_INLINE_BLOCKS);
        let mut inline_blocks = Vec::with_capacity(inline_len);
        for i in 0..inline_len {
            let offset = OFFSET_INLINE_MAP + i * 4;
            inline_blocks.push(u32::from_le_bytes(
                bytes[offset..offset + 4].try_into().unwrap(),
            ));
        }

        let entry = Self {
            entry_type,
            flags,
            mode,
            uid,
            gid,
            size_bytes,
            created,
            modified,
            crc32,
            block_count,
            overflow_block,
            filename,
            inline_blocks,
        };
        entry.validate()?;
        Ok(entry)
    }

    fn validate(&self) -> Result<(), FileTableError> {
        let name_bytes = self.filename.as_bytes();
        if name_bytes.len() > MAX_FILENAME_BYTES {
            return Err(FileTableError::FilenameTooLong {
                len: name_bytes.len(),
                max: MAX_FILENAME_BYTES,
            });
        }
        if self.inline_blocks.len() > MAX_INLINE_BLOCKS {
            return Err(FileTableError::TooManyInlineBlocks {
                count: self.inline_blocks.len(),
                max: MAX_INLINE_BLOCKS,
            });
        }
        let expected_inline = (self.block_count as usize).min(MAX_INLINE_BLOCKS);
        if self.inline_blocks.len() != expected_inline {
            return Err(FileTableError::InlineBlockCountMismatch {
                inline: self.inline_blocks.len(),
                expected: expected_inline,
                block_count: self.block_count,
            });
        }
        if self.block_count as usize > MAX_INLINE_BLOCKS && self.overflow_block == NO_BLOCK {
            return Err(FileTableError::MissingOverflowBlock {
                block_count: self.block_count,
            });
        }
        Ok(())
    }
}

fn decode_filename(bytes: &[u8]) -> Result<String, FileTableError> {
    let end = bytes
        .iter()
        .position(|&b| b == 0)
        .ok_or(FileTableError::FilenameNotNullTerminated)?;
    String::from_utf8(bytes[..end].to_vec()).map_err(|_| FileTableError::FilenameNotUtf8)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileTableBlock {
    pub entries: Vec<FileEntry>,
}

impl FileTableBlock {
    pub fn empty() -> Self {
        Self {
            entries: (0..ENTRIES_PER_BLOCK).map(|_| FileEntry::free()).collect(),
        }
    }

    pub fn encode(&self) -> Result<[u8; crate::BLOCK_SIZE], FileTableError> {
        if self.entries.len() != ENTRIES_PER_BLOCK {
            return Err(FileTableError::EntryCountMismatch {
                actual: self.entries.len(),
                expected: ENTRIES_PER_BLOCK,
            });
        }
        let mut bytes = [0_u8; crate::BLOCK_SIZE];
        for (index, entry) in self.entries.iter().enumerate() {
            let encoded = entry.encode()?;
            let offset = index * ENTRY_BYTES;
            bytes[offset..offset + ENTRY_BYTES].copy_from_slice(&encoded);
        }
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, FileTableError> {
        if bytes.len() != crate::BLOCK_SIZE {
            return Err(FileTableError::InvalidBlockLength {
                expected: crate::BLOCK_SIZE,
                actual: bytes.len(),
            });
        }
        let mut entries = Vec::with_capacity(ENTRIES_PER_BLOCK);
        for index in 0..ENTRIES_PER_BLOCK {
            let offset = index * ENTRY_BYTES;
            let entry = FileEntry::decode(&bytes[offset..offset + ENTRY_BYTES])?;
            entries.push(entry);
        }
        Ok(Self { entries })
    }
}

/// Number of block-index entries carried in one overflow block (§6).
pub const OVERFLOW_ENTRIES_PER_BLOCK: usize = 1023;
const OVERFLOW_NEXT_OFFSET: usize = 0;
const OVERFLOW_ENTRIES_OFFSET: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverflowBlock {
    pub next: u32,
    pub entries: Vec<u32>,
}

impl OverflowBlock {
    pub fn new(next: u32, entries: Vec<u32>) -> Result<Self, FileTableError> {
        if entries.len() > OVERFLOW_ENTRIES_PER_BLOCK {
            return Err(FileTableError::TooManyOverflowEntries {
                count: entries.len(),
                max: OVERFLOW_ENTRIES_PER_BLOCK,
            });
        }
        Ok(Self { next, entries })
    }

    pub fn encode(&self) -> Result<[u8; crate::BLOCK_SIZE], FileTableError> {
        if self.entries.len() > OVERFLOW_ENTRIES_PER_BLOCK {
            return Err(FileTableError::TooManyOverflowEntries {
                count: self.entries.len(),
                max: OVERFLOW_ENTRIES_PER_BLOCK,
            });
        }
        let mut bytes = [0_u8; crate::BLOCK_SIZE];
        bytes[OVERFLOW_NEXT_OFFSET..OVERFLOW_NEXT_OFFSET + 4]
            .copy_from_slice(&self.next.to_le_bytes());
        for (i, &entry) in self.entries.iter().enumerate() {
            let offset = OVERFLOW_ENTRIES_OFFSET + i * 4;
            bytes[offset..offset + 4].copy_from_slice(&entry.to_le_bytes());
        }
        Ok(bytes)
    }

    /// `live_count` is how many of the up-to-1023 entries in this block are
    /// meaningful — tail is zero-padded. Callers know the total from the
    /// file entry's `block_count`.
    pub fn decode(bytes: &[u8], live_count: usize) -> Result<Self, FileTableError> {
        if bytes.len() != crate::BLOCK_SIZE {
            return Err(FileTableError::InvalidBlockLength {
                expected: crate::BLOCK_SIZE,
                actual: bytes.len(),
            });
        }
        if live_count > OVERFLOW_ENTRIES_PER_BLOCK {
            return Err(FileTableError::TooManyOverflowEntries {
                count: live_count,
                max: OVERFLOW_ENTRIES_PER_BLOCK,
            });
        }
        let next = u32::from_le_bytes(
            bytes[OVERFLOW_NEXT_OFFSET..OVERFLOW_NEXT_OFFSET + 4]
                .try_into()
                .unwrap(),
        );
        let mut entries = Vec::with_capacity(live_count);
        for i in 0..live_count {
            let offset = OVERFLOW_ENTRIES_OFFSET + i * 4;
            entries.push(u32::from_le_bytes(
                bytes[offset..offset + 4].try_into().unwrap(),
            ));
        }
        Ok(Self { next, entries })
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum FileTableError {
    #[error("invalid entry length: expected {expected}, got {actual}")]
    InvalidEntryLength { expected: usize, actual: usize },
    #[error("invalid file-table block length: expected {expected}, got {actual}")]
    InvalidBlockLength { expected: usize, actual: usize },
    #[error("invalid file entry type byte: {0}")]
    InvalidEntryType(u8),
    #[error("filename exceeds max length: got {len} bytes, max {max}")]
    FilenameTooLong { len: usize, max: usize },
    #[error("filename is not UTF-8")]
    FilenameNotUtf8,
    #[error("filename field has no null terminator within 96 bytes")]
    FilenameNotNullTerminated,
    #[error("inline block count {count} exceeds max {max}")]
    TooManyInlineBlocks { count: usize, max: usize },
    #[error(
        "inline block vector has {inline} entries, expected {expected} for block_count {block_count}"
    )]
    InlineBlockCountMismatch {
        inline: usize,
        expected: usize,
        block_count: u32,
    },
    #[error("block_count {block_count} exceeds inline capacity but overflow_block is NO_BLOCK")]
    MissingOverflowBlock { block_count: u32 },
    #[error("file-table block must have {expected} entries, got {actual}")]
    EntryCountMismatch { actual: usize, expected: usize },
    #[error("overflow block entry count {count} exceeds max {max}")]
    TooManyOverflowEntries { count: usize, max: usize },
}
