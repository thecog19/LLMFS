use crc32fast::Hasher;
use thiserror::Error;

pub const SUPERBLOCK_MAGIC: [u8; 5] = *b"LLMDB";
pub const INTEGRITY_MAGIC: [u8; 4] = *b"ICHK";
pub const SUPERBLOCK_VERSION: u16 = 1;
pub const NO_BLOCK: u32 = u32::MAX;
pub const ENTRIES_PER_INTEGRITY_BLOCK: usize = 1020;
const SUPERBLOCK_CHECKSUM_OFFSET: usize = 0x25;
const PENDING_TARGET_BLOCK_OFFSET: usize = 0x29;
const PENDING_TARGET_CRC32_OFFSET: usize = 0x2D;
const PENDING_METADATA_OP_OFFSET: usize = 0x31;
const PENDING_METADATA_BLOCK_OFFSET: usize = 0x32;
const PENDING_METADATA_AUX_OFFSET: usize = 0x36;
const GENERATION_OFFSET: usize = 0x3A;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegrityBootstrap {
    pub entries_per_block: usize,
}

impl Default for IntegrityBootstrap {
    fn default() -> Self {
        Self {
            entries_per_block: ENTRIES_PER_INTEGRITY_BLOCK,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuperblockFields {
    pub total_blocks: u32,
    pub free_list_head: u32,
    pub table_directory_block: u32,
    pub integrity_chain_head: u32,
    pub wal_region_start: u32,
    pub wal_region_length: u32,
    pub shadow_block: u32,
    pub pending_target_block: u32,
    pub pending_target_crc32: u32,
    pub pending_metadata_op: PendingMetadataOp,
    pub pending_metadata_block: u32,
    pub pending_metadata_aux: u32,
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PendingMetadataOp {
    None = 0,
    AllocHeadAdvance = 1,
    FreeHeadPush = 2,
}

impl PendingMetadataOp {
    fn from_raw(raw: u8) -> Result<Self, IntegrityError> {
        match raw {
            0 => Ok(Self::None),
            1 => Ok(Self::AllocHeadAdvance),
            2 => Ok(Self::FreeHeadPush),
            _ => Err(IntegrityError::UnknownPendingMetadataOp(raw)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Superblock {
    pub fields: SuperblockFields,
}

impl Superblock {
    pub fn new(fields: SuperblockFields) -> Self {
        Self { fields }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = vec![0_u8; crate::BLOCK_SIZE];
        bytes[0..5].copy_from_slice(&SUPERBLOCK_MAGIC);
        bytes[0x05..0x07].copy_from_slice(&SUPERBLOCK_VERSION.to_le_bytes());
        bytes[0x07..0x09].copy_from_slice(&(crate::BLOCK_SIZE as u16).to_le_bytes());
        bytes[0x09..0x0D].copy_from_slice(&self.fields.total_blocks.to_le_bytes());
        bytes[0x0D..0x11].copy_from_slice(&self.fields.free_list_head.to_le_bytes());
        bytes[0x11..0x15].copy_from_slice(&self.fields.table_directory_block.to_le_bytes());
        bytes[0x15..0x19].copy_from_slice(&self.fields.integrity_chain_head.to_le_bytes());
        bytes[0x19..0x1D].copy_from_slice(&self.fields.wal_region_start.to_le_bytes());
        bytes[0x1D..0x21].copy_from_slice(&self.fields.wal_region_length.to_le_bytes());
        bytes[0x21..0x25].copy_from_slice(&self.fields.shadow_block.to_le_bytes());
        bytes[PENDING_TARGET_BLOCK_OFFSET..PENDING_TARGET_BLOCK_OFFSET + 4]
            .copy_from_slice(&self.fields.pending_target_block.to_le_bytes());
        bytes[PENDING_TARGET_CRC32_OFFSET..PENDING_TARGET_CRC32_OFFSET + 4]
            .copy_from_slice(&self.fields.pending_target_crc32.to_le_bytes());
        bytes[PENDING_METADATA_OP_OFFSET] = self.fields.pending_metadata_op as u8;
        bytes[PENDING_METADATA_BLOCK_OFFSET..PENDING_METADATA_BLOCK_OFFSET + 4]
            .copy_from_slice(&self.fields.pending_metadata_block.to_le_bytes());
        bytes[PENDING_METADATA_AUX_OFFSET..PENDING_METADATA_AUX_OFFSET + 4]
            .copy_from_slice(&self.fields.pending_metadata_aux.to_le_bytes());
        bytes[GENERATION_OFFSET..GENERATION_OFFSET + 8]
            .copy_from_slice(&self.fields.generation.to_le_bytes());

        let checksum = superblock_crc32(&bytes);
        bytes[SUPERBLOCK_CHECKSUM_OFFSET..SUPERBLOCK_CHECKSUM_OFFSET + 4]
            .copy_from_slice(&checksum.to_le_bytes());
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, IntegrityError> {
        if bytes.len() != crate::BLOCK_SIZE {
            return Err(IntegrityError::InvalidBlockLength {
                context: "superblock",
                expected: crate::BLOCK_SIZE,
                actual: bytes.len(),
            });
        }

        if bytes[0..5] != SUPERBLOCK_MAGIC {
            return Err(IntegrityError::InvalidMagic {
                context: "superblock",
            });
        }

        let version = u16::from_le_bytes([bytes[0x05], bytes[0x06]]);
        if version != SUPERBLOCK_VERSION {
            return Err(IntegrityError::UnsupportedVersion(version));
        }

        let block_size = u16::from_le_bytes([bytes[0x07], bytes[0x08]]) as usize;
        if block_size != crate::BLOCK_SIZE {
            return Err(IntegrityError::UnexpectedBlockSize(block_size));
        }

        let expected_checksum = u32::from_le_bytes([
            bytes[SUPERBLOCK_CHECKSUM_OFFSET],
            bytes[SUPERBLOCK_CHECKSUM_OFFSET + 1],
            bytes[SUPERBLOCK_CHECKSUM_OFFSET + 2],
            bytes[SUPERBLOCK_CHECKSUM_OFFSET + 3],
        ]);
        let actual_checksum = superblock_crc32(bytes);
        if expected_checksum != actual_checksum {
            return Err(IntegrityError::ChecksumMismatch {
                expected: expected_checksum,
                actual: actual_checksum,
            });
        }

        Ok(Self {
            fields: SuperblockFields {
                total_blocks: u32::from_le_bytes(bytes[0x09..0x0D].try_into().unwrap()),
                free_list_head: u32::from_le_bytes(bytes[0x0D..0x11].try_into().unwrap()),
                table_directory_block: u32::from_le_bytes(bytes[0x11..0x15].try_into().unwrap()),
                integrity_chain_head: u32::from_le_bytes(bytes[0x15..0x19].try_into().unwrap()),
                wal_region_start: u32::from_le_bytes(bytes[0x19..0x1D].try_into().unwrap()),
                wal_region_length: u32::from_le_bytes(bytes[0x1D..0x21].try_into().unwrap()),
                shadow_block: u32::from_le_bytes(bytes[0x21..0x25].try_into().unwrap()),
                pending_target_block: u32::from_le_bytes(
                    bytes[PENDING_TARGET_BLOCK_OFFSET..PENDING_TARGET_BLOCK_OFFSET + 4]
                        .try_into()
                        .unwrap(),
                ),
                pending_target_crc32: u32::from_le_bytes(
                    bytes[PENDING_TARGET_CRC32_OFFSET..PENDING_TARGET_CRC32_OFFSET + 4]
                        .try_into()
                        .unwrap(),
                ),
                pending_metadata_op: PendingMetadataOp::from_raw(
                    bytes[PENDING_METADATA_OP_OFFSET],
                )?,
                pending_metadata_block: u32::from_le_bytes(
                    bytes[PENDING_METADATA_BLOCK_OFFSET..PENDING_METADATA_BLOCK_OFFSET + 4]
                        .try_into()
                        .unwrap(),
                ),
                pending_metadata_aux: u32::from_le_bytes(
                    bytes[PENDING_METADATA_AUX_OFFSET..PENDING_METADATA_AUX_OFFSET + 4]
                        .try_into()
                        .unwrap(),
                ),
                generation: u64::from_le_bytes(
                    bytes[GENERATION_OFFSET..GENERATION_OFFSET + 8]
                        .try_into()
                        .unwrap(),
                ),
            },
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreeListBlock {
    pub next_free_block: u32,
}

impl FreeListBlock {
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = vec![0_u8; crate::BLOCK_SIZE];
        bytes[0..4].copy_from_slice(&self.next_free_block.to_le_bytes());
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, IntegrityError> {
        if bytes.len() != crate::BLOCK_SIZE {
            return Err(IntegrityError::InvalidBlockLength {
                context: "free_list_block",
                expected: crate::BLOCK_SIZE,
                actual: bytes.len(),
            });
        }

        Ok(Self {
            next_free_block: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegrityBlock {
    pub first_data_block: u32,
    pub entry_count: u32,
    pub next_integrity_block: u32,
    pub crc32_entries: Vec<u32>,
}

impl IntegrityBlock {
    pub fn encode(&self) -> Result<Vec<u8>, IntegrityError> {
        if self.crc32_entries.len() > ENTRIES_PER_INTEGRITY_BLOCK {
            return Err(IntegrityError::TooManyIntegrityEntries(
                self.crc32_entries.len(),
            ));
        }

        let mut bytes = vec![0_u8; crate::BLOCK_SIZE];
        bytes[0..4].copy_from_slice(&INTEGRITY_MAGIC);
        bytes[0x04..0x08].copy_from_slice(&self.first_data_block.to_le_bytes());
        bytes[0x08..0x0C].copy_from_slice(&self.entry_count.to_le_bytes());
        bytes[0x0C..0x10].copy_from_slice(&self.next_integrity_block.to_le_bytes());

        for (index, entry) in self.crc32_entries.iter().copied().enumerate() {
            let offset = 0x10 + (index * 4);
            bytes[offset..offset + 4].copy_from_slice(&entry.to_le_bytes());
        }

        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, IntegrityError> {
        if bytes.len() != crate::BLOCK_SIZE {
            return Err(IntegrityError::InvalidBlockLength {
                context: "integrity_block",
                expected: crate::BLOCK_SIZE,
                actual: bytes.len(),
            });
        }
        if bytes[0..4] != INTEGRITY_MAGIC {
            return Err(IntegrityError::InvalidMagic {
                context: "integrity_block",
            });
        }

        let entry_count = u32::from_le_bytes(bytes[0x08..0x0C].try_into().unwrap());
        let entry_len = usize::try_from(entry_count)
            .map_err(|_| IntegrityError::EntryCountOverflow(entry_count))?;
        if entry_len > ENTRIES_PER_INTEGRITY_BLOCK {
            return Err(IntegrityError::TooManyIntegrityEntries(entry_len));
        }

        let mut crc32_entries = Vec::with_capacity(entry_len);
        for index in 0..entry_len {
            let offset = 0x10 + (index * 4);
            crc32_entries.push(u32::from_le_bytes(
                bytes[offset..offset + 4].try_into().unwrap(),
            ));
        }

        Ok(Self {
            first_data_block: u32::from_le_bytes(bytes[0x04..0x08].try_into().unwrap()),
            entry_count,
            next_integrity_block: u32::from_le_bytes(bytes[0x0C..0x10].try_into().unwrap()),
            crc32_entries,
        })
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum IntegrityError {
    #[error("invalid block length for {context}: expected {expected}, got {actual}")]
    InvalidBlockLength {
        context: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("invalid magic for {context}")]
    InvalidMagic { context: &'static str },
    #[error("unsupported superblock version {0}")]
    UnsupportedVersion(u16),
    #[error("unexpected block size {0}")]
    UnexpectedBlockSize(usize),
    #[error("superblock checksum mismatch: expected {expected:#x}, got {actual:#x}")]
    ChecksumMismatch { expected: u32, actual: u32 },
    #[error("too many integrity entries: {0}")]
    TooManyIntegrityEntries(usize),
    #[error("integrity entry count overflow: {0}")]
    EntryCountOverflow(u32),
    #[error("unknown pending metadata op {0}")]
    UnknownPendingMetadataOp(u8),
}

fn superblock_crc32(bytes: &[u8]) -> u32 {
    let mut working = bytes.to_vec();
    working[SUPERBLOCK_CHECKSUM_OFFSET..SUPERBLOCK_CHECKSUM_OFFSET + 4].fill(0);
    let mut hasher = Hasher::new();
    hasher.update(&working);
    hasher.finalize()
}
