use crc32fast::Hasher;
use thiserror::Error;

use crate::gguf::quant::GgufQuantType;

pub const SUPERBLOCK_MAGIC: [u8; 5] = *b"LLMDB";
pub const INTEGRITY_MAGIC: [u8; 4] = *b"ICHK";
pub const SUPERBLOCK_VERSION: u8 = 1;
pub const NO_BLOCK: u32 = u32::MAX;
pub const ENTRIES_PER_INTEGRITY_BLOCK: usize = 1020;

const SUPERBLOCK_CRC_RANGE_END: usize = 0x38;
const SUPERBLOCK_CRC_OFFSET: usize = 0x38;
const FLAG_LOBOTOMY: u8 = 0x01;
const FLAG_DIRTY: u8 = 0x02;

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

/// Superblock layout per DESIGN-NEW §5.
///
/// ```text
/// 0x00  5     magic "LLMDB"
/// 0x05  1     version (1)
/// 0x06  2     block size (4096)
/// 0x08  4     total blocks
/// 0x0C  4     free list head
/// 0x10  4     integrity chain head
/// 0x14  4     redirection table start
/// 0x18  4     redirection table length
/// 0x1C  4     file table start
/// 0x20  4     file table length
/// 0x24  1     flags (bit 0 lobotomy, bit 1 dirty)
/// 0x25  1     quant profile
/// 0x26  2     reserved
/// 0x28  8     generation counter (monotonic, bumped on every superblock persist)
/// 0x30  4     shadow_block (NO_BLOCK = no write in flight)
/// 0x34  4     shadow_target (logical block being replaced; NO_BLOCK if shadow_block is NO_BLOCK)
/// 0x38  4     CRC32 of bytes 0x00..0x38
/// 0x3C  4036  reserved / padding
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuperblockFields {
    pub total_blocks: u32,
    pub free_list_head: u32,
    pub integrity_chain_head: u32,
    pub redirection_table_start: u32,
    pub redirection_table_length: u32,
    pub file_table_start: u32,
    pub file_table_length: u32,
    pub flags: u8,
    pub quant_profile: u8,
    pub generation: u64,
    pub shadow_block: u32,
    pub shadow_target: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Superblock {
    pub fields: SuperblockFields,
}

impl Superblock {
    pub fn new(fields: SuperblockFields) -> Self {
        Self { fields }
    }

    pub fn is_lobotomy(&self) -> bool {
        self.fields.flags & FLAG_LOBOTOMY != 0
    }

    pub fn is_dirty(&self) -> bool {
        self.fields.flags & FLAG_DIRTY != 0
    }

    pub fn set_dirty(&mut self, dirty: bool) {
        if dirty {
            self.fields.flags |= FLAG_DIRTY;
        } else {
            self.fields.flags &= !FLAG_DIRTY;
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = vec![0_u8; crate::BLOCK_SIZE];
        bytes[0x00..0x05].copy_from_slice(&SUPERBLOCK_MAGIC);
        bytes[0x05] = SUPERBLOCK_VERSION;
        bytes[0x06..0x08].copy_from_slice(&(crate::BLOCK_SIZE as u16).to_le_bytes());
        bytes[0x08..0x0C].copy_from_slice(&self.fields.total_blocks.to_le_bytes());
        bytes[0x0C..0x10].copy_from_slice(&self.fields.free_list_head.to_le_bytes());
        bytes[0x10..0x14].copy_from_slice(&self.fields.integrity_chain_head.to_le_bytes());
        bytes[0x14..0x18].copy_from_slice(&self.fields.redirection_table_start.to_le_bytes());
        bytes[0x18..0x1C].copy_from_slice(&self.fields.redirection_table_length.to_le_bytes());
        bytes[0x1C..0x20].copy_from_slice(&self.fields.file_table_start.to_le_bytes());
        bytes[0x20..0x24].copy_from_slice(&self.fields.file_table_length.to_le_bytes());
        bytes[0x24] = self.fields.flags;
        bytes[0x25] = self.fields.quant_profile;
        // 0x26..0x28 reserved (zeroed above)
        bytes[0x28..0x30].copy_from_slice(&self.fields.generation.to_le_bytes());
        bytes[0x30..0x34].copy_from_slice(&self.fields.shadow_block.to_le_bytes());
        bytes[0x34..0x38].copy_from_slice(&self.fields.shadow_target.to_le_bytes());

        let crc = superblock_crc32(&bytes);
        bytes[SUPERBLOCK_CRC_OFFSET..SUPERBLOCK_CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
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

        if bytes[0x00..0x05] != SUPERBLOCK_MAGIC {
            return Err(IntegrityError::InvalidMagic {
                context: "superblock",
            });
        }

        let version = bytes[0x05];
        if version != SUPERBLOCK_VERSION {
            return Err(IntegrityError::UnsupportedVersion(version));
        }

        let block_size = u16::from_le_bytes([bytes[0x06], bytes[0x07]]) as usize;
        if block_size != crate::BLOCK_SIZE {
            return Err(IntegrityError::UnexpectedBlockSize(block_size));
        }

        let expected_crc = u32::from_le_bytes(
            bytes[SUPERBLOCK_CRC_OFFSET..SUPERBLOCK_CRC_OFFSET + 4]
                .try_into()
                .unwrap(),
        );
        let actual_crc = superblock_crc32(bytes);
        if expected_crc != actual_crc {
            return Err(IntegrityError::ChecksumMismatch {
                expected: expected_crc,
                actual: actual_crc,
            });
        }

        Ok(Self {
            fields: SuperblockFields {
                total_blocks: u32::from_le_bytes(bytes[0x08..0x0C].try_into().unwrap()),
                free_list_head: u32::from_le_bytes(bytes[0x0C..0x10].try_into().unwrap()),
                integrity_chain_head: u32::from_le_bytes(bytes[0x10..0x14].try_into().unwrap()),
                redirection_table_start: u32::from_le_bytes(bytes[0x14..0x18].try_into().unwrap()),
                redirection_table_length: u32::from_le_bytes(bytes[0x18..0x1C].try_into().unwrap()),
                file_table_start: u32::from_le_bytes(bytes[0x1C..0x20].try_into().unwrap()),
                file_table_length: u32::from_le_bytes(bytes[0x20..0x24].try_into().unwrap()),
                flags: bytes[0x24],
                quant_profile: bytes[0x25],
                generation: u64::from_le_bytes(bytes[0x28..0x30].try_into().unwrap()),
                shadow_block: u32::from_le_bytes(bytes[0x30..0x34].try_into().unwrap()),
                shadow_target: u32::from_le_bytes(bytes[0x34..0x38].try_into().unwrap()),
            },
        })
    }
}

/// Quant-profile byte: each bit indicates a quant type present in the
/// tensor map for the `status` command.
pub fn encode_quant_profile(types: &[GgufQuantType]) -> u8 {
    let mut profile = 0_u8;
    for qt in types {
        profile |= match qt {
            GgufQuantType::Q8_0 => 0x01,
            GgufQuantType::Q6K => 0x02,
            GgufQuantType::Q5K => 0x04,
            GgufQuantType::Q4K => 0x08,
            GgufQuantType::Q3K => 0x10,
            GgufQuantType::F16 => 0x20,
            GgufQuantType::F32 => 0x40,
            _ => 0x00,
        };
    }
    profile
}

pub fn decode_quant_profile(profile: u8) -> Vec<GgufQuantType> {
    let mut types = Vec::new();
    if profile & 0x01 != 0 {
        types.push(GgufQuantType::Q8_0);
    }
    if profile & 0x02 != 0 {
        types.push(GgufQuantType::Q6K);
    }
    if profile & 0x04 != 0 {
        types.push(GgufQuantType::Q5K);
    }
    if profile & 0x08 != 0 {
        types.push(GgufQuantType::Q4K);
    }
    if profile & 0x10 != 0 {
        types.push(GgufQuantType::Q3K);
    }
    if profile & 0x20 != 0 {
        types.push(GgufQuantType::F16);
    }
    if profile & 0x40 != 0 {
        types.push(GgufQuantType::F32);
    }
    types
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
    UnsupportedVersion(u8),
    #[error("unexpected block size {0}")]
    UnexpectedBlockSize(usize),
    #[error("superblock checksum mismatch: expected {expected:#x}, got {actual:#x}")]
    ChecksumMismatch { expected: u32, actual: u32 },
    #[error("too many integrity entries: {0}")]
    TooManyIntegrityEntries(usize),
    #[error("integrity entry count overflow: {0}")]
    EntryCountOverflow(u32),
}

fn superblock_crc32(bytes: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(&bytes[..SUPERBLOCK_CRC_RANGE_END]);
    hasher.finalize()
}
