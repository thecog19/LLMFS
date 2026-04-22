//! V2 directory — a list of named entries pointing to child inodes.
//!
//! # Model
//!
//! A directory is just a regular file: it has an [`Inode`] whose data
//! chunks, when concatenated, form the serialized entry list. The
//! super-root's `root_dir_inode` points at the top-level directory's
//! inode; nested directories are referenced by entries within their
//! parent (same mechanism — the entry's [`Pointer`] points at the
//! child's inode).
//!
//! That structural uniformity is why we can reuse the same CoW +
//! CDC + dedup + dirty-bitmap machinery for directories as for files.
//!
//! # On-disk layout
//!
//! ```text
//! Directory:
//!   count:    u32 LE                 (4 bytes, entry count)
//!   entries:  [Entry; count]         (stored sorted ascending by name)
//!
//! Entry:
//!   kind:     u8                     (0 = file, 1 = directory)
//!   name_len: u8                     (1..=255, bytes)
//!   reserved: [u8; 2]                (must be zero on decode)
//!   inode:    [u8; 16]               (Pointer to child inode)
//!   name:     [u8; name_len]         (UTF-8; no '/' or NUL)
//! ```
//!
//! Names are stored UTF-8 and restricted: no empty strings, no embedded
//! `/` or NUL, length 1..=255. These match POSIX filename constraints
//! closely enough that FUSE paths can pass through unmodified.
//!
//! Entries are kept sorted by name. Lookups are `O(log n)` via binary
//! search; in-place updates and removals preserve the invariant.
//! Serialized blobs are validated on decode: out-of-order is tolerated
//! (we sort on read) but duplicates are rejected.
//!
//! [`Inode`]: crate::v2::inode::Inode

use thiserror::Error;

use crate::v2::pointer::{Pointer, PointerError};

/// Maximum entry-name length in bytes — matches the `u8` field in the
/// entry header.
pub const MAX_NAME_LEN: usize = 255;

/// Bytes for the directory header (entry count).
pub const DIRECTORY_HEADER_BYTES: usize = 4;

/// Fixed-size entry header (kind + name_len + reserved + inode ptr).
/// The variable-length name follows.
pub const ENTRY_HEADER_BYTES: usize = 1 + 1 + 2 + Pointer::SIZE;

/// Distinguishes files (leaf inodes; data bytes) from directories
/// (inodes whose data bytes are a serialized [`Directory`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EntryKind {
    File,
    Directory,
}

impl EntryKind {
    fn from_byte(b: u8) -> Result<Self, DirectoryError> {
        match b {
            0 => Ok(EntryKind::File),
            1 => Ok(EntryKind::Directory),
            other => Err(DirectoryError::InvalidKind(other)),
        }
    }

    fn to_byte(self) -> u8 {
        match self {
            EntryKind::File => 0,
            EntryKind::Directory => 1,
        }
    }

    /// True for directories — handy for fs-layer traversal.
    pub fn is_directory(self) -> bool {
        matches!(self, EntryKind::Directory)
    }
}

/// One entry in a directory: name → (kind, child inode pointer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirEntry {
    pub kind: EntryKind,
    pub name: String,
    pub inode: Pointer,
}

/// A directory's in-memory form: the entry list, kept sorted by name.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Directory {
    entries: Vec<DirEntry>,
}

#[derive(Debug, Error)]
pub enum DirectoryError {
    #[error("truncated directory header: need {DIRECTORY_HEADER_BYTES} bytes, got {0}")]
    TruncatedHeader(usize),

    #[error(
        "truncated entry header at offset {offset}: need {ENTRY_HEADER_BYTES} bytes, {available} available"
    )]
    TruncatedEntryHeader { offset: usize, available: usize },

    #[error("truncated entry name at offset {offset}: need {needed} bytes, {available} available")]
    TruncatedName {
        offset: usize,
        needed: usize,
        available: usize,
    },

    #[error("invalid entry kind byte: {0}")]
    InvalidKind(u8),

    #[error("entry reserved bytes at offset {offset} must be zero, got {bytes:?}")]
    ReservedNotZero { offset: usize, bytes: [u8; 2] },

    #[error("invalid entry name: must be 1..={MAX_NAME_LEN} UTF-8 bytes and contain no '/' or NUL")]
    InvalidName,

    #[error("directory already contains an entry named '{0}'")]
    NameCollision(String),

    #[error("pointer codec: {0}")]
    Pointer(#[from] PointerError),

    #[error("trailing bytes after all entries: {count} bytes")]
    TrailingBytes { count: usize },
}

impl Directory {
    /// Construct an empty directory.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` if the directory has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Entries as a slice, in sorted order.
    pub fn entries(&self) -> &[DirEntry] {
        &self.entries
    }

    /// Binary-search lookup by name. `None` if absent.
    pub fn find(&self, name: &str) -> Option<&DirEntry> {
        self.index_of(name).map(|i| &self.entries[i])
    }

    /// Insert a new entry. Errors on collision, invalid name, or any
    /// constraint violation.
    pub fn insert(&mut self, entry: DirEntry) -> Result<(), DirectoryError> {
        validate_name(&entry.name)?;
        match self.entries.binary_search_by(|e| e.name.cmp(&entry.name)) {
            Ok(_) => Err(DirectoryError::NameCollision(entry.name)),
            Err(i) => {
                self.entries.insert(i, entry);
                Ok(())
            }
        }
    }

    /// In-place replace the `(kind, inode)` of an existing entry.
    /// Returns the previous entry, or `None` if `name` is absent.
    pub fn replace(&mut self, name: &str, kind: EntryKind, inode: Pointer) -> Option<DirEntry> {
        let i = self.index_of(name)?;
        let old = self.entries[i].clone();
        self.entries[i].kind = kind;
        self.entries[i].inode = inode;
        Some(old)
    }

    /// Remove and return the entry named `name`, or `None` if absent.
    pub fn remove(&mut self, name: &str) -> Option<DirEntry> {
        let i = self.index_of(name)?;
        Some(self.entries.remove(i))
    }

    /// Serialize to the on-disk layout.
    pub fn serialize(&self) -> Vec<u8> {
        let entry_bytes: usize = self
            .entries
            .iter()
            .map(|e| ENTRY_HEADER_BYTES + e.name.len())
            .sum();
        let mut bytes = Vec::with_capacity(DIRECTORY_HEADER_BYTES + entry_bytes);
        bytes.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for e in &self.entries {
            bytes.push(e.kind.to_byte());
            bytes.push(e.name.len() as u8);
            bytes.extend_from_slice(&[0, 0]); // reserved
            bytes.extend_from_slice(&e.inode.encode());
            bytes.extend_from_slice(e.name.as_bytes());
        }
        bytes
    }

    /// Deserialize from the on-disk layout. Validates every entry,
    /// rejects duplicates, tolerates out-of-order input (sorts on
    /// read).
    pub fn deserialize(bytes: &[u8]) -> Result<Self, DirectoryError> {
        if bytes.len() < DIRECTORY_HEADER_BYTES {
            return Err(DirectoryError::TruncatedHeader(bytes.len()));
        }
        let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let mut cursor = DIRECTORY_HEADER_BYTES;
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            if cursor + ENTRY_HEADER_BYTES > bytes.len() {
                return Err(DirectoryError::TruncatedEntryHeader {
                    offset: cursor,
                    available: bytes.len().saturating_sub(cursor),
                });
            }
            let kind = EntryKind::from_byte(bytes[cursor])?;
            let name_len = bytes[cursor + 1] as usize;
            let reserved = [bytes[cursor + 2], bytes[cursor + 3]];
            if reserved != [0, 0] {
                return Err(DirectoryError::ReservedNotZero {
                    offset: cursor + 2,
                    bytes: reserved,
                });
            }
            let ptr = Pointer::decode(&bytes[cursor + 4..cursor + 4 + Pointer::SIZE])?;
            cursor += ENTRY_HEADER_BYTES;
            if cursor + name_len > bytes.len() {
                return Err(DirectoryError::TruncatedName {
                    offset: cursor,
                    needed: name_len,
                    available: bytes.len().saturating_sub(cursor),
                });
            }
            let name_bytes = &bytes[cursor..cursor + name_len];
            let name = std::str::from_utf8(name_bytes)
                .map_err(|_| DirectoryError::InvalidName)?
                .to_owned();
            validate_name(&name)?;
            cursor += name_len;
            entries.push(DirEntry {
                kind,
                name,
                inode: ptr,
            });
        }
        if cursor != bytes.len() {
            return Err(DirectoryError::TrailingBytes {
                count: bytes.len() - cursor,
            });
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        for pair in entries.windows(2) {
            if pair[0].name == pair[1].name {
                return Err(DirectoryError::NameCollision(pair[0].name.clone()));
            }
        }
        Ok(Self { entries })
    }

    fn index_of(&self, name: &str) -> Option<usize> {
        self.entries
            .binary_search_by(|e| e.name.as_str().cmp(name))
            .ok()
    }
}

fn validate_name(name: &str) -> Result<(), DirectoryError> {
    if name.is_empty() || name.len() > MAX_NAME_LEN {
        return Err(DirectoryError::InvalidName);
    }
    if name.as_bytes().iter().any(|b| *b == b'/' || *b == 0) {
        return Err(DirectoryError::InvalidName);
    }
    Ok(())
}
