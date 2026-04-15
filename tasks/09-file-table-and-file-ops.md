# Task 09: File Table And File Operations

Status: todo
Depends on: 08-shadow-copy-and-recovery.md
Spec refs: DESIGN-NEW.MD section "6. File Storage Layer"

Objective:
Implement the flat file table and the store / get / ls / rm operations on top
of the block device. This is the first task that makes LLMDB directly useful
from the CLI without NBD or inference.

Scope:

- Create `src/fs/file_table.rs`:
  - `enum FileEntryType { Free = 0, Regular = 1, Directory = 2, Symlink = 3 }`
  - `struct FileEntry { entry_type: FileEntryType, flags: u8, mode: u16, uid: u32, gid: u32, size_bytes: u64, created: u64, modified: u64, first_data_block: u32, block_count: u32, crc32: u32, filename: String }` (256 bytes encoded per §6).
  - `fn encode(&self) -> [u8; 256]` / `fn decode(&[u8; 256]) -> Result<Self, FileTableError>`.
  - `struct FileTableBlock { entries: [FileEntry; 16] }` — 16 × 256 = 4096 bytes.
  - `fn encode_block(&self) -> [u8; 4096]` / `fn decode_block(&[u8; 4096])`.
  - Validation: filename ≤ 207 UTF-8 bytes, null-terminated; flags bit 0 = deleted tombstone.
- Create `src/fs/file_ops.rs`:
  - `impl StegoDevice { fn store_file(&mut self, host_path: &Path, stego_name: &str, mode: u16) -> Result<FileEntry, FsError> }`:
    1. Read host file into memory (V1 assumes files fit in RAM; §6 doesn't specify a streaming path).
    2. Compute blocks needed: `ceil(size / 4092)`.
    3. Allocate that many data blocks from the free list.
    4. Write data blocks, chaining via the 4-byte next-pointer (per §6).
    5. Find a free slot in the file table (scan through file-table blocks).
    6. Write the file entry. Flush.
  - `fn get_file(&self, stego_name: &str, host_path: &Path) -> Result<(), FsError>`:
    1. Scan file table for a regular (non-deleted) entry matching the name.
    2. Walk the block chain, concatenating 4092-byte payloads, stopping at the entry's exact `size_bytes`.
    3. Verify end-to-end CRC32 against the entry's `crc32` field.
    4. Write to host path.
  - `fn list_files(&self) -> Result<Vec<FileEntry>, FsError>`: return non-free, non-deleted entries.
  - `fn delete_file(&mut self, stego_name: &str) -> Result<(), FsError>`:
    1. Find entry; return error if not found.
    2. Walk the block chain, `free_block` on each.
    3. Set flags bit 0 (tombstone) in the entry; keep name/size for forensics. Flush.
- `FsError` enum: FileNotFound, FileTooLarge (for V1 max = available free blocks × 4092), DuplicateName, InvalidFilename, TableFull, Crc32Mismatch, plus transparent `#[from] DeviceError`.
- In superblock `format`, allocate one block for the initial file table and record its block index in `file_table_start`, with `file_table_length = 1` (16 entries = 16 files cap in V1; task backlog notes that chaining additional table blocks is V2 or later).
- Tests in `tests/file_ops.rs` (net-new):
  - Store a 1-KB file, list, get, compare SHA256 — equal.
  - Store a 10-KB file (spans 3 data blocks), get, SHA256 match.
  - Store → delete → store-same-name returns error at first store OR overwrite — decide: V1 disallows overwrite (returns `FsError::DuplicateName` on the second store even if the first is tombstoned; explicit `rm` must happen first). Document in the task and add test.
  - List returns non-deleted entries only.
  - `get_file` with wrong name returns `FsError::FileNotFound`.
  - Delete frees all data blocks: `device.used_blocks()` after delete matches the pre-store count.

Existing code to reuse / rework / delete:
- Reuse: `StegoDevice::alloc_block`, `read_block`, `write_block`, `free_block` from Task 07; CRC32 helper
- Rework: add file-table pointers to `StegoDevice::format`
- Delete: nothing

Acceptance criteria:
- `cargo test --offline` passes `tests::file_ops`.
- A 10-KB file roundtrips with SHA256 equality.
- `delete_file` followed by `list_files` excludes the deleted entry; `device.used_blocks()` returns to the pre-store count.
- Attempting to store a file whose block count exceeds free blocks returns `FsError::FileTooLarge` without partial allocation.
