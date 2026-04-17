# Task 09: File Table And File Operations

Status: done
Depends on: 08-shadow-copy-and-recovery.md
Spec refs: DESIGN-NEW.MD section "6. File Storage Layer"

Note on block layout — this task follows `DESIGN-NEW.MD §6` verbatim:
**data blocks are clean 4096 bytes with no internal chain pointer**, and
`FileEntry` carries `block_count + overflow_block_index + inline_block_map[28]`
per §6. An earlier draft of this task said `ceil(size / 4092)` and "chaining
via 4-byte next-pointer", which would shear ext4 pages once NBD (Task 12) is
plumbed through — NBD hands 4096-byte slices straight from `read_block` to
the kernel, so any in-block metadata corrupts the filesystem above.

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

Deviations from spec (intentional):

- `delete_file` writes the tombstone **before** freeing blocks, reversing the
  sequence in the task scope. Tombstoning with zeroed `block_count /
  overflow_block / inline_blocks` is the atomic commit point: a crash after
  the tombstone leaves an orphaned block set that recovery's orphan scan
  reclaims on next open. The spec order (free blocks first, then tombstone)
  would risk the symmetric failure where a partially-freed file still shows
  up in `list_files` with block references pointing at free-list entries.
  `name`, `size_bytes`, and timestamps stay on the tombstone for forensics.

- `store_bytes` is a public sibling of `store_file` that accepts an
  in-memory payload. Task 09 mandated file-from-disk only; the in-memory
  variant avoids unnecessary temp-file churn in tests and will be reused by
  the NBD and `ask` layers.

- Content comparison in tests uses byte-equality rather than SHA256. Adding
  `sha2` as a dev-dependency buys nothing beyond what `assert_eq!` on `Vec<u8>`
  already covers; the end-to-end CRC32 check inside `read_file_bytes`
  guarantees retrieval matches what was stored.

- V1's same-name-after-delete rule: `store_bytes` rejects a name that a
  tombstone currently holds with `FsError::DuplicateName`. The task called
  this out as a V1 decision; there is no "vacuum" operation that frees the
  slot, so a name is permanently reserved once used.
