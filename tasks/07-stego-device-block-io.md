# Task 07: Stego Device Block I/O

Status: todo
Depends on: 06-redirection-table.md
Spec refs: DESIGN-NEW.MD section "5. Stego Device" (Block Device Interface, Block Read Implementation, Block Write Implementation)

Objective:
Rework `StegoDevice` to use the redirection table for logical→physical block
translation, and move the device to the simpler V1 metadata layout. Read and
write paths become: translate logical index → physical index → translate
physical byte range through the tensor map → pack/unpack. V0 did this with
an in-place weight-shadow scheme; V1's shadow is a freshly allocated
physical block whose CRC goes into the integrity table before the
redirection pointer flips.

Scope:

- Rewrite `StegoDevice` in `src/stego/device.rs`:
  - `fn new(path: &Path, lobotomy: bool) -> Result<Self, DeviceError>` opens an already-initialized device.
  - `fn format(path: &Path, lobotomy: bool) -> Result<Self, DeviceError>` initializes a fresh device: superblock at block 0, integrity blocks next, redirection table next, file table next (Task 09 sets the file-table initial size to 1 block), then all remaining blocks chained into the free list.
  - `fn read_block(&self, logical: u32) -> Result<[u8; BLOCK_SIZE], DeviceError>`:
    1. `physical = redirection.logical_to_physical(logical)`
    2. read physical block from stego byte space via tensor map (existing `read_stego_bytes` logic)
    3. verify CRC32 against integrity block
  - `fn write_block(&mut self, logical: u32, data: &[u8; BLOCK_SIZE]) -> Result<(), DeviceError>`:
    1. pop a shadow physical block from the free list (Task 06's redirection table uses this as the new home for logical N)
    2. write data bytes into that physical block's stego-byte range
    3. compute CRC32, update integrity block for the shadow physical index
    4. flush (msync) — shadow fully durable
    5. record the old physical for logical N, then `redirection.set_mapping(logical, shadow)` and persist the redirection table block that holds entry N
    6. flush — mapping update durable
    7. push the old physical block back onto the free list
  - `fn alloc_block(&mut self) -> Result<u32, DeviceError>` returns a free logical block index (or new slot in the logical space, backed by an allocated physical block).
  - `fn free_block(&mut self, logical: u32) -> Result<(), DeviceError>` pushes the physical block back onto the free list and marks the logical slot as free.
  - `fn verify_block(&self, logical: u32)` / `fn verify_all(&self)` for integrity scans.
  - `fn utilization_pct(&self) -> f32`.
- Drop `PendingMetadataOp` recovery paths (Task 08 replaces them with the simpler dirty-flag + orphan-scan recovery).
- Drop the single-shadow-block scheme — the new shadow is "any free physical block".
- Drop the mirrored-backup-superblock handling — §5 has one superblock.
- Extract free-list manipulation into `src/stego/freelist.rs` (`fn push(&mut self, block: u32)`, `fn pop(&mut self) -> Option<u32>`, `fn count(&self) -> u32`).
- Update `tests/stego_device.rs`:
  - `fresh_device_init_chains_free_blocks_and_roundtrips_one_block` — update to new layout, assert free-list head points into the post-metadata region.
  - `read_block_detects_crc_mismatch_after_out_of_band_corruption` — keep (CRC mechanism unchanged).
  - `mixed_quant_device_roundtrips_across_reopen_and_integrity_scan` — keep, update metadata assertions.
  - Delete `reopen_rolls_back_pending_alloc_before_head_advance`, `reopen_finalizes_pending_free_after_block_rewrite`, `reopen_recovers_pending_shadow_write_after_interrupted_commit`, `reopen_uses_backup_superblock_when_primary_is_corrupt`. These test behaviors that Task 08 replaces.
- Add new tests:
  - `write_block_updates_redirection_and_preserves_old_until_flush`.
  - `alloc_then_write_then_free_returns_block_to_freelist`.

Existing code to reuse / rework / delete:
- Reuse: `read_stego_bytes` / `write_stego_bytes` and `decode_slot` / `encode_slot` in `src/stego/device.rs` (the bit-packing dispatch — though Task 03's `packer_for` helper should replace the inline match); `build_tensor_byte_slots`; `TensorByteSlot`; CRC32 helper
- Rework: all of `StegoDevice`'s public API, `format`, `open_internal`, `recover_if_needed`, `write_block`, `alloc_block`, `free_block`
- Delete: `PendingMetadataOp`-driven recovery; `stage_pending_write`; `stage_pending_metadata`; `clear_pending_*` helpers; mirrored-backup logic; `stage_pending_*_for_test` helpers (tests that need to simulate crashes will do so via Task 08's explicit crash-point API)

Acceptance criteria:
- `StegoDevice::format` writes superblock, integrity blocks, redirection table (identity), file table placeholder, and chains remaining blocks into the free list.
- `StegoDevice::read_block(n)` returns bytes written by `write_block(n, data)` after the write flushes.
- Corrupting the underlying mmap at a physical-block region triggers `DeviceError::IntegrityMismatch` on `read_block`.
- `cargo test --offline` passes the updated `stego_device` tests.
- `grep "PendingMetadataOp" src/ tests/` returns nothing.
