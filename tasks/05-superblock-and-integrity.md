# Task 05: Superblock And Integrity

Status: todo
Depends on: 04-tensor-eligibility-and-map.md
Spec refs: DESIGN-NEW.MD sections "5. Stego Device" (Superblock, Free List, Integrity Blocks)

Objective:
Replace the V0 superblock layout (mirrored, generation counter, WAL region,
pending-metadata op, pending-target block) with the simpler DESIGN-NEW §5
layout (single superblock, redirection-table pointer, file-table pointer,
dirty flag, lobotomy flag, quant profile byte). Keep integrity-block format
and free-list block format — DESIGN-NEW specifies them identically to V0.

Scope:

- Rewrite `Superblock` in `src/stego/integrity.rs` to match §5 exactly:
  - 0x00 magic (5 bytes), 0x05 version (1 byte, was 2 in V0), 0x06 block size (2 bytes), 0x08 total blocks (4), 0x0C free list head (4), 0x10 integrity chain head (4), 0x14 redirection table start (4), 0x18 redirection table length (4), 0x1C file table start (4), 0x20 file table length (4), 0x24 flags (1 byte: bit 0 lobotomy, bit 1 dirty), 0x25 quant profile (1), 0x26 reserved (2), 0x28 CRC32 of 0x00-0x27 (4).
- Drop `PendingMetadataOp`, `pending_target_block`, `pending_target_crc32`, `pending_metadata_block`, `pending_metadata_aux`, `generation`, `wal_region_start`, `wal_region_length`, `shadow_block`, `table_directory_block`. Shadow-copy mechanics move to the redirection table (Task 06).
- Drop mirrored backup superblock. §5 specifies one superblock (block 0). Task 08 handles recovery via dirty-flag + orphan scan instead.
- Add `FreeListBlock` (already matches §5, no change needed: 4-byte next-pointer + 4092 unused).
- `IntegrityBlock` layout (ICHK magic, first_data_block, entry_count, next chain pointer, 1020 CRC32 entries) already matches §5 exactly — keep as-is.
- Add `QuantProfile` byte encoding helper: summarizes which quant types are present in the tensor map (bit 0 Q8_0, bit 1 Q6_K, bit 2 Q5_K, bit 3 Q4_K, bit 4 Q3_K, bit 5 F16, bit 6 F32, bit 7 reserved).
- Rewrite `tests/stego_metadata.rs`:
  - `superblock_roundtrips_with_checksum_validation` — update assertions for the new field set.
  - `integrity_block_roundtrips_crc_entries` — unchanged.
  - `integrity_block_capacity_matches_design` — unchanged (still 1020).
  - Add `superblock_rejects_invalid_magic_version_and_checksum`.
  - Add `superblock_flags_roundtrip` covering lobotomy and dirty flags independently.

Existing code to reuse / rework / delete:
- Reuse: `IntegrityBlock` (encode/decode unchanged), `FreeListBlock` (unchanged), `ENTRIES_PER_INTEGRITY_BLOCK`, `NO_BLOCK`, `INTEGRITY_MAGIC`, `SUPERBLOCK_MAGIC`
- Rework: `Superblock`, `SuperblockFields`, `superblock_crc32`, `tests/stego_metadata.rs`
- Delete: `PendingMetadataOp` enum, all pending-metadata field handling

Acceptance criteria:
- `Superblock::encode` produces a 4096-byte block with CRC32 over bytes 0x00–0x27 only (not the whole block — V0 hashed the whole block with checksum zeroed; §5 specifies 0x00–0x27).
- `Superblock::decode` rejects invalid magic, version != 1, block size != 4096, and checksum mismatch with distinct error variants.
- `SUPERBLOCK_VERSION == 1` and the version field is at offset 0x05 as a single byte (V0 had 2 bytes at 0x05).
- Flags byte at 0x24 cleanly round-trips the lobotomy and dirty bits independently.
- `cargo test --offline` passes all `stego_metadata` tests.
