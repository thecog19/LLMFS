# Task 06: Redirection Table

Status: done
Depends on: 05-superblock-and-integrity.md
Spec refs: DESIGN-NEW.MD section "5. Stego Device" (Redirection Table, Block Write Implementation)

Objective:
Introduce the logical→physical block mapping that DESIGN-NEW §5 specifies as
the mechanism for shadow-copy-then-swap. V0 did shadow-copy by reserving a
single designated "shadow block" and copying bytes in weight space on each
write; V1 instead points each logical block through a redirection table so
the swap is a pointer flip in a metadata block.

Scope:

- Create `src/stego/redirection.rs` with:
  - `struct RedirectionTableBlock { first_logical_block: u32, entry_count: u32, entries: Vec<u32> }` encoded per §5 (4-byte first logical index, 4-byte count, 1022 u32 entries = 4088 bytes payload, 4096 total).
  - `struct RedirectionTable { blocks: Vec<RedirectionTableBlock> }` in-memory view.
  - `fn logical_to_physical(&self, logical: u32) -> u32` — straight table lookup.
  - `fn set_mapping(&mut self, logical: u32, physical: u32)` — mutates the table in memory.
  - `fn encode(&self) -> Vec<Vec<u8>>` / `fn decode(blocks: &[Vec<u8>]) -> Result<Self, RedirectionError>`.
  - Identity-mapping constructor: `fn identity(total_blocks: u32) -> Self`.
- Add `RedirectionError` enum with InvalidBlockLength, InvalidBlockCount variants.
- Store the redirection table in dedicated stego blocks (1022 entries per 4096-byte block; for 300K total blocks that's ~294 blocks ≈ 1.2 MB of metadata, per §5 sizing).
- The table is populated at `init` time as identity (logical == physical) and persisted through the stego device's normal block-write path (Task 07).
- Unit tests in `tests/redirection_table.rs`:
  - Identity-mapping roundtrip: encode a table, decode, verify every `logical_to_physical(n) == n`.
  - Update-and-read: `set_mapping(5, 42)`, encode, decode, verify.
  - Cross-block: table spans multiple blocks (e.g. 3000 entries), verify mapping for logical indices in the second and third blocks.
  - Decode rejects wrong block length and wrong entry count.

Existing code to reuse / rework / delete:
- Reuse: CRC32 helper in `src/stego/integrity.rs` (if the table needs its own checksums — optional in V1, data blocks have CRC32 already)
- Rework: nothing
- Delete: nothing

Acceptance criteria:
- `RedirectionTableBlock::encode` produces exactly 4096 bytes.
- Each block holds exactly 1022 entries.
- `logical_to_physical(n) == n` holds after `RedirectionTable::identity(total)`.
- `cargo test --offline` passes redirection_table tests.
- No CRC32 verification is required on the table itself in V1 (the stego block layer already verifies each of its data blocks including the table blocks).
