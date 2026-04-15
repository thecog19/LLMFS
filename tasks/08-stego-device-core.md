# Task 08: Stego Device Core

Status: todo
Depends on: `06-tensor-map-and-address-translation.md`, `07-metadata-layout-and-integrity.md`
Spec refs: `DESIGN.MD` sections "API Surface", "Physical I/O", "The 'Model Quality' Gauge"

Objective:
Build the primary block-device abstraction over mmap-backed GGUF tensor storage.

Scope:

- Open GGUF files via `memmap2`.
- Initialize or mount superblock metadata.
- Implement `read_block`, `alloc_block`, `free_block`, `total_blocks`, `used_blocks`, `verify_integrity`, and `flush`.
- Verify CRC32 on reads.
- Keep higher layers isolated from physical quant packing.

Acceptance criteria:

- The device can initialize a fresh GGUF-backed block space.
- Allocated blocks can be written and read back through the logical block interface.
- Integrity verification reports corrupted blocks accurately.

Tests first:

- End-to-end roundtrip tests through `StegoDevice`.
- Allocation tests covering empty, partially used, and full-device states.
- Read-path corruption tests that fail with an explicit integrity error.
