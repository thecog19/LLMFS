# Task 07: Metadata Layout And Integrity

Status: todo
Depends on: `03-tensor-selection-and-capacity-planner.md`
Spec refs: `DESIGN.MD` sections "On-Disk Metadata", "Integrity", "Checksums"

Objective:
Define and implement the logical metadata layout that lives inside the stego-backed block space.

Scope:

- Implement the superblock layout, versioning, checksum, and reserved fields.
- Implement free-list blocks and initialization rules.
- Implement dedicated integrity blocks with CRC32 coverage tables.
- Define the logical metadata model independent of the SQLite VFS and raw-mode choice.

Acceptance criteria:

- A fresh init creates a valid superblock, free list, and integrity chain.
- Superblock checksum verification detects corruption.
- Integrity blocks cover the expected range of logical data blocks.

Tests first:

- Serialization and deserialization tests for the superblock.
- Tests that a new device chains all free blocks except reserved blocks.
- CRC mismatch tests that surface corruption instead of silently accepting it.
