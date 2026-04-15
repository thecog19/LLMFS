# Task 16: Raw-Mode Storage Engine

Status: todo
Depends on: `07-metadata-layout-and-integrity.md`, `08-stego-device-core.md`, `11-cli-core-commands.md`
Spec refs: `DESIGN.MD` sections "Table Directory (Block 1)", "Data Blocks", "SQLite Integration (Layer Above)"

Objective:
Implement the non-default custom storage engine described in the design as `--raw` mode.

Scope:

- Implement the table directory layout and fixed-width row blocks.
- Add raw-mode allocation and row-packing logic.
- Expose raw-mode creation and query-path behavior behind an explicit CLI flag.
- Keep this isolated from the default SQLite VFS path.

Acceptance criteria:

- A user can create a raw-mode table, insert fixed-width rows, and read them back.
- Raw-mode metadata coexists safely with the shared superblock and integrity machinery.
- The default user path remains SQLite VFS, not raw mode.

Tests first:

- Table-directory serialization tests.
- Fixed-width row packing tests across block boundaries.
- Raw-mode smoke tests for create, insert, and read operations.
