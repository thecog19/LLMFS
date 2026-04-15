# Task 10: SQLite VFS

Status: todo
Depends on: `08-stego-device-core.md`, `09-atomic-writes-and-recovery.md`
Spec refs: `DESIGN.MD` sections "SQLite Integration (Layer Above)", "SQL Engine: SQLite via Custom VFS"

Objective:
Make SQLite the default storage engine over the stego-backed logical block device.

Scope:

- Register a custom SQLite VFS through `rusqlite` FFI.
- Implement `xOpen`, `xDelete`, `xAccess`, `xFullPathname`, `xRead`, `xWrite`, `xTruncate`, `xSync`, `xFileSize`, `xLock`, `xUnlock`, and `xCheckReservedLock`.
- Force WAL mode and reject rollback-journal mode.
- Translate SQLite page I/O onto the logical 4096-byte block interface.

Acceptance criteria:

- SQLite can create tables, insert rows, and query data through the VFS.
- WAL mode is enabled and used in the stego-backed database.
- File locking behaves correctly for write access.

Tests first:

- SQL smoke tests for `CREATE TABLE`, `INSERT`, `SELECT`, and `CREATE INDEX`.
- WAL-mode tests proving transactions survive reopen after commit.
- Locking tests for concurrent writer attempts.
