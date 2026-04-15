# Task 11: CLI Core Commands

Status: todo
Depends on: `08-stego-device-core.md`, `10-sqlite-vfs.md`
Spec refs: `DESIGN.MD` section "CLI", plus "Resolved Decisions" item 5

Objective:
Ship the minimum product surface for interacting with the database from the terminal.

Scope:

- Implement `llmdb init`, `status`, `query`, `dump`, `load`, and `wipe`.
- Wire `init` to standard mode and lobotomy mode creation.
- Wire `query`, `dump`, and `load` through SQLite over the VFS.
- Make `wipe` zero stego bits and reset metadata safely.

Acceptance criteria:

- A user can initialize a GGUF file, run SQL, export SQL, import SQL, and wipe the device from the CLI.
- `status` reports block counts and mount state without mutating the model.
- `dump` and `load` provide the ejection-seat backup path promised in the design.

Tests first:

- CLI integration tests for each command.
- Golden-output tests for basic `status` and `init` reporting.
- Dump/load roundtrip tests across a small sample database.
