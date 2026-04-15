# Task 01: Repo Reset And Dependency Baseline

Status: todo
Depends on: none
Spec refs: DESIGN-NEW.MD sections "2. Version Roadmap", "10. Crate Structure"

Objective:
Update `Cargo.toml` and the crate module tree for the V1 design. Drop SQL-era
dependencies, add NBD and HTTP dependencies for the new surface, and reshape
`src/lib.rs` / `src/main.rs` module declarations to match the new crate
structure (even if most module bodies are still placeholder stubs — they get
filled in by subsequent tasks).

Scope:

- Remove `rusqlite` from `Cargo.toml` dependencies.
- Add dependencies for:
  - HTTP client for llama-server bridge: `ureq` (blocking, no async runtime needed for V1).
  - NBD: start with a hand-rolled protocol implementation; no crate dep yet.
  - Async runtime: not needed in V1 (`ureq` and blocking socket I/O are fine).
- Update `src/lib.rs` module declarations: remove `compress`, `nlq`, `vfs`; add `fs`, `nbd`, `ask`, `stego::redirection`, `stego::freelist`.
- Update `src/main.rs` clap subcommand enum to the V1 surface (`init`, `status`, `store`, `get`, `ls`, `rm`, `verify`, `mount`, `unmount`, `ask`, `dump`, `wipe`). Leave bodies as `unimplemented!()` or stub — the subsequent tasks wire them.
- Delete the dead V0 module directories and placeholder files in ONE commit at the end of this task (not spread across later tasks):
  - `src/vfs/` (740 LOC SQLite VFS)
  - `src/compress/` (12-line BPE placeholder)
  - `src/nlq/` (all four placeholder files)
  - `src/gguf/tokenizer.rs` (12-line placeholder)
  - `tests/sql_smoke.rs` (SQLite integration test)
  - `examples/real_model_smoke.rs` (uses `rusqlite`)
- The `src/stego/device.rs` pending-metadata-op recovery paths and mirrored-backup superblock stay for now — Tasks 05, 07, 08 rework them.
- `src/diagnostics.rs` placeholder stays until Task 14 fills it.

Existing code to reuse / rework / delete:
- Reuse: `Cargo.toml` baseline (clap, memmap2, thiserror, crc32fast, tempfile); `src/lib.rs` structure
- Rework: `Cargo.toml` (deps), `src/lib.rs` (module tree), `src/main.rs` (subcommand enum)
- Delete: `src/vfs/`, `src/compress/`, `src/nlq/`, `src/gguf/tokenizer.rs`, `tests/sql_smoke.rs`, `examples/real_model_smoke.rs`

Acceptance criteria:
- `cargo build --offline` succeeds with the new dependency set.
- `cargo test --offline` passes (tests from deleted modules are also deleted; all surviving tests still pass).
- `cargo run -- --help` lists the V1 subcommand set.
- No `rusqlite`, SQLite, BPE, or NLQ references remain in the source tree (`grep -R "rusqlite\|sqlite\|compress::\|nlq::" src/ tests/` returns nothing).
