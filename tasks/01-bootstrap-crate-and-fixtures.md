# Task 01: Bootstrap Crate And Fixtures

Status: todo
Depends on: none
Spec refs: `DESIGN.MD` sections "Implementation Plan", "Crate Structure", "Resolved Decisions"

Objective:
Create the Rust workspace skeleton, dependency baseline, and test fixture strategy needed by every later task.

Scope:

- Initialize `Cargo.toml`, `src/`, `tests/`, and `benches/` to match the proposed crate structure.
- Add baseline dependencies for `clap`, `memmap2`, `rusqlite`, CRC32 support, and test tooling.
- Decide how GGUF fixtures are sourced for tests.
- Add a tiny synthetic fixture format or generator so packer and parser tests do not rely on a multi-GB real model.

Acceptance criteria:

- `cargo test` runs and passes with placeholder module wiring.
- The repo contains a documented fixture strategy for GGUF v2/v3 tests.
- Module boundaries exist for `gguf`, `stego`, `compress`, `vfs`, `nlq`, and diagnostics.

Tests first:

- Add a smoke test that the crate builds and the top-level modules link.
- Add a fixture-loading test proving the test data path is stable on all supported dev machines.
