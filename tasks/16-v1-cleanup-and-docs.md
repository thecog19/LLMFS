# Task 16: V1 Cleanup And Docs

Status: todo
Depends on: 10-cli-file-commands.md, 12-nbd-server.md, 13-ask-bridge.md, 14-diagnostics-and-status.md, 15-benchmarks.md
Spec refs: DESIGN-NEW.MD section "9. CLI", "Distribution" (from DESIGN-OLD §Distribution, preserved)

Objective:
Final V1 polish: remove any remaining dead V0 code not already deleted, write
the project README, archive `DESIGN-OLD.MD`, tidy clippy lints, and produce a
V1 release tag commit. This task is the "V1 is done" marker.

Scope:

- Verify all dead V0 modules are gone (Task 01 did most of this; this is a sweep):
  - `src/vfs/` — gone
  - `src/compress/` — gone
  - `src/nlq/` — gone
  - `src/gguf/tokenizer.rs` — gone
  - `tests/sql_smoke.rs` — gone
  - `examples/real_model_smoke.rs` — gone
  - `src/stego/device.rs` pending-metadata paths — gone (Task 07/08)
  - `src/stego/integrity.rs` mirrored-backup / generation counter — gone (Task 05)
- Move `DESIGN-OLD.MD` to `docs/history/DESIGN-V0.md`. Leave a one-line pointer in the repo root README so the history is discoverable but does not clutter the top level.
- Write `README.md` at the repo root:
  - Project description (one paragraph — cribbing from DESIGN-NEW §1).
  - Build: `cargo build --release`.
  - Quickstart: `init` → `store` → `ls` → `get` on a tiny GGUF from `/models/` (user provides).
  - Mount: `mount` example (flag root requirement).
  - Ask: `ask` example (flag llama-server requirement).
  - Link to `DESIGN-NEW.MD` for the architecture.
  - Link to `docs/history/DESIGN-V0.md` for "how this project started".
- Clippy pass: `cargo clippy --offline --all-targets -- -D warnings`. Fix the lints that survive from V0 (`manual_is_multiple_of` ×4 in `src/stego/packing/float.rs`, `collapsible_if` ×1 in `src/vfs/sqlite_vfs.rs` — the latter dies with Task 01 anyway).
- Format pass: `cargo fmt --check` must pass.
- Update `tasks/README.md` statuses to `done` for everything landed. Keep this folder in the repo — it is useful project history, and the V2/V3 backlog goes here when those are specced.
- Tag the release commit: `git tag v1.0.0 -m "V1: naive stego + files + NBD + ask"`.
- Add a concise CHANGELOG.md entry for V1 listing the feature set.

Existing code to reuse / rework / delete:
- Reuse: everything shipped in Tasks 01–15
- Rework: README.md (new), CHANGELOG.md (new), `tasks/README.md` (status updates)
- Delete: any remaining V0 placeholders (sweep, not structural changes)

Acceptance criteria:
- `cargo build --release --offline` produces a binary with no warnings.
- `cargo clippy --offline --all-targets -- -D warnings` is clean.
- `cargo fmt --check` is clean.
- `cargo test --offline` passes.
- `README.md` exists and covers init / store / get / ls / rm / verify / mount / ask quickstarts.
- `git tag --list` shows `v1.0.0`.
- No file in `src/` or `tests/` mentions `rusqlite`, `sqlite`, `nlq`, `compress::`, `BPE`, or `WAL mode` (the last one is kernel-level for ext4, not for us).
