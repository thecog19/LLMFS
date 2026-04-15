# Task 10: CLI File Commands

Status: todo
Depends on: 09-file-table-and-file-ops.md
Spec refs: DESIGN-NEW.MD section "9. CLI"

Objective:
Wire the CLI subcommands for direct file operations so a user can exercise
the stego device from the terminal without NBD or `ask`. Completes the
critical-path demo surface: `init → store → ls → get → rm → verify → dump → wipe`.

Scope:

- In `src/main.rs`, replace the stub `println!`/`exit(2)` dispatch with real handlers calling into `StegoDevice` and `src/fs/file_ops.rs`.
- Subcommand bodies:
  - `init <model.gguf> [--lobotomy]` — calls `StegoDevice::format(path, lobotomy)`. Reports: total blocks, metadata blocks, data blocks, quant-type breakdown, lobotomy flag.
  - `status <model.gguf>` — opens device (read-only recovery only), prints: total blocks, used blocks, free blocks, utilization %, per-tier capacity contribution, quant profile, dirty flag, file count.
  - `store <model.gguf> <host_path> [--name <stego_name>] [--mode <octal>]` — default name is `host_path.file_name()`, default mode is 0o644.
  - `get <model.gguf> <stego_name> [--output <host_path>]` — default output is `./<stego_name>`.
  - `ls <model.gguf> [-l]` — short format: names only; long format (-l): mode, size, modified time, name.
  - `rm <model.gguf> <stego_name>` — prompts for confirmation unless `--yes` is passed.
  - `verify <model.gguf>` — calls `StegoDevice::verify_all()`; reports corrupted block indices and maps them back to file entries where possible.
  - `dump <model.gguf>` — streams a tar archive of all stored files to stdout. Dependency: `tar` crate (add in Task 01 revision or here — decide at implementation time).
  - `wipe <model.gguf>` — prompts for confirmation; zeroes all stego bits by writing a zero block to every data and metadata block. Destructive. `--yes` to skip prompt.
- Exit codes: 0 success, 1 user error (file not found, duplicate, etc), 2 internal error.
- `-v` / `--verbose` flag already wired; extend `DeviceOptions { verbose }` plumbing so verbose mode prints per-block I/O at debug level.
- Tests in `tests/cli_smoke.rs` (net-new, uses `assert_cmd`):
  - `init → store → ls → get → SHA256 match → rm → ls empty` across a synthetic GGUF fixture.
  - `init → store → wipe --yes → status` shows zero files.
  - `verify` on an intentionally corrupted fixture reports the corrupted block.

Existing code to reuse / rework / delete:
- Reuse: `src/main.rs` clap `Cli` and `Command` enums (subcommand variants already exist from V0 — just change bodies); `tests/common/mod.rs` for synthetic fixtures
- Rework: all command bodies in `src/main.rs`
- Delete: the `"command '{}' is not implemented yet"` stub branch

Acceptance criteria:
- `cargo run -- init <fixture.gguf>` initializes a device and reports capacity.
- `cargo run -- store <gguf> README.md` then `cargo run -- get <gguf> README.md --output /tmp/roundtrip.md` produces a file with the same SHA256 as the original.
- `cargo test --offline tests::cli_smoke` passes end-to-end.
- Every subcommand has a `--help` that names its arguments.
- `assert_cmd` or equivalent is used (no raw subprocess spawning) for testability.
