# Changelog

## v1.0.0 — 2026-04-17

First tagged release. V1 scope per `DESIGN-NEW.MD §2`.

### Storage stack
- GGUF v2 / v3 parser with full K-quant support.
- Packers for Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, F16, F32 with a
  generic `blockwise_write_range` / `blockwise_read_range` path
  so a 4 KiB NBD write only touches the affected quant blocks.
- Superblock + redirection table + integrity CRC table + file
  table + free list, all stored inside the GGUF itself.
- Split logical / physical namespaces: shadow-copy-then-swap
  under direct-addressed writes no longer aliases logical and
  physical indices.
- Dirty-flag on open + orphan-scan recovery on reopen.
- Four-tier allocation planner (Tier 1 FFN, Tier 2 Attn,
  Lobotomy embeddings, Untouchable norm/output).

### CLI
- `init`, `store`, `get`, `ls`, `rm`, `verify`, `status`,
  `dump`, `wipe`.
- `mount` / `unmount`: full NBD + `nbd-client` + optional
  `mkfs.ext4` + kernel mount, with `Ctrl-C` shutdown and a
  sidecar state file for cross-shell unmount.
- `serve`: just the NBD server, for manual driving.
- `ask`: spawns `llama-server` against the GGUF, runs a tool
  REPL with `list_files` / `read_file` / `file_info`.
- `dump-block`: per-block hex trace for debugging.

### NBD
- Newstyle handshake (NBDMAGIC + IHAVEOPT + flags,
  `NBD_OPT_EXPORT_NAME` / `NBD_OPT_GO` / `NBD_OPT_ABORT`).
- Read / write / flush / disc commands.
- EIO/EINVAL replies on device or range errors instead of
  tearing down the connection.
- `LLMDB_NBD_TRACE=1` env var for per-request tracing.

### Diagnostics
- Per-tier utilization with perplexity-impact heuristic
  (0.5 / 1.0 / 5.0 weighting for Tier1 / Tier2 / Lobotomy).

### Cover-file viability
- F16 / F32 covers stay functional models post-stego (mantissa-
  only steal); confirmed on SmolLM2-135M-f16 at 97% utilization.
- Int-quantized covers (Q8_0, Q6_K, Q5_K, Q4_K, Q3_K) are
  storage-only: `init` alone collapses inference. Documented
  as a V1 limitation in `DESIGN-NEW.MD §2`; V2 is responsible
  for sensitivity-aware allocation.

### Tests
- 131 tests across GGUF, packers, redirection, shadow-copy
  crash points, file table, file ops, CLI smoke, NBD protocol,
  NBD socket roundtrip, diagnostics, and `ask` tool dispatch.
- Clippy clean with `-D warnings`; `cargo fmt` clean.

### Deferred to post-V1
- Criterion benches (Task 15). **Landed post-v1.0.0.**
- `ask` end-to-end test against a real `llama-server`. **Landed
  post-v1.0.0** as `tests/ask_e2e.rs`, gated by `LLMDB_E2E_ASK=1`.
- `dump` subcommand body (tar archive of stored files). **Landed
  post-v1.0.0.**
