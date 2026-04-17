# Task 14: Diagnostics And Status

Status: done
Depends on: 09-file-table-and-file-ops.md
Spec refs: DESIGN-NEW.MD section "9. CLI" (`status`), "5. Stego Device" (utilization), "13. Failure Modes" (dirty flag reporting)

Objective:
Replace the V0 placeholder `src/diagnostics.rs` with real reporting for the
`status` CLI command: utilization %, file count, total stored bytes,
per-tier capacity contribution, quant-type distribution, lobotomy flag,
dirty flag, and an estimated perplexity-impact summary (per-tier lookup,
not live measurement).

Scope:

- Rewrite `src/diagnostics.rs`:
  - `struct DeviceStatus { total_blocks: u32, used_blocks: u32, free_blocks: u32, utilization_pct: f32, file_count: u32, total_stored_bytes: u64, tier_utilization: HashMap<TensorTier, TierUsage>, quant_profile: QuantProfile, lobotomy: bool, dirty: bool, estimated_perplexity_impact: f32 }`.
  - `struct TierUsage { capacity_bytes: u64, used_bytes: u64, tensor_count: u32 }`.
  - `fn gather(device: &StegoDevice) -> Result<DeviceStatus, DeviceError>`.
  - `fn format_human(status: &DeviceStatus) -> String` — produces the human-readable report for `llmdb status`.
- Estimated perplexity impact (V1 simple heuristic, not a real measurement):
  - Utilization 0 → 0.
  - Tier1 (FFN) used × 0.5 weight.
  - Tier2 (Attn) used × 1.0 weight.
  - Lobotomy used × 5.0 weight.
  - Output is a coarse "low / moderate / severe" bucket plus a numeric score. Real perplexity goes in Task 15's quality harness.
- Wire `status` subcommand in `src/main.rs` (Task 10 leaves it pointed at this module):
  - Open device read-only.
  - Call `gather` and print `format_human` to stdout.
  - Exit 1 if the dirty flag is set and the user did not run `init`/recover in this invocation (the device auto-recovers on open per Task 08, but `status` still flags "this device had an unclean shutdown" as an informational line).
- Tests in `tests/diagnostics.rs`:
  - Fresh device: `utilization_pct == 0.0`, `file_count == 0`, `total_stored_bytes == 0`, `dirty == false`.
  - After storing two files: `file_count == 2`, `total_stored_bytes == sum_of_sizes`, utilization > 0.
  - Lobotomy-initialized device: `lobotomy == true` and Lobotomy-tier capacity appears in `tier_utilization`.
  - `format_human` output contains each required field label (snapshot test with string matching — not full golden-file).

Existing code to reuse / rework / delete:
- Reuse: `StegoDevice::total_blocks`, `used_blocks`, `utilization_pct` from Task 07; `list_files` from Task 09; `TensorMap` metadata
- Rework: `src/diagnostics.rs`
- Delete: the V0 `DiagnosticsBootstrap` placeholder struct

Acceptance criteria:
- `cargo run -- status <fixture.gguf>` after storing a 10-KB file reports file count 1, stored bytes ≥ 10240, utilization > 0, and lists at least one tier in the tier breakdown.
- `cargo test --offline tests::diagnostics` passes.
- Estimated perplexity impact is monotonically non-decreasing as storage utilization grows within a single tier (regression test: store files, re-gather, assert the impact field only goes up).
