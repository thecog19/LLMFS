# Task 13: Diagnostics, Lobotomy, And Defrag

Status: todo
Depends on: `03-tensor-selection-and-capacity-planner.md`, `08-stego-device-core.md`, `09-atomic-writes-and-recovery.md`, `12-bpe-compression-layer.md`
Spec refs: `DESIGN.MD` sections "`--lobotomy` Mode", "The 'Model Quality' Gauge", "Intelligence Gauge v2", "CLI"

Objective:
Expose the model-health story and the operational tools that depend on allocation tiers.

Scope:

- Implement the intelligence gauge and per-tier degradation reporting.
- Add lobotomy-mode warnings and per-component integrity reporting.
- Implement `llmdb defrag` to move data toward higher-priority tiers when space permits.
- Include compression ratio and capacity stats in diagnostics output.

Acceptance criteria:

- `status` reports total storage use, tier use, and estimated degradation.
- Lobotomy mode clearly exposes the extra-risk tiers in output.
- `defrag` measurably reduces low-priority tier occupancy when eligible space exists.

Tests first:

- Diagnostic rendering tests from synthetic tier-utilization snapshots.
- Lobotomy-mode tests for warning text and extra integrity fields.
- Defrag tests proving data remains readable after block migration.
