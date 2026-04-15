# Task 09: Atomic Writes And Recovery

Status: todo
Depends on: `08-stego-device-core.md`
Spec refs: `DESIGN.MD` sections "Failure and Recovery Semantics", "Atomicity Problem", "Write Strategy: Shadow-Copy-Then-Swap", "Crash Recovery Procedure"

Objective:
Make block writes crash-safe enough for SQLite WAL mode and device-level recovery.

Scope:

- Reserve and manage shadow blocks for write swap operations.
- Implement the shadow-copy-then-swap sequence with explicit flush points.
- Track dirty WAL metadata in the superblock.
- Expose recovery checks run by `init` and `status`.

Acceptance criteria:

- A simulated interrupted write never leaves the canonical block partially updated.
- Recovery logic detects dirty state and runs the expected integrity checks.
- Old blocks return to the free list only after a successful swap.

Tests first:

- Crash-injection tests at every write step.
- Tests that interrupted swaps preserve either the old block or the fully swapped block.
- Tests that dirty WAL state is surfaced on reopen.
