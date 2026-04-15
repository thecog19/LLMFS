# Task 03: Tensor Selection And Capacity Planner

Status: todo
Depends on: `02-gguf-parser-and-quant-model.md`
Spec refs: `DESIGN.MD` sections "Tensor Eligibility", "Allocation Order", "`--lobotomy` Mode", "Allocation Priority (Updated)", "Intelligence Gauge v2"

Objective:
Turn parsed tensor inventory into an ordered allocation plan with standard-mode and lobotomy-mode policies.

Scope:

- Encode eligibility rules by tensor name pattern and quant type.
- Implement standard allocation tiers and lobotomy extension tiers.
- Compute per-tensor stealable capacity and whole-device capacity.
- Track per-tier utilization stats needed later by diagnostics.
- Surface unsupported quant types with zero capacity instead of undefined behavior.

Acceptance criteria:

- Given a tensor list, the planner emits a deterministic ordered allocation table.
- Standard mode skips embeddings, LM head, and layer norms.
- Lobotomy mode includes the skipped tiers at the end of the plan.
- Capacity math is exposed per tensor, per tier, and total.

Tests first:

- Table-driven tests for tensor classification by name.
- Tests that standard and lobotomy mode produce different ordered plans.
- Tests that Q2_K contributes zero capacity.
