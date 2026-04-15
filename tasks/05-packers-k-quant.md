# Task 05: Packers For K-Quant Formats

Status: todo
Depends on: `02-gguf-parser-and-quant-model.md`
Spec refs: `DESIGN.MD` sections "K-Quant Block Layouts", "Bit Budget Summary"

Objective:
Support every K-quant format promised by the design, with explicit refusal for `Q2_K`.

Scope:

- Implement packers for `Q6_K`, `Q5_K_*`, `Q4_K_*`, and `Q3_K_*`.
- Encode the per-format bit budget and physical extraction logic.
- Refuse writes for `Q2_K` with a clear unsupported-capacity result.
- Normalize all variants behind the same packer trait used by later layers.

Acceptance criteria:

- Each supported format has a verified byte roundtrip path.
- `Q2_K` is never treated as writable storage.
- The effective stolen-bit counts match the design doc.

Tests first:

- Property tests for each supported quant format.
- Fixture tests covering packed nibble and bitfield extraction.
- A negative test proving `Q2_K` yields zero capacity and no write path.
