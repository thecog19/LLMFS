# Task 06: Tensor Map And Address Translation

Status: todo
Depends on: `03-tensor-selection-and-capacity-planner.md`, `04-packers-q8-and-float.md`, `05-packers-k-quant.md`
Spec refs: `DESIGN.MD` sections "Block Device Abstraction", "Physical Layout", "Physical I/O", "Mixed-Quantization Files"

Objective:
Map logical bytes and blocks onto concrete tensor locations across mixed quantization formats.

Scope:

- Build the flattened `TensorMap` from the ordered allocation plan.
- Translate logical byte offsets into `(tensor, file offset, quant type, bit position)`.
- Support variable-rate capacity across tensors with different stealable bits per weight.
- Expose block-level address translation for 4096-byte pages.

Acceptance criteria:

- Logical byte offsets map deterministically to valid physical slots.
- Cross-tensor boundaries are handled without data loss or off-by-one errors.
- Mixed-quant fixtures can translate addresses across multiple tensor types.

Tests first:

- Sequential mapping tests over small synthetic mixed-quant fixtures.
- Boundary tests at tensor transitions and block boundaries.
- A roundtrip test that writes across multiple tensors and reads the same bytes back.
