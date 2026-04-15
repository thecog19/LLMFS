# Task 04: Packers For Q8_0 And Float Tensors

Status: todo
Depends on: `02-gguf-parser-and-quant-model.md`
Spec refs: `DESIGN.MD` sections "Target: 4 LSBs per Q8_0 weight", "Bit Budget Summary", "K-Quant Support"

Objective:
Implement the first concrete bit-stealing routines for the simplest supported formats.

Scope:

- Implement Q8_0 read and write routines for 4 stolen bits per weight.
- Implement F16 and F32 bit stealing per the capacity table.
- Define a shared packer interface used later by the tensor map and device.
- Preserve non-stolen bits exactly.

Acceptance criteria:

- Random logical bytes written through Q8_0 roundtrip without loss.
- The packers modify only the intended low bits.
- Float packers support the configured steal budget without corrupting unrelated bits.

Tests first:

- Property tests for random byte roundtrips through Q8_0.
- Bit-mask tests proving untouched bits remain stable.
- Boundary tests for first and last weight inside a physical block.
