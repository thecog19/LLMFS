# Task 02: GGUF Parser And Quant Model

Status: todo
Depends on: `01-bootstrap-crate-and-fixtures.md`
Spec refs: `DESIGN.MD` sections "GGUF Internals", "Resolved Decisions", "K-Quant Support"

Objective:
Parse GGUF files well enough to enumerate tensors, metadata, tokenizer assets, and quantization formats for downstream storage mapping.

Scope:

- Parse GGUF header, KV metadata, tensor info table, alignment, and tensor data offsets.
- Support GGUF v2 and v3. Reject unsupported versions explicitly.
- Model tensor metadata with names, shapes, quant types, logical weight counts, and file offsets.
- Extract tokenizer metadata needed later by the compression layer.
- Define the internal quant enum and per-format block metadata.

Acceptance criteria:

- Parser returns a stable in-memory representation of a GGUF file.
- Tensor offsets and quant types roundtrip correctly against fixture expectations.
- Unsupported GGUF versions fail with a clear error.

Tests first:

- Golden tests for GGUF v2 and v3 fixture parsing.
- A failure test for an unsupported version.
- A test that tokenizer metadata is surfaced when present.
