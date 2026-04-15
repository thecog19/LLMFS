# Task 12: BPE Compression Layer

Status: todo
Depends on: `02-gguf-parser-and-quant-model.md`, `10-sqlite-vfs.md`
Spec refs: `DESIGN.MD` section "Compression Layer: BPE Tokenizer"

Objective:
Add the tokenizer-based page compression layer between SQLite pages and stego block writes.

Scope:

- Extract tokenizer vocabulary and merge data from GGUF metadata.
- Implement page compression and decompression with a length header and padding rules.
- Support `--no-compress` at the CLI layer.
- Measure and expose compression ratios per workload.

Acceptance criteria:

- Compressible text-heavy pages roundtrip losslessly through the tokenizer path.
- Incompressible or adversarial data still roundtrips correctly, even when expansion occurs.
- Compression can be disabled cleanly.

Tests first:

- Roundtrip tests for English-like text, JSON-like data, binary data, and adversarial payloads.
- Header-length boundary tests for nearly full pages.
- Tests that `--no-compress` bypasses the compression path entirely.
