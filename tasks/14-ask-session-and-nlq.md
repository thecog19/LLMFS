# Task 14: Ask Session And Natural-Language Query Loop

Status: todo
Depends on: `10-sqlite-vfs.md`, `11-cli-core-commands.md`, `12-bpe-compression-layer.md`, `13-diagnostics-lobotomy-and-defrag.md`
Spec refs: `DESIGN.MD` sections "Cache Layer: KV Cache", "Index Layer: Attention", "Write-Ahead Log: The Prompt", "Transaction Model: Autoregressive Decoding", "Consistency Model: Temperature", "The `ask` Command: How Recursion Works"

Objective:
Implement the interactive mode where the model queries data stored in its own weights.

Scope:

- Add `llmdb ask` with a model runner bridge to `llama.cpp` or equivalent Rust bindings.
- Implement tool-use routing from natural language to `llmdb query`.
- Track session KV-cache stats and report warm vs cold behavior.
- Support semantic index loading into the prompt and temperature-to-consistency reporting.

Acceptance criteria:

- A session can answer a question by issuing SQL through the tool bridge.
- Follow-up questions can report cache reuse metrics.
- Mutations update the prompt state and index state in-session.

Tests first:

- Mocked tool-use tests for NLQ to SQL bridging.
- Session tests for cache accounting and index updates.
- Determinism tests for temperature `0` mode.
