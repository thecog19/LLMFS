# Task 15: Benchmarks And Quality Harness

Status: todo
Depends on: `10-sqlite-vfs.md`, `12-bpe-compression-layer.md`, `14-ask-session-and-nlq.md`
Spec refs: `DESIGN.MD` sections "Capacity Math", "The 'Model Quality' Gauge", "Why ship this", "CLI", "Distribution"

Objective:
Generate the empirical data the design explicitly says must exist instead of being assumed.

Scope:

- Implement `llmdb bench`.
- Benchmark block read/write throughput.
- Benchmark tokenizer compression ratio versus gzip and uncompressed storage.
- Add a quality harness for perplexity and inference-latency measurements at different storage utilizations.

Acceptance criteria:

- The bench command emits reproducible metrics for throughput and compression.
- The quality harness can compare a baseline model against at least one degraded storage level.
- Output is suitable for inclusion in the README or a benchmark report.

Tests first:

- Smoke tests for benchmark command wiring.
- Deterministic tests for metric formatting and result serialization.
- Guardrail tests that fail if required benchmark inputs are missing.
