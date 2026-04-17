# Task 15: Benchmarks And Quality Harness

Status: deferred (post-V1)
Depends on: 09-file-table-and-file-ops.md
Spec refs: DESIGN-NEW.MD section "11. Testing Strategy" (Benchmarks)

Objective:
Populate the currently empty `benches/` directory with Criterion benchmarks for
the hot paths, and add a scaffolded quality harness (not a full perplexity
measurement — that is a V2/V3 concern) that can compare a baseline vs a
degraded model on a canned inference prompt.

Scope:

- Add `criterion` to `[dev-dependencies]` in `Cargo.toml` (do this in Task 01 if convenient, or here).
- Create `benches/block_read.rs`:
  - Format a synthetic GGUF fixture with ≥64 Q8_0 tensors (enough capacity for ≥10,000 stego blocks).
  - Benchmark: read 1000 random blocks. Report throughput in blocks/s and MB/s.
- Create `benches/block_write.rs`:
  - Same fixture shape.
  - Benchmark: write 1000 random blocks. Report throughput.
- Create `benches/address_translation.rs`:
  - Benchmark `TensorMap::map_logical_byte` across the full capacity.
  - This is the inner loop DESIGN-NEW §14 flags as the top optimization candidate.
- Add `benches/file_roundtrip.rs`:
  - Measure end-to-end store_file → get_file for files of 1 KB, 100 KB, 1 MB.
- Quality harness `benches/quality.rs` (guarded by `LLMDB_E2E_GGUF`):
  - Accept a real GGUF path.
  - Measure: run inference on a fixed prompt via `llama-cli` or the Task 13 `LlamaServer`, capture the first 100 tokens, compute byte-level hash as a coarse "did the output change" signal.
  - Baseline: unmodified model. Comparison: model with X% stego utilization. Report both hashes and a diff count.
  - This is NOT a perplexity benchmark; it is a "did we notice" regression detector. Real perplexity goes in a V2 task.
- `cargo bench` should run all four throughput benches on a synthetic fixture without a real model.
- Document bench invocation in `tasks/README.md`-style notes inside this task file.

Existing code to reuse / rework / delete:
- Reuse: `tests/common/mod.rs` for synthetic fixture construction (will need to re-export helpers so benches can import them; simplest path is a `llmdb-test-fixtures` internal crate, but V1 can just copy helpers into `benches/common.rs`)
- Rework: `Cargo.toml` (add `[[bench]]` entries)
- Delete: nothing

Acceptance criteria:
- `cargo bench --offline` runs all four throughput benches to completion (no `LLMDB_E2E_GGUF` required for the first three).
- Output includes blocks/sec and MB/sec for read and write.
- Address-translation benchmark shows operations/sec.
- The quality harness runs with `LLMDB_E2E_GGUF=...` set and produces a comparison line without crashing.

Prior empirical data (2026-04-17, ad-hoc smoke test, not yet harnessed):

Ran `llmdb init` + `llmdb store` on real GGUFs with greedy
temperature=0 completion as the quality signal.

- **F16 cover** (SmolLM2-135M-f16, 53 MB stego capacity): inference
  survives end-to-end. Post-init output identical to baseline. At
  48% utilization (25 MB stored) outputs diverge in minor word-choice
  on a few prompts but stay coherent. At 97% utilization (50 MB
  stored, ~50M weight nibbles randomized) the model still produces
  fluent English and correct Python for a Fibonacci prompt.
- **Q8_0 cover** (stories15M-q8_0, smollm2-135m-q8_0): inference
  collapses into degenerate-repetition garbage after `llmdb init`
  alone (before any user data is stored).

When this task lands, the harness should reproduce both results
automatically and bake them into `cargo bench` output so we can track
quality regressions over time. See the DESIGN-NEW.MD §2 V1 "Cover-file
viability" paragraph for the root-cause explanation (int-quant packer
destroys magnitude; float packer preserves sign+exponent).
