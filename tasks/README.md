# LLMDB Task Backlog

This folder turns `DESIGN.MD` into an execution backlog.

Implementation order:

0. `00-testing-strategy.md`
1. `01-bootstrap-crate-and-fixtures.md`
2. `02-gguf-parser-and-quant-model.md`
3. `03-tensor-selection-and-capacity-planner.md`
4. `04-packers-q8-and-float.md`
5. `05-packers-k-quant.md`
6. `06-tensor-map-and-address-translation.md`
7. `07-metadata-layout-and-integrity.md`
8. `08-stego-device-core.md`
9. `09-atomic-writes-and-recovery.md`
10. `10-sqlite-vfs.md`
11. `11-cli-core-commands.md`
12. `12-bpe-compression-layer.md`
13. `13-diagnostics-lobotomy-and-defrag.md`
14. `14-ask-session-and-nlq.md`
15. `15-benchmarks-and-quality-harness.md`
16. `16-raw-mode-storage-engine.md`

Notes:

- The critical path is `01` through `11`. That gets us a usable stego-backed SQLite database in a GGUF file.
- `00` defines the fixture policy so the project does not accidentally depend on a full-size model for correctness tests.
- `12` through `15` add the project-specific differentiators: tokenizer compression, model-health reporting, the `ask` loop, and benchmark data.
- `16` is explicitly optional in V1. The design doc makes SQLite VFS the default and the custom raw engine a fallback.
- Every task assumes tests are written before implementation for the behavior introduced by that task.

Coverage map:

- GGUF parsing and quant support: `02`, `04`, `05`
- Allocation policy and mixed-quant address translation: `03`, `06`
- Metadata, integrity, and recovery: `07`, `08`, `09`
- SQLite and CLI surface: `10`, `11`
- Compression, diagnostics, and product UX: `12`, `13`
- Interactive model loop and benchmark story: `14`, `15`
- Fallback custom storage engine: `16`
