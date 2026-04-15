# Task 02: GGUF Parser Extensions

Status: done
Depends on: 01-repo-reset-and-dependency-baseline.md
Spec refs: DESIGN-NEW.MD sections "3. GGUF Parser", "4. Quantization Block Layouts"

Objective:
Extend the V0 GGUF parser to match the enum DESIGN-NEW §3 specifies. The V0
parser already handles headers, KV metadata (all scalar/array/string types),
tensor info, and alignment padding for v2 and v3 — that stays. What needs
to change is the quant-type enum (V0 has 8 variants, §3 lists 13+), and
a note about what V1 actually uses vs what the enum declares.

Scope:

- Extend `GgufQuantType` in `src/gguf/quant.rs` with Q4_0, Q4_1, Q5_0, Q5_1, Q8_1, Q8_K variants. These are declared for completeness (§3 enum) but not supported for stego storage in V1; they fall into the Skip tier via `stealable_bits_hint() == 0`.
- Add `GGML_TYPE_*_ID` constants for the new variants.
- Update `from_raw_ggml_type` to dispatch the new variants.
- Update `stealable_bits_hint` — Q4_0 through Q8_1 get 0 bits (not supported for stego in V1), Q8_K gets 0 bits. Q2_K already returns 0.
- Keep `src/gguf/parser.rs` unchanged. The parser is agnostic to quant type — it just stores `raw_type_id: u32`.
- Add a unit test for each new variant's `from_raw_ggml_type` mapping.
- Add a unit test asserting every unsupported-for-stego variant (Q2_K, Q4_0, Q4_1, Q5_0, Q5_1, Q8_1, Q8_K) has `stealable_bits_hint() == 0`.

Existing code to reuse / rework / delete:
- Reuse: `src/gguf/parser.rs` (no changes), `src/gguf/mod.rs`, `tests/gguf_parser.rs`, `tests/fixture_strategy.rs`, `tests/common/mod.rs`
- Rework: `src/gguf/quant.rs`, `tests/tensor_planner.rs` (only the `raw_ggml_type_ids_map_to_supported_quant_types` test — add assertions for the new variants)
- Delete: nothing (the tokenizer placeholder dies in Task 01)

Acceptance criteria:
- `cargo test --offline` passes, including new variant-mapping tests.
- `grep -R "GgufQuantType::" src/` shows dispatch arms for all 14 variants (F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K).
- A synthetic fixture with a Q4_0 tensor parses without error and is skipped during tensor map construction (by Task 04's classifier).
