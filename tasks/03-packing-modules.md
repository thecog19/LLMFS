# Task 03: Packing Modules

Status: done
Depends on: 02-gguf-parser-extensions.md
Spec refs: DESIGN-NEW.MD sections "4. Quantization Block Layouts"

Objective:
Confirm the V0 packing modules match the bit operations specified in
DESIGN-NEW §4, and introduce the common `QuantPacker` trait the new design
names explicitly. The V0 modules implement the correct bit algorithms for
all six supported types (Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, F16/F32) with roundtrip
tests that preserve non-stolen bits. What's missing is a unified trait so
callers don't dispatch on `GgufQuantType` by hand.

Scope:

- Define `QuantPacker` trait in `src/stego/packing/mod.rs` per §4:
  - `bits_per_weight() -> u32`
  - `block_size_bytes() -> usize`
  - `weights_per_block() -> usize`
  - `extract(&self, block_bytes: &[u8]) -> Vec<u8>` (returns payload bytes per DESIGN-NEW signature; keep the existing `read_payload_block`/`write_payload_block` as the underlying impl)
  - `embed(&self, block_bytes: &[u8], data: &[u8]) -> Vec<u8>`
  - `stealable_byte_offsets() -> Vec<usize>` — used by the future hot-path optimization to batch reads within a quant block
- Implement the trait for each of the 7 packers (Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, F16Packer, F32Packer). Keep the existing free functions; the trait methods call into them.
- Confirm the existing `read_payload_block` / `write_payload_block` pairs match the §4 snippets:
  - Q8_0: 34 bytes, 4 LSBs per int8 quant, scale bytes preserved. ✓ V0 verified.
  - Q6_K: 210 bytes, 2 LSBs per nibble in ql, qh/scales/d preserved. ✓ V0 verified.
  - Q5_K: 176 bytes, 1 LSB per nibble in qs, qh/scales/d/dmin preserved. ✓ V0 verified.
  - Q4_K: 144 bytes, 1 LSB per nibble in qs, scales/d/dmin preserved. ✓ V0 verified.
  - Q3_K: 110 bytes, 1 LSB per 2-bit pair in qs, hmask/scales/d preserved. ✓ V0 verified.
  - F16: 2 bytes, 4 LSBs of mantissa (low byte). ✓ V0 verified.
  - F32: 4 bytes, 8 LSBs of mantissa (low byte). ✓ V0 verified.
- Q2_K: implement a trait method that returns `bits_per_weight() == 0` and `extract`/`embed` as unreachable — this keeps dispatch tables uniform.
- Add a dispatch helper `fn packer_for(quant_type: GgufQuantType) -> &'static dyn QuantPacker` to replace `StegoDevice::decode_slot` / `encode_slot` match statements in Task 07.

Existing code to reuse / rework / delete:
- Reuse: `src/stego/packing/{q8_0.rs, q6_k.rs, q5_k.rs, q4_k.rs, q3_k.rs, float.rs}` (free functions unchanged), `tests/quant_packing.rs`, `tests/k_quant_packing.rs`
- Rework: `src/stego/packing/mod.rs` (add trait, dispatch helper)
- Delete: nothing

Acceptance criteria:
- `cargo test --offline` passes all existing packer tests (unchanged).
- New trait-dispatch test: for each supported quant type, `packer_for(t).block_size_bytes()` and `.bits_per_weight()` match §4 table.
- `cargo clippy --offline --all-targets` has no new warnings introduced by the trait layer.
