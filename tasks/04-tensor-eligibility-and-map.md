# Task 04: Tensor Eligibility And Tensor Map

Status: todo
Depends on: 03-packing-modules.md
Spec refs: DESIGN-NEW.MD sections "5. Stego Device" (Tensor Eligibility, Tensor Map, Address Translation)

Objective:
Collapse the V0 15-tier allocation planner down to the 4 tiers DESIGN-NEW §5
specifies (`Skip`, `Tier1`, `Tier2`, `Lobotomy`), and verify the tensor map
still produces contiguous bit ranges sorted by allocation priority.

Scope:

- Replace `AllocationTier` in `src/stego/planner.rs` with `TensorTier { Skip, Tier1, Tier2, Lobotomy }` matching §5.
- Replace `classify_tensor_role` with `classify_tensor(name: &str, lobotomy: bool) -> TensorTier` per §5 pseudocode:
  - `token_embd` or `output.weight` → Lobotomy if flag set, else Skip
  - contains `_norm` → Lobotomy if flag set, else Skip
  - contains `ffn_gate` / `ffn_up` / `ffn_down` → Tier1
  - contains `attn_q` / `attn_k` / `attn_v` / `attn_output` → Tier2
  - anything else → Skip
- Remove the per-quant-type tier breakdown (FfnQ80, AttentionQ5K, etc.) — V1 does sequential allocation by tensor tier and layer depth only. Quant type still determines capacity via `stealable_bits_hint()` but no longer fragments the tier list.
- Rewrite allocation ordering: Tier1 before Tier2 before Lobotomy. Within each tier, sort by layer index descending (deepest first). Tie-break by tensor name.
- Update `src/stego/tensor_map.rs` to consume the new `PlannedTensor` shape. The `TensorSlot` / `map_logical_byte` logic is unchanged — it operates on `bit_start` / `bit_end` / `stealable_bits_per_weight`, which survive the planner rework.
- Rewrite `tests/tensor_planner.rs` to assert the new tier taxonomy. Expected tests:
  - `classify_tensor` returns correct tier for FFN, attention, embedding, norm, output, unknown tensor names, with and without lobotomy.
  - Standard mode sorts by tier asc, then layer desc; embeddings/norms/outputs appear only in skipped list.
  - Lobotomy mode appends embeddings/norms/output after tier2.
- `tests/tensor_map.rs` should still pass with minor fixture updates.

Existing code to reuse / rework / delete:
- Reuse: `src/stego/tensor_map.rs` (minor type renames), `tests/tensor_map.rs` (fixture adjustments)
- Rework: `src/stego/planner.rs` (tier taxonomy), `tests/tensor_planner.rs` (assertions)
- Delete: nothing

Acceptance criteria:
- `AllocationTier` enum has exactly 4 variants: `Skip`, `Tier1`, `Tier2`, `Lobotomy`.
- `cargo test --offline` passes for `tensor_planner.rs` and `tensor_map.rs`.
- A tensor named `blk.5.ffn_down.weight` (Q4_K) lands in Tier1 even though V0 would have put it in `FfnQ4K` (tier 9).
- Lobotomy-off plan excludes `token_embd.weight`, `output.weight`, and any `*_norm.weight`; lobotomy-on plan includes them at the end.
