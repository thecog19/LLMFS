# Perplexity saturation sweeps

Per-model CSVs measuring wikitext-2 perplexity as a function of LLMDB
stego utilization. Each row is one llama-perplexity run; the `pristine`
row is the cover file before any LLMDB touches it. All other rows are
post-`llmdb init` with the named percentage of stego capacity filled
by deterministic random bytes (SHA-256 counter mode, seeded by level
so re-runs reproduce exactly).

## How to run

```
cargo build --release
bash scripts/perplexity-sweep.sh models/pristine/<name>.gguf \
    --levels 0,10,25,50,75,95 \
    --chunks 50 --ctx 512 \
    --out-csv benches/perplexity-results/<name>.csv
```

Always point at `models/pristine/` — those GGUFs are `chmod 0444`
and match `MANIFEST.sha256`, so the sweep starts from a known-good
baseline. The sweep copies each file with `--no-preserve=mode` to a
scratch working dir before touching it; the source stays untouched.

Requires `llama-perplexity` from llama.cpp. The script looks at
`$LLMDB_LLAMA_PERPLEXITY`, then the common build paths under
`llama.cpp/build*/bin/`. `wiki.test.raw` is downloaded to
`benches/fixtures/` on first run.

## Reading the CSV

```
model,level_requested_pct,utilization_actual_pct,used_blocks,total_blocks,ppl,ppl_stderr,chunks,ctx
```

- `level_requested_pct` is the sweep input (`pristine` for the unmodified
  baseline, or a percentage).
- `utilization_actual_pct` is `100 * used_blocks / total_blocks` after
  the store — slightly above the target because metadata also counts.
- `ppl` / `ppl_stderr` come from `llama-perplexity`'s "Final estimate"
  line.

## Current headline findings

**F16, SmolLM2-135M** (`smollm2-135m-f16.csv`): perplexity flat
across the full sweep. Pristine 18.80 → 95%-full 18.82, every step
inside ±0.49 stderr. Writing 50 MB into the low 12 mantissa bits of
an F16 cover is invisible to wikitext-2 perplexity.

**F16, Qwen2.5-0.5B** (`qwen2.5-0.5b-instruct-f16.csv`): same story,
different scale. Pristine 14.01 → 95%-full 14.03, every step inside
±0.35 stderr. A 4× larger model carries 175 MB of stego payload (vs
SmolLM2-135M's 50 MB) with no measurable perplexity impact. The F16
result is not SmolLM-specific.

**Q8_0, SmolLM2-135M** (`smollm2-135m-q8_0.csv`): catastrophic
collapse on `llmdb init` alone. Pristine 18.12 → post-init (0% user
data) 34,167,762. That's ~1.9 million× degradation before any stego
payload lands — every subsequent row in the 10–95% range sits
between 30M and 73M. The 4-bit theft from each int8 weight
randomizes magnitudes; the model stops being a model.

All results match the empirical findings already captured in
`DESIGN-NEW.MD §2` and `project_v1_stego_destroys_inference`. These
CSVs are the first quantitative evidence, reproducible per-commit.

## V2 comparison protocol

V2's sensitivity-aware allocator should preserve the F16 property
while narrowing the Q8_0 gap — ideally close to the pristine ceiling
on both. When that work lands, regenerate both CSVs with an identical
invocation and diff. A same-cover, same-eval-config, same-seed
comparison isolates the allocation-strategy delta.
