# Pristine model cache

Bit-for-bit reference copies of cover GGUFs used by benchmarks and
E2E tests. Never pass a file under this directory to `llmdb init`,
`llmdb store`, `llmdb mount`, or anything else that opens the
device for write — always copy to scratch first.

The files are `chmod 0444` as a hard guard: `StegoDevice::open_*`
opens the file `read + write`, so any write-oriented path refuses
to mmap a pristine GGUF and errors out instead of quietly
corrupting it. The `scripts/perplexity-sweep.sh` harness already
copies the source before touching it.

## Provenance

| File | Source |
|------|--------|
| `smollm2-135m-f16.gguf` | [bartowski/SmolLM2-135M-Instruct-GGUF](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF) — F16 variant |
| `smollm2-135m-q8_0.gguf` | [bartowski/SmolLM2-135M-Instruct-GGUF](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF) — Q8_0 variant |
| `qwen2.5-0.5b-instruct-f16.gguf` | [bartowski/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF) — F16 variant |
| `qwen2.5-0.5b-instruct-q8_0.gguf` | [bartowski/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF) — Q8_0 variant |

The GGUFs themselves are gitignored (they're >100 MB). Only this
README and `MANIFEST.sha256` live in the repo. Redownload with:

```
curl -sL -o models/pristine/smollm2-135m-f16.gguf \
    "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-F16.gguf"
curl -sL -o models/pristine/smollm2-135m-q8_0.gguf \
    "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf"
curl -sL -o models/pristine/qwen2.5-0.5b-instruct-f16.gguf \
    "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-f16.gguf"
curl -sL -o models/pristine/qwen2.5-0.5b-instruct-q8_0.gguf \
    "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q8_0.gguf"
chmod 0444 models/pristine/*.gguf
```

## Verification

Run from the repo root:

```
(cd models/pristine && sha256sum -c MANIFEST.sha256)
```

Must print `OK` for every file. A mismatch means the cache is
contaminated — delete the file, redownload, re-chmod, and rerun
any benchmarks that depended on it.

## Why this exists

Earlier in the project, `models/smollm2-135m-q8_0.gguf` was quietly
stego-mutated by a prior session. `llmdb status` reported no
superblock (someone had wiped metadata) but the weights carried
residual perturbation — perplexity on wikitext-2 came back at 24.8
million instead of the expected ~18. Anything benchmarked against
that file produced nonsense numbers until we caught it.

The pristine folder exists so that never happens again: one
explicit directory with a verifiable manifest, protected at the
filesystem layer. Everything outside `models/pristine/` is
considered scratch — free to mutate, not safe to benchmark against.
