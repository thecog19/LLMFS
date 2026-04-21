#!/usr/bin/env bash
# Regenerate the K-quant validation fixture used by
# tests/kquant_ggml_reference.rs.
#
# Quantizes models/pristine/smollm2-135m-f16.gguf with llama-quantize
# in default Q4_K_M mode (mixed Q4_K / Q5_K / Q6_K / Q5_0 with F32
# norms). The output GGUF is the cross-implementation reference: our
# Rust decoders must produce values close to the F16 originals
# within Q-quant tolerances.
#
# The output file is gitignored (it's ~100 MB). Re-run this script
# whenever the test wants to re-validate the decoders against ggml.
#
# Requires:
#   - models/pristine/smollm2-135m-f16.gguf  (provenance: see
#     models/pristine/README.md)
#   - llama-quantize binary built from a llama.cpp checkout. The
#     script searches LLMDB_LLAMA_QUANTIZE then common build paths.

set -euo pipefail

SRC="${SRC:-models/pristine/smollm2-135m-f16.gguf}"
OUT_DIR="${OUT_DIR:-benches/fixtures/k-quant}"
OUT_Q3="${OUT_DIR}/smollm2-135m-q3_k_s.gguf"
OUT_Q4="${OUT_DIR}/smollm2-135m-q4_k_m.gguf"
OUT_Q5="${OUT_DIR}/smollm2-135m-q5_k_m.gguf"

QBIN="${LLMDB_LLAMA_QUANTIZE:-}"
if [ -z "$QBIN" ]; then
    for c in \
        "/mnt/c/Users/suero/Documents/code/llama.cpp/build-llmdb/bin/llama-quantize" \
        "/mnt/c/Users/suero/Documents/code/llama.cpp/build/bin/llama-quantize" \
        "$(command -v llama-quantize 2>/dev/null || true)"; do
        if [ -n "$c" ] && [ -x "$c" ]; then QBIN="$c"; break; fi
    done
fi
[ -x "$QBIN" ] || { echo "llama-quantize not found; set LLMDB_LLAMA_QUANTIZE" >&2; exit 1; }
[ -f "$SRC" ] || { echo "missing source: $SRC" >&2; exit 1; }

mkdir -p "$OUT_DIR"
"$QBIN" "$SRC" "$OUT_Q3" Q3_K_S
echo "wrote $OUT_Q3"
"$QBIN" "$SRC" "$OUT_Q4" Q4_K_M
echo "wrote $OUT_Q4"
"$QBIN" "$SRC" "$OUT_Q5" Q5_K_M
echo "wrote $OUT_Q5"
