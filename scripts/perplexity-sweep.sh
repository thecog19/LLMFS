#!/usr/bin/env bash
# Perplexity saturation sweep. For a given GGUF cover file, runs
# llama-perplexity against wikitext-2 across a set of stego-utilization
# levels and emits CSV to stdout / --out-csv.
#
# Measures inference degradation as a function of how much stego data
# the cover carries. Baseline at 0% (empty device), then up through
# ~95% (nearly full). Random data is deterministic per level so
# re-running reproduces the same bytes; V1 vs V2 comparisons should
# differ only in how the bits land inside the tensors.
#
# Deps:
#   llama-perplexity  — `LLMDB_LLAMA_PERPLEXITY`, or common build paths
#   llmdb             — release binary at ./target/release/llmdb
#   python3           — for deterministic byte generation
#
# Usage:
#   scripts/perplexity-sweep.sh <model.gguf> [--levels 0,10,50,95]
#                               [--chunks 50] [--ctx 512]
#                               [--out-csv path]
#   Levels are percent of stego capacity. Default 0,10,25,50,75,95.

set -euo pipefail

usage() {
    cat >&2 <<EOF
usage: $0 <model.gguf> [--levels LIST] [--chunks N] [--ctx N] [--out-csv PATH]

options:
  --levels LIST    comma-separated percentages (default: 0,10,25,50,75,95)
  --chunks N       perplexity chunks to evaluate (default: 50)
  --ctx N          context length passed to llama-perplexity (default: 512)
  --out-csv PATH   also write CSV to PATH (stdout always gets CSV)
  --keep-tmp       don't delete the working directory after the sweep
EOF
    exit 1
}

MODEL=""
LEVELS="0,10,25,50,75,95"
CHUNKS=50
CTX=512
OUT_CSV=""
KEEP_TMP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --levels) LEVELS="$2"; shift 2 ;;
        --chunks) CHUNKS="$2"; shift 2 ;;
        --ctx) CTX="$2"; shift 2 ;;
        --out-csv) OUT_CSV="$2"; shift 2 ;;
        --keep-tmp) KEEP_TMP=1; shift ;;
        -h|--help) usage ;;
        -*) echo "unknown flag: $1" >&2; usage ;;
        *) if [ -z "$MODEL" ]; then MODEL="$1"; shift; else usage; fi ;;
    esac
done

[ -n "$MODEL" ] || usage
[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }

# ── locate tools ────────────────────────────────────────────────────────
LLMDB="${LLMDB_BIN:-./target/release/llmdb}"
if [ ! -x "$LLMDB" ]; then
    echo "llmdb release binary not found at $LLMDB (set LLMDB_BIN or cargo build --release)" >&2
    exit 1
fi

PPL_BIN="${LLMDB_LLAMA_PERPLEXITY:-}"
if [ -z "$PPL_BIN" ]; then
    for candidate in \
        "/mnt/c/Users/suero/Documents/code/llama.cpp/build-llmdb/bin/llama-perplexity" \
        "/mnt/c/Users/suero/Documents/code/llama.cpp/build/bin/llama-perplexity" \
        "$(command -v llama-perplexity 2>/dev/null || true)"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            PPL_BIN="$candidate"
            break
        fi
    done
fi
[ -x "$PPL_BIN" ] || {
    echo "llama-perplexity not found. Set LLMDB_LLAMA_PERPLEXITY or build llama.cpp." >&2
    exit 1
}

# ── test set ────────────────────────────────────────────────────────────
FIXTURE_DIR="$(dirname "$0")/../benches/fixtures"
mkdir -p "$FIXTURE_DIR"
WIKI="$FIXTURE_DIR/wiki.test.raw"
if [ ! -f "$WIKI" ]; then
    echo "fetching wikitext-2-raw (ggml-org mirror) ..." >&2
    curl -sL "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip" -o /tmp/wikitext2.zip
    python3 -c "import zipfile; zipfile.ZipFile('/tmp/wikitext2.zip').extractall('/tmp/')"
    cp /tmp/wikitext-2-raw/wiki.test.raw "$WIKI"
    rm -rf /tmp/wikitext2.zip /tmp/wikitext-2-raw
fi

# ── working dir ─────────────────────────────────────────────────────────
WORK="$(mktemp -d /tmp/llmdb-ppl-XXXXXX)"
cleanup() {
    if [ $KEEP_TMP -eq 0 ]; then rm -rf "$WORK"; else echo "kept: $WORK" >&2; fi
}
trap cleanup EXIT

# ── helpers ─────────────────────────────────────────────────────────────
# Generate n bytes deterministically from (seed_label, level). Uses SHA-256
# counter-mode so a 50 MB file is cheap to produce and re-running the sweep
# yields identical bytes.
gen_bytes() {
    local out="$1" nbytes="$2" level="$3"
    python3 - "$out" "$nbytes" "$level" <<'PY'
import hashlib, sys
out, nbytes, level = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
seed = f"llmdb-perplexity-sweep-v1-level-{level}".encode()
with open(out, "wb") as f:
    counter = 0
    written = 0
    while written < nbytes:
        h = hashlib.sha256(seed + counter.to_bytes(8, "little")).digest()
        chunk = h[: min(len(h), nbytes - written)]
        f.write(chunk)
        written += len(chunk)
        counter += 1
PY
}

# Capture "total capacity: N bytes" from `llmdb init` output.
init_and_report_capacity() {
    local model="$1"
    "$LLMDB" init "$model" | awk '/total capacity:/ { print $3 }'
}

# Parse "Final estimate: PPL = X +/- Y" from llama-perplexity log.
parse_ppl() {
    awk '
        /Final estimate: PPL/ {
            for (i = 1; i <= NF; i++) {
                if ($i == "=") { ppl = $(i+1) }
                if ($i == "+/-") { sd = $(i+1) }
            }
            printf("%s,%s\n", ppl, sd)
            exit
        }
    ' "$1"
}

# ── sweep ───────────────────────────────────────────────────────────────
MODEL_ABS="$(readlink -f "$MODEL")"
MODEL_NAME="$(basename "$MODEL_ABS")"
# Header
HEADER="model,level_requested_pct,utilization_actual_pct,used_blocks,total_blocks,ppl,ppl_stderr,chunks,ctx"
echo "$HEADER"
if [ -n "$OUT_CSV" ]; then
    mkdir -p "$(dirname "$OUT_CSV")"
    echo "$HEADER" > "$OUT_CSV"
fi

# Pristine baseline: unmodified cover file, no init. For F16 this is
# indistinguishable from the post-init reading; for int-quantized covers
# (Q8_0, K-quants) the delta between pristine and post-init is the
# headline damage from `llmdb init` alone.
echo "=== pristine (pre-init) ===" >&2
PRISTINE_LOG="$WORK/ppl-pristine.log"
"$PPL_BIN" -m "$MODEL_ABS" -f "$WIKI" -c "$CTX" -b "$CTX" --chunks "$CHUNKS" \
    --no-warmup > "$PRISTINE_LOG" 2>&1 || { cat "$PRISTINE_LOG" >&2; exit 3; }
PRISTINE_CSV=$(parse_ppl "$PRISTINE_LOG")
[ -n "$PRISTINE_CSV" ] || { echo "failed to parse PPL from pristine log" >&2; exit 4; }
PRISTINE_ROW="$MODEL_NAME,pristine,0.00,0,0,$PRISTINE_CSV,$CHUNKS,$CTX"
echo "$PRISTINE_ROW"
[ -n "$OUT_CSV" ] && echo "$PRISTINE_ROW" >> "$OUT_CSV"

IFS=',' read -ra LEVEL_ARRAY <<< "$LEVELS"
for LEVEL in "${LEVEL_ARRAY[@]}"; do
    echo "=== level ${LEVEL}% ===" >&2

    WORK_MODEL="$WORK/model-${LEVEL}.gguf"
    # --no-preserve=mode so a 0444 source (models/pristine/*) doesn't
    # propagate its read-only bit onto the mutable working copy.
    cp --no-preserve=mode,ownership "$MODEL_ABS" "$WORK_MODEL"
    chmod u+w "$WORK_MODEL"

    CAPACITY=$(init_and_report_capacity "$WORK_MODEL")
    [ -n "$CAPACITY" ] || { echo "failed to parse capacity from init output" >&2; exit 2; }

    if [ "$LEVEL" -gt 0 ]; then
        # target bytes = level% of capacity, minus ~28 metadata blocks.
        # Overshoot a little — llmdb store will refuse overfills cleanly.
        TARGET=$(python3 -c "print(int($CAPACITY * $LEVEL / 100.0) - 4096 * 32)")
        if [ "$TARGET" -gt 0 ]; then
            PAYLOAD="$WORK/payload-${LEVEL}.bin"
            gen_bytes "$PAYLOAD" "$TARGET" "$LEVEL"
            echo "storing ${TARGET} bytes at level ${LEVEL}%..." >&2
            "$LLMDB" store "$WORK_MODEL" "$PAYLOAD" --name saturate.bin > /dev/null
            rm -f "$PAYLOAD"
        fi
    fi

    # Snapshot actual utilization.
    USED=$("$LLMDB" status "$WORK_MODEL" | awk '/^used:/ { print $2 }')
    TOTAL=$("$LLMDB" status "$WORK_MODEL" | awk '/^total:/ { print $2 }')
    UTIL=$(python3 -c "print(f'{100.0 * $USED / $TOTAL:.2f}')")

    echo "measuring perplexity..." >&2
    PPL_LOG="$WORK/ppl-${LEVEL}.log"
    "$PPL_BIN" -m "$WORK_MODEL" -f "$WIKI" -c "$CTX" -b "$CTX" --chunks "$CHUNKS" \
        --no-warmup > "$PPL_LOG" 2>&1 || { cat "$PPL_LOG" >&2; exit 3; }

    PPL_CSV=$(parse_ppl "$PPL_LOG")
    [ -n "$PPL_CSV" ] || { echo "failed to parse PPL from log:" >&2; tail -20 "$PPL_LOG" >&2; exit 4; }

    ROW="$MODEL_NAME,$LEVEL,$UTIL,$USED,$TOTAL,$PPL_CSV,$CHUNKS,$CTX"
    echo "$ROW"
    [ -n "$OUT_CSV" ] && echo "$ROW" >> "$OUT_CSV"
done
