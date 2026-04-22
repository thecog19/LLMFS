#!/usr/bin/env bash
# Inference-damage validation. For a given GGUF cover, measures
# perplexity at three stages and emits CSV:
#
#   1. pristine     — the unmodified cover straight from disk.
#   2. post_init    — after `llmdb init` (writes anchor + empty
#                     root directory + empty dirty bitmap; no user
#                     data).
#   3. post_write   — after mounting the filesystem, writing
#                     ${PAYLOAD_BYTES} bytes of deterministic
#                     pseudo-random data into a single file, and
#                     unmounting (committing the cover back to disk).
#
# DESIGN-NEW §15's anchor + inode + CoW design keeps Q8_0 covers
# from collapsing on `init` — the V1-era failure mode recorded in
# the memory file. This script is the headline check for that claim.
#
# Deps:
#   llama-perplexity  — env LLMDB_LLAMA_PERPLEXITY, or common paths
#   llmdb             — release binary at ./target/release/llmdb
#   python3           — deterministic payload generation
#   fusermount3       — unmount helper
#
# Usage:
#   scripts/perplexity-check.sh <model.gguf>
#                               [--payload BYTES] [--chunks N]
#                               [--ctx N] [--out-csv PATH]
#                               [--keep-tmp]

set -euo pipefail

usage() {
    cat >&2 <<EOF
usage: $0 <model.gguf> [--payload BYTES] [--chunks N] [--ctx N]
                       [--out-csv PATH] [--keep-tmp]

options:
  --payload BYTES   bytes to write via FUSE at stage 3 (default: 1048576)
  --chunks N        perplexity chunks to evaluate (default: 50)
  --ctx N           context length (default: 512)
  --out-csv PATH    also write CSV to PATH (stdout always gets CSV)
  --keep-tmp        don't delete the working directory after the sweep
EOF
    exit 1
}

MODEL=""
PAYLOAD_BYTES=1048576
CHUNKS=50
CTX=512
OUT_CSV=""
KEEP_TMP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --payload) PAYLOAD_BYTES="$2"; shift 2 ;;
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

# ── test set (wikitext-2) ───────────────────────────────────────────────
FIXTURE_DIR="$(dirname "$0")/../benches/fixtures"
mkdir -p "$FIXTURE_DIR"
WIKI="$FIXTURE_DIR/wiki.test.raw"
if [ ! -f "$WIKI" ]; then
    echo "fetching wikitext-2-raw (ggml-org mirror) ..." >&2
    curl -sL "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip" \
        -o /tmp/wikitext2.zip
    python3 -c "import zipfile; zipfile.ZipFile('/tmp/wikitext2.zip').extractall('/tmp/')"
    cp /tmp/wikitext-2-raw/wiki.test.raw "$WIKI"
    rm -rf /tmp/wikitext2.zip /tmp/wikitext-2-raw
fi

# ── working dir ─────────────────────────────────────────────────────────
WORK="$(mktemp -d /tmp/llmdb-v2-ppl-XXXXXX)"
cleanup() {
    if [ $KEEP_TMP -eq 0 ]; then
        rm -rf "$WORK"
    else
        echo "kept: $WORK" >&2
    fi
}
trap cleanup EXIT

# ── helpers ─────────────────────────────────────────────────────────────
gen_bytes() {
    local out="$1" nbytes="$2"
    python3 - "$out" "$nbytes" <<'PY'
import hashlib, sys
out, nbytes = sys.argv[1], int(sys.argv[2])
seed = b"llmdb-v2-validation-v1"
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

measure_ppl() {
    local model="$1" label="$2"
    local log="$WORK/ppl-${label}.log"
    echo "measuring perplexity [${label}] ..." >&2
    "$PPL_BIN" -m "$model" -f "$WIKI" -c "$CTX" -b "$CTX" --chunks "$CHUNKS" \
        --no-warmup > "$log" 2>&1 || { cat "$log" >&2; exit 3; }
    local csv
    csv=$(parse_ppl "$log")
    [ -n "$csv" ] || { echo "failed to parse PPL from $log" >&2; tail -20 "$log" >&2; exit 4; }
    echo "$csv"
}

wait_for_mount() {
    local mnt="$1"
    for i in $(seq 1 180); do
        if mount | grep -q "llmdb-v2.*$mnt"; then
            return 0
        fi
        sleep 1
    done
    echo "FAIL: mount never appeared at $mnt" >&2
    return 1
}

# ── sweep ───────────────────────────────────────────────────────────────
MODEL_ABS="$(readlink -f "$MODEL")"
MODEL_NAME="$(basename "$MODEL_ABS")"
HEADER="model,stage,bytes_loaded,ppl,ppl_stderr,chunks,ctx"
echo "$HEADER"
if [ -n "$OUT_CSV" ]; then
    mkdir -p "$(dirname "$OUT_CSV")"
    echo "$HEADER" > "$OUT_CSV"
fi

emit() {
    local row="$1"
    echo "$row"
    [ -n "$OUT_CSV" ] && echo "$row" >> "$OUT_CSV"
}

# (1) pristine
echo "=== stage 1: pristine ===" >&2
PRISTINE_PPL=$(measure_ppl "$MODEL_ABS" "pristine")
emit "$MODEL_NAME,pristine,0,$PRISTINE_PPL,$CHUNKS,$CTX"

# (2) post-init: copy → init → measure
WORK_MODEL="$WORK/model-v2.gguf"
cp --no-preserve=mode,ownership "$MODEL_ABS" "$WORK_MODEL"
chmod u+w "$WORK_MODEL"

echo "=== stage 2: init ===" >&2
"$LLMDB" init "$WORK_MODEL" > "$WORK/init.log"
cat "$WORK/init.log" >&2

POSTINIT_PPL=$(measure_ppl "$WORK_MODEL" "post_init")
emit "$MODEL_NAME,post_init,0,$POSTINIT_PPL,$CHUNKS,$CTX"

# (3) post-write: mount → write PAYLOAD bytes → unmount → measure
MNT="$WORK/mnt"
mkdir -p "$MNT"
PAYLOAD="$WORK/payload.bin"
gen_bytes "$PAYLOAD" "$PAYLOAD_BYTES"

echo "=== stage 3: writing ${PAYLOAD_BYTES} bytes via FUSE ===" >&2
"$LLMDB" mount "$WORK_MODEL" "$MNT" > "$WORK/mount.log" 2>&1 &
MOUNT_PID=$!
if ! wait_for_mount "$MNT"; then
    kill "$MOUNT_PID" 2>/dev/null || true
    cat "$WORK/mount.log" >&2
    exit 5
fi

cp "$PAYLOAD" "$MNT/payload.bin"
# Sanity check: content survives through FUSE.
if ! cmp -s "$MNT/payload.bin" "$PAYLOAD"; then
    echo "FAIL: payload readback over FUSE doesn't match source" >&2
    "$LLMDB" unmount "$MNT" || true
    wait "$MOUNT_PID" || true
    exit 6
fi

echo "--- unmounting ---" >&2
"$LLMDB" unmount "$MNT"
wait "$MOUNT_PID" || true
cat "$WORK/mount.log" >&2

POSTWRITE_PPL=$(measure_ppl "$WORK_MODEL" "post_write")
emit "$MODEL_NAME,post_write,$PAYLOAD_BYTES,$POSTWRITE_PPL,$CHUNKS,$CTX"
