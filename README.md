# LLMDB

Steganographic storage that hides a real filesystem in the low-order
bits of weight tensors inside a GGUF file. The backing file stays a
valid GGUF and stays a functional model — even on int-quantized
covers, since V2's anchor + inode + CoW design places writes inside
ceiling-magnitude-bounded free runs rather than zeroing out every
weight at init time.

See [`DESIGN-NEW.MD`](DESIGN-NEW.MD) §15 for the architecture and
[`docs/history/DESIGN-V0.md`](docs/history/DESIGN-V0.md) for the
pre-pivot prototype.

## Build

```
cargo build --release --offline
```

The binary lands at `target/release/llmdb`. No external runtime
dependencies except:

- `fusermount3` (or `fusermount`) on `PATH` + `/dev/fuse` accessible,
  for `mount` / `unmount`. On Debian/Ubuntu/WSL2: `apt install fuse3`.
- `llama-server` on `PATH` (for `ask`).

## Capacity

The practical ceiling is "cover fits on disk", not "cover fits in
RAM". For a 280 GB cover (Llama-3.1-70B-F32, Llama-3.1-405B at
Q5_K, etc.):

- **Cover storage** is memory-mapped, not loaded. `llmdb init` and
  `llmdb mount` open the GGUF read/write and `mmap` it; the OS
  pages bytes in on demand and msyncs dirty pages at unmount.
  RAM use is bounded by the working set (ceiling scan, anchor
  heap, dirty bitmap), not by cover size. See `src/v2/cover.rs`
  for the `CoverStorage` abstraction and `DESIGN-NEW.MD §15.3`.
- **Dirty bitmap** is sparse-page + streaming-serialize
  (`src/v2/dirty.rs`). A 280 GB F16 cover's 17.5 GB dense bitmap
  never materialises as a `Vec<u8>`; only pages with set bits
  allocate (at 4 KB granularity), and commit streams bytes
  through CDC without buffering. `DESIGN-NEW.MD §15.5`.
- **Mount cost** is disk-bound: the anchor-placement heap + the
  ceiling-magnitude scan each walk every stealable weight once.
  At ~3 GB/s on NVMe that's ~90 s for the initial scan on 280
  GB. After that, read-hot paths stay in the page cache.

Proof tests (no real 280 GB model needed):

- `tests/v2_dirty_huge_cover.rs` constructs a synthetic 140 G-weight
  `TensorMap` (17.5 GB worth of bitmap bits) and exercises the
  full bitmap round-trip. `/proc/self/status` RSS delta: ~128 KB.
- `tests/v2_fs_mmap.rs` init/write/remount round-trip over a real
  `MmapMut`-backed cover, plus a 16 GB sparse-file mmap test
  that touches widely-separated pages without OOMing.

What isn't fixed: `read_file()` and `WriteBuffer` are still
cover-size-independent but **app-bounded** — reading a 10 GB file
allocates 10 GB. Not a mount blocker; fix shapes are recorded in
`DESIGN-NEW.MD §15.12`.

## Quickstart: file storage CLI

Works on any GGUF. No mount, no root, no inference runtime needed.

```sh
# Initialise an inode + CoW filesystem on the cover. Writes the
# anchor, an empty root directory, and an empty dirty bitmap, then
# saves the cover bytes back. Re-running discards prior state.
./target/release/llmdb init model.gguf

# Store a host file at an absolute path inside the filesystem.
# Parent directories are created on demand. Store warms a Full
# compensation runtime by default; use --no-compensation only for
# explicit fast/debug writes.
./target/release/llmdb store model.gguf ./notes.txt --stego-path /notes.txt

# List entries (defaults to `/`).
./target/release/llmdb ls model.gguf
./target/release/llmdb ls model.gguf /docs

# Retrieve it.
./target/release/llmdb get model.gguf /notes.txt --output ./notes.out

# Structured status: file count, generation, dirty-bit balance, dedup
# table size, allocator usage, quant profile.
./target/release/llmdb status model.gguf

# Delete a file (or empty directory).
./target/release/llmdb rm model.gguf /notes.txt --yes
```

## Mount as a FUSE filesystem

Unprivileged mount via `fusermount3`. The `mount` command blocks
until `Ctrl-C` or until `llmdb unmount` is run in another shell;
drop it into the background with `&` / `nohup` / `disown` if you
need cross-shell lifetime.

```sh
./target/release/llmdb mount model.gguf /mnt/llmdb

# In another shell:
echo "hello" > /mnt/llmdb/greet.txt
mkdir -p /mnt/llmdb/docs
cp notes.md /mnt/llmdb/docs/
ls /mnt/llmdb/docs

# Clean shutdown:
./target/release/llmdb unmount /mnt/llmdb
```

Files stored via the CLI (`llmdb store … --stego-path /docs/notes.md`)
show up in the mount and vice versa.

## Ask the model about its own files

Needs `llama-server` on `PATH` (from llama.cpp). `ask` spawns a
local `llama-server` against the GGUF, exposes four read-only
tools (`ls`, `read`, `stat`, `list_all_files`), and runs a REPL
where the model sees the embedded filesystem as its own context.

```sh
./target/release/llmdb ask model.gguf
> What files do I have stored?
> Summarize /docs/notes.md.
```

Best on an F16 / F32 cover where post-init inference quality is
unchanged. Q8_0 covers also work — V2 placement keeps perturbation
bounded — but go in with eyes open: text generation can still drift
relative to a pristine model.

## Running the test suite

```sh
cargo test --offline
```

The test suite covers the GGUF parser, every quant packer, the V2
filesystem (anchor, inode codec, indirect pointers, CDC, dedup,
sparse dirty bitmap, hierarchical directories), the FUSE driver
(including concurrent dispatch), the V2 diagnostics report, and
the `ask` tool dispatcher. Several tests that need expensive
fixtures are `#[ignore]`'d by default — run them with
`cargo test --offline -- --ignored` if you have
`models/smollm2-135m-f16.gguf` and the k-quant reference fixture
available.

## Diagnostics and tooling

- `examples/mount_timing.rs` — phase-by-phase wall-clock timing
  for the mount path. Run with `cargo run --release --example
  mount_timing -- <model.gguf>`; prints time spent in each of
  parse / map-build / cover-read / ceiling scan / full mount.
  Regression check for the anchor-placement heap and any future
  mount-cost changes.
- `scripts/perplexity-check.sh <model.gguf>` — three-stage
  perplexity measurement (pristine → post-init → post-write via
  FUSE) on an F16 or int-quant cover. Needs `llama-perplexity`
  (llama.cpp) on PATH or at `$LLMDB_LLAMA_PERPLEXITY`. The
  headline check for "V2 init doesn't kill inference on Q8_0
  covers."
- `LLMDB_E2E_ASK=1 cargo test --test ask_e2e -- --nocapture` —
  end-to-end ask REPL against a real `llama-server` and the
  bundled SmolLM2-135M-F16 model. Gated off by default because
  it needs a built llama.cpp toolchain.
```
