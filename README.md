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

## Quickstart: file storage CLI

Works on any GGUF. No mount, no root, no inference runtime needed.

```sh
# Initialise an inode + CoW filesystem on the cover. Writes the
# anchor, an empty root directory, and an empty dirty bitmap, then
# saves the cover bytes back. Re-running discards prior state.
./target/release/llmdb init model.gguf

# Store a host file at an absolute path inside the filesystem.
# Parent directories are created on demand.
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
dirty bitmap, hierarchical directories), the FUSE driver, the V2
diagnostics report, and the `ask` tool dispatcher. The end-to-end
ask test against a real `llama-server` is gated by `LLMDB_E2E_ASK=1`.
```
