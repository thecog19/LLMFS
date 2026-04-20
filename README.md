# LLMDB

Steganographic block storage that hides a real filesystem in the
low-order bits of quantized model weights inside a GGUF file. The
backing file stays a valid GGUF and — for F16 / F32 cover files —
stays a functional model. See
[`DESIGN-NEW.MD`](DESIGN-NEW.MD) for the architecture and
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
- `llama-server` on `PATH` (for `ask`)

## Cover-file viability

V1's dual-use property — the GGUF is both a working model and a
block store — holds for **F16 / F32** cover files only. The
mantissa-only mask (`0xFFF0` / `0xFFFF_FF00`) preserves sign and
exponent, so weight magnitudes survive at reduced precision.

Int-quantized covers (Q8_0, Q6_K, Q5_K, Q4_K, Q3_K) are
**storage-only** in V1. Stealing 4 bits from every int8 weight
randomizes magnitudes; stories15M-q8_0 and smollm2-135m-q8_0
collapse into degenerate repetition after `llmdb init` alone.
Restoring dual-use on int covers is V2's job (sensitivity-aware
allocation). See `DESIGN-NEW.MD §2`.

## Quickstart: file storage CLI

Works on any GGUF. No mount, no root, no inference runtime needed.

```sh
# Prepare a GGUF as a stego device. Zeroes the stealable bits and
# writes the superblock / redirection / file table / integrity blocks.
./target/release/llmdb init model.gguf

# Store a host file into the device.
./target/release/llmdb store model.gguf ./notes.txt --name notes.txt

# List contents (add -l for mode/size/mtime).
./target/release/llmdb ls model.gguf -l

# Retrieve it.
./target/release/llmdb get model.gguf notes.txt --output ./notes.out

# Full integrity scan (walks every data block's CRC).
./target/release/llmdb verify model.gguf

# Capacity / tier / perplexity-impact report.
./target/release/llmdb status model.gguf

# Delete a stored file.
./target/release/llmdb rm model.gguf notes.txt
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

Storage is flat under the hood — virtual directories come from
`/`-separated filenames and persist as long as there are files
under them. Empty directories live only in the mount session
(touch a `.keep` file to make one persistent). Files stored via
the CLI (`llmdb store ... --name docs/notes.md`) show up in the
mount and vice versa.

## Ask the model about its own files

Needs `llama-server` on `PATH` (from llama.cpp). `ask` spawns a
local `llama-server` against the GGUF, exposes three read-only
tools (`list_files`, `read_file`, `file_info`), and runs a REPL
where the model sees the stego filesystem as its own context.

```sh
./target/release/llmdb ask model.gguf
> What files do I have stored?
> Summarize notes.txt.
```

Only meaningful on an F16 / F32 cover file where the model still
inferences after `init`.

## Diagnostics

```sh
./target/release/llmdb dump-block model.gguf 4    # hex of one block
./target/release/llmdb verify model.gguf          # CRC scan
./target/release/llmdb status model.gguf          # generation, tiers, utilization
```

## Running the test suite

```sh
cargo test --offline
```

The test suite covers the GGUF parser, every quant packer, the
redirection table, shadow-copy crash points, the file table
(including chain extension), the FUSE driver via real kernel
mounts (skipped gracefully if `/dev/fuse` or `fusermount` is
missing), and the `ask` tool dispatcher.
