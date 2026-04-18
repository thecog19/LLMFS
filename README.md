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

- `nbd-client` + the `nbd` kernel module (for `mount` / `unmount`)
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

Works on any GGUF. No NBD, no root, no inference runtime needed.

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

## Mount as ext4

Requires root and the `nbd` kernel module. The `mount` command
runs the full stack: NBD server → `nbd-client` → optional
`mkfs.ext4` → `mount`, then blocks until `Ctrl-C` or until
`llmdb unmount` is invoked from another shell.

If you want to drive the root-only steps through the repo helper
instead of granting broader sudo, set `LLMDB_ROOT_HELPER` to
[`scripts/llmdb-e2e-root.sh`](scripts/llmdb-e2e-root.sh). The CLI
will route `nbd-client`, `mkfs.ext4`, `mount`, and `umount`
through that helper.

```sh
sudo modprobe nbd nbds_max=16
sudo ./target/release/llmdb mount model.gguf /mnt/llmdb --format --yes

# In another shell — read/write files through ext4:
sudo cp somefile /mnt/llmdb/
ls /mnt/llmdb

# Clean shutdown:
sudo ./target/release/llmdb unmount /mnt/llmdb
```

With the helper script installed in `sudoers`, the same flow can run
without `sudo` on the `llmdb` command itself:

```sh
export LLMDB_ROOT_HELPER="$PWD/scripts/llmdb-e2e-root.sh"
./target/release/llmdb mount model.gguf /mnt/llmdb --format --yes
./target/release/llmdb unmount /mnt/llmdb
```

The device is persistent — the ext4 superblock and all file data
are hidden inside the GGUF's stego space. Remount the same GGUF
later and your files are still there.

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

## Manual NBD flow

For debugging or if you prefer to drive `nbd-client` / `mkfs`
yourself:

```sh
./target/release/llmdb serve model.gguf      # binds Unix socket
# — in another shell —
sudo nbd-client -unix /tmp/llmdb.sock /dev/nbd0
sudo mkfs.ext4 -F /dev/nbd0
sudo mount /dev/nbd0 /mnt/llmdb
```

## Diagnostics

```sh
./target/release/llmdb dump-block model.gguf 4    # hex of one block
./target/release/llmdb verify model.gguf          # CRC scan
LLMDB_NBD_TRACE=1 ./target/release/llmdb mount ... # per-request NBD log
```

## Running the test suite

```sh
cargo test --offline
```

The test suite covers the GGUF parser, every quant packer, the
redirection table, shadow-copy crash points, the file table,
the NBD wire protocol, the NBD socket roundtrip, and the
`ask` tool dispatcher.
