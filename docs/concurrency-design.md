# Concurrency — design

**Scope:** how LLMDB's mount path coexists with external readers of the cover file (other llama.cpp instances, `ollama`, `llmdb ask`, direct mmap consumers) while mutating the cover via stego writes.

**Companion:** [`compensation-design.md`](compensation-design.md) is the math side — how each write preserves model quality across committed states. This doc is the systems side — what a concurrent reader actually observes.

## 1. The answer, short version

Mount writes the cover in place, via `pwrite`. Readers `mmap(MAP_SHARED)` the cover the way they already do. The kernel's page cache is the shared medium between writer and reader — there is no copy, no snapshot, no buffer. Compensation math (`compensation-design.md` §1) bounds the magnitude of the per-write perturbation, so a reader mid-forward-pass during an active write sees weights that differ from either the pre-write or post-write stable state by a small, quantified amount.

That's the whole design. No coordination protocol, no commit scheduler, no snapshot command, no filesystem-capability probes. The cover file on disk is always the current state, always a valid GGUF, always the stego invariant's at-rest artifact, always the same thing an external reader would see.

## 2. What a concurrent reader actually observes

### 2.1 Mechanism on Linux

1. `mmap(MAP_SHARED, PROT_READ)` creates a VMA pointing at the cover file.
2. On first access to a page, the kernel faults it in from the page cache (reading from disk if not already cached) and maps the physical page into the reader's address space.
3. `pwrite(fd, bytes, offset)` from LLMDB lands in the kernel: it updates the page cache page for that offset and marks it dirty. The reader's mapping shares that same physical page.
4. CPU cache coherency (MESI on x86, equivalent on arm64) ensures the reader's next load from that weight address sees the new bytes. No re-fault, no syscall.

There is exactly one physical memory location for each weight — the page-cache page — and both the reader's mapping and LLMDB's `pwrite` target it.

### 2.2 Atomicity of each byte

- Single-byte writes are atomic at the CPU level.
- Aligned 2/4/8-byte writes are atomic on x86 and on arm64 for aligned addresses.
- A Q8_0 weight is one byte. Compensation updates to Q8_0 weights are one atomic byte write per weight. Nothing tears at the weight level.
- For F16/F32/K-quant weights, the same alignment rule gives atomicity per quantized value.

### 2.3 Where the wobble actually lives

A single compensation operation writes many weights in sequence (the compensation set `C` in `compensation-design.md §1.3`). Each individual write is atomic; the sequence is not. A reader can observe an intermediate state where some weights have the new values and others still have the old ones.

In a forward pass, llama.cpp's matmul kernels load weights into CPU registers for the dot product. Within a single matmul, the weight view is self-consistent — the kernel already read everything it needs. But across matmuls (different layers, different tokens), different ones may see different post-write states. A forward pass that straddles a compensation update produces logits as if the model were a "half-and-half" mix.

Both the pre-compensation-update state and the post-compensation-update state are quality-preserved by design (that's `compensation-design.md`'s whole job). The mixed state is close to both. Logits drift by a small bounded amount from either stable endpoint, for exactly the duration of the in-flight compensation — typically microseconds per write on a CPU-bound L1 hit, per `compensation-design.md §4`.

### 2.4 Quantifying the bound

With 4 stolen bits per Q8_0 weight, the raw steal delta is ≤ 8/128 ≈ 6% of the weight's magnitude. Compensation cancels this to first order across the compensation set. During the window between "payload bits flipped" and "compensation applied," the un-compensated view shows a single-weight 6% perturbation in one layer — an O(1) weight out of billions, with first-order impact on one layer's matmul output, attenuated by normalization layers downstream. The resulting logit shift is a small fraction of typical logit magnitudes; tokens in near-ties may flip; anything with a clear margin doesn't.

Over a burst of many writes, the wobble is proportional to the number of compensation updates currently in-flight simultaneously — bounded by whatever concurrency LLMDB's write path allows (currently single-threaded writes through the mount, so one in-flight compensation at a time).

Between bursts, zero visible wobble.

### 2.5 GPU backends

llama.cpp with CUDA / Metal / Vulkan copies weights from the mmap'd page cache to GPU memory at load time. After the copy completes, weight tensors live in VRAM and the file is not re-read during inference. LLMDB writes to the file become invisible to the running GPU inference — zero wobble. The GPU's VRAM copy is effectively a snapshot, as a side effect of residency, not as an explicit design choice.

Any wobble observed during the LOAD phase (weights streaming from file → CPU RAM → VRAM) is baked into the VRAM copy. The bound from §2.4 applies: the VRAM copy may reflect a mid-write moment, which is a bounded small perturbation from either stable state.

### 2.6 CPU inference without `--no-mmap`

Pure CPU inference with the default mmap path is the most-observable concurrency mode. The page cache is the shared medium; matmul kernels read directly from it; concurrent writes are visible on the next access. Wobble bounds from §2.4 apply in real time.

Users who specifically want zero wobble (and have the RAM for it) can pass `--no-mmap` to llama.cpp, which reads the whole model into process-owned memory at load and decouples subsequent inference from the file entirely. This is a tool, not a requirement. For a 7B Q4 on a 16 GB machine, `--no-mmap` costs ~4 GB RSS — fine. For a 70B F16 on a 32 GB machine, it OOMs. Users choose based on their hardware and tolerance.

## 3. What the design is *not* doing

- **No buffer layer.** Writes go directly to the page cache via `pwrite`.
- **No commit scheduler / rename-and-swap protocol.** The on-disk cover is always the current state, never "last committed."
- **No `llmdb snapshot` command.** Not needed — readers that want a stable copy use `--no-mmap` (process-RAM copy), GPU backends (VRAM copy), or an ordinary `cp` if they want it on disk.
- **No filesystem-capability probes (`FICLONE` / reflink detection).** No code path cares about the FS.
- **No coordination protocol between mount and `ask`.** They both access the cover naturally; compensation bounds the overlap.
- **No scratch-disk spill.** No buffer to spill.

## 4. `llmdb ask`

`ask` launches `llama-server` against the cover with default args — no `--no-mmap`, no snapshot, no special coordination. Mount keeps taking writes while the ask session runs; compensation bounds the visible wobble. For GPU-backend asks (the common production case), the load-time wobble is the only exposure, and it's itself bounded.

Users who want their ask session fully isolated from concurrent writes can pass `--no-mmap` to llama.cpp via whatever CLI plumbing `ask` exposes — but it's an opt-in, not a default, because on large models it's an OOM risk.

## 5. Crash durability

`pwrite` writes land in the page cache immediately and in disk per the kernel's writeback schedule (default 30s dirty timer on Linux). A force-kill between a write and writeback leaves the cover file on disk missing the last few seconds of writes. That's standard POSIX buffered-write behavior.

Users who want durability call `llmdb sync` (a thin wrapper around `fsync` on the cover fd). Unmount calls `fsync` automatically before releasing the fd.

No WAL, no journal, no write-ahead state. Same contract as any in-place POSIX writer. If we ever need crash-durable uncommitted state with the at-rest stego invariant preserved, that's a separate design thread (the WAL would need to live inside the cover itself via stego, which has a chicken-and-egg with calibration) — out of scope here.

## 6. What external readers see at rest

When LLMDB is unmounted, the on-disk cover is whatever it was at unmount time (last `pwrite` + `fsync`). External readers opening the cover at rest see a standalone valid GGUF — the stego-invariant-at-rest artifact. Nothing has changed about this from the original project framing.

During active mount, the on-disk cover is the current state, mutating in real time. External readers see the same thing LLMDB sees: a valid GGUF whose weights are being perturbed at stego-write pace.

## 7. Summary of decisions

| Decision | Choice |
|---|---|
| On-disk cover during mount | Always current, mutated in place via `pwrite` |
| Writer-side buffer | None |
| Commit protocol | None — `pwrite` hits the page cache directly |
| Snapshot mechanism | Whatever readers use anyway (`--no-mmap`, GPU residency, manual `cp`) |
| `llmdb ask` default | Default mmap via `llama-server` |
| Reader/writer coordination | None — compensation bounds the observable wobble |
| Atomicity of individual writes | Per-byte / per-quantized-value, as given by CPU architecture |
| Tolerable wobble bound | Compensation magnitude per in-flight update; quantified in §2.4 |
| Crash durability | POSIX page-cache semantics; `llmdb sync` / unmount call `fsync` |
| FS-dependent behavior | None |
