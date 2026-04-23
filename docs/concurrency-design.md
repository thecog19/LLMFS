# Concurrency — design

**Scope:** how LLMDB's mount path coexists with external readers of the cover file (other llama.cpp instances, `ollama`, anything that mmaps the GGUF to run inference), and with LLMDB's own `ask` subcommand. This is the read/writer coordination layer. It sits adjacent to the math in [`compensation-design.md`](compensation-design.md) but addresses a different problem: compensation preserves *model quality across committed states*; this doc is about ensuring external readers never see partial or torn state.

## 1. The problem

The cover GGUF is simultaneously:

- A valid standalone model file that external inference tools can `mmap(MAP_SHARED)` and load like any other GGUF.
- The backing store for V2's filesystem. Every stego write mutates weights in the cover; every compensation update mutates more weights in the cover.

Individual stego + compensation operations touch multiple non-contiguous byte ranges. An external reader with `MAP_SHARED` can observe intermediate state between the payload flip and the compensation adjustment, or between layers of compensation. That intermediate state is an un-compensated perturbation of the model — not a valid LLMDB state, not a reasonable GGUF state.

`compensation-design.md` guarantees that *each committed state* preserves model quality. It makes no claim about transient states visible to an mmap reader mid-write. That's what this document fixes.

## 2. Invariant

**The on-disk cover file always represents the last committed LLMDB state.** Writes during active mount never mutate the on-disk cover in place. Writes accumulate in an in-memory buffer (with scratch-disk spill for large batches, permitted by the at-rest stego invariant — see memory `project_stego_invariant_at_rest`). Commits flush the buffer into a new cover file and install it via `rename(2)`. External readers holding a prior fd continue to see their inode (Linux/POSIX preserve unlinked inodes with live fds); new opens see the new inode.

Between commits, the on-disk cover lags the mount's in-memory state. That lag is the cost of reader-consistency. External readers who need the up-to-date state use `llmdb snapshot` (§4).

## 3. Commit policy

Commit cost depends on the underlying filesystem's atomic-file-replacement primitives:

| Filesystem class | Examples | Atomic-replace cost | Commit cost |
|---|---|---|---|
| Reflink-capable | btrfs, XFS (with `reflink=1`), APFS, ZFS | `O(1)` file-clone + writes to the clone | `O(bytes_written_since_last_commit)` |
| Non-reflink | ext4, FAT, plain XFS, most NTFS | Full file copy before modify | `O(cover_size)` regardless of how little changed |

We probe for reflink capability at mount time, in the directory that holds the cover, via a `FICLONE` ioctl against a disposable probe file. Result drives the commit policy:

### 3.1 Reflink-capable filesystem

- **Background commit thread** runs every `T` seconds (default 10s) flushing any buffered writes into a cloned-and-modified cover, atomic-rename.
- **Commit on explicit `llmdb sync`**, bypassing the timer.
- **Commit on unmount**, final flush of any trailing buffered state.
- Because each commit is delta-bound, the background policy imposes ~O(writes-in-last-10s) of disk I/O — usually negligible.

### 3.2 Non-reflink filesystem

- **No background commits.** Running a periodic O(cover_size) rewrite is a disk-I/O disaster on ext4 for gigabyte-scale covers.
- **Commit on explicit `llmdb sync`.** Users who care about reader-freshness or crash-durability call this.
- **Commit on unmount.** Bounded by one commit cycle; exactly once.
- Staleness of the on-disk cover during active mount is proportional to time since last sync. External readers that accept stale state read the on-disk cover directly; readers that need fresh state use `llmdb snapshot` (next section).

### 3.3 What commit does, precisely

1. Acquire the write lock on the buffer (blocks new writes briefly).
2. Clone or copy the current on-disk cover to `<cover>.commit-<gen>.new`:
   - Reflink if available (`copy_file_range` / `FICLONE`).
   - Full copy via `copy_file_range` with fallback to read-write loop otherwise.
3. Apply every buffered `(offset, new_bytes)` delta to the new file.
4. `fsync` the new file.
5. `rename(2)` the new file over the original path. POSIX guarantees this is atomic at the directory-entry level: any reader with an open fd continues reading the old inode (which lives until their fd closes); any new open sees the new inode.
6. Release the write lock. Discard buffered deltas.

This never mutates the original on-disk inode. External readers with existing mmaps are undisturbed; they continue to see the pre-commit cover until they re-open.

## 4. `llmdb snapshot`

A first-class CLI command that materializes the current (buffered + on-disk) cover state as a standalone GGUF at an explicit user-chosen path:

```
llmdb snapshot <cover-path> <out-path>
```

Semantics:

- Output is a self-contained valid GGUF at `<out-path>`.
- Reflects mount state as of snapshot time; subsequent writes to the mount don't affect it.
- Independent inode from the cover's on-disk file — safe to point any external tool at it.
- User-owned cleanup. Snapshots persist until the user deletes them.

Cost mirrors commit:
- Reflink-capable FS: clone + write buffered deltas to the clone. O(buffered_bytes).
- Non-reflink FS: full cover copy + write buffered deltas. O(cover_size).

`snapshot` is the universal answer to "I want to run [llama.cpp / ollama / random external tool / my own code] against the current cover state while keeping mount active." Document this as *the* supported pattern.

## 5. `llmdb ask` integration

`ask` spawns `llama-server` against the cover. Under this design, it does so through a transparent snapshot:

1. `ask` runs `llmdb snapshot <cover> <session-tmp>/ask-cover.gguf` at session start.
2. `ask` points `llama-server --model <session-tmp>/ask-cover.gguf`.
3. Mount continues to accept writes as normal; they don't affect the `ask` session.
4. `ask` deletes `<session-tmp>/ask-cover.gguf` on session exit (including signal/crash paths — guarded by a temp dir that the process unlinks in its signal handler).

No user ceremony, no flag-setting. The snapshot is invisible to the caller; they just see `ask` behaving correctly while mount remains usable.

## 6. Full outcome matrix

| Scenario | What happens | Cost dimension |
|---|---|---|
| `mount` alone, `unmount` | Buffer → commit via rename on unmount | One commit of last delta |
| `mount` + `llmdb ask` concurrently | `ask` snapshots internally; mount keeps taking writes | Snapshot cost (`O(1)` reflink / `O(size)` non-reflink) |
| `mount` + external tool, wants fresh state | User runs `llmdb snapshot`; points tool at output | Explicit snapshot cost |
| `mount` + external tool, accepts stale state | Tool reads on-disk cover directly; sees last committed state | Zero |
| `mount` + user wants durable write | `llmdb sync` flushes buffer to cover | One commit cost |
| External tool, no mount active | Reads on-disk cover directly | Zero |
| Force-kill during `mount` | On-disk cover unchanged (never mutated in place); uncommitted writes lost | Uncommitted writes lost (no WAL) |
| Force-kill during commit | `rename(2)` is atomic: either old inode or new, never half | None; no corruption |

Every row has a defined outcome. "Don't do X" doesn't appear because no external access pattern is undefined.

## 7. Relationship to `compensation-design.md`

The compensation cache (L1 operators, L2 Cholesky factors, L3 recompute) operates on *in-memory model state*, not the on-disk cover. A write's sequence is:

1. Apply the payload perturbation to the in-memory model state.
2. Run the compensation math (§4 in compensation-design.md) against in-memory state.
3. Append the resulting byte deltas to the commit buffer.
4. Sherman-Morrison update the in-memory L1 cache.

The on-disk cover is touched only at commit time, and only via the rename protocol in §3.3. The in-memory state is always consistent (compensation completes atomically in RAM before the buffer entry is appended); the on-disk state is always consistent (changes only via atomic rename). No intermediate state is ever observable.

Cold-start lazy recompute (compensation-design.md §9) reads from whatever on-disk cover exists at mount time — that's the last committed state by construction, which is exactly the correct input for recomputing `H_layer` and its Cholesky factor.

## 8. Crash and durability

LLMDB does not currently provide a write-ahead log for uncommitted writes. Force-kill during active mount loses any writes since the last commit (sync, unmount, or background commit on reflink-capable FS). This is consistent with "scratch spill is wiped on unmount" — uncommitted state is ephemeral by design.

Users who want durability guarantees for bulk operations call `llmdb sync` explicitly at the appropriate checkpoint. This is the same contract any buffered filesystem offers; LLMDB just exposes the knob rather than hiding it.

Durability-across-crashes with the at-rest stego invariant intact is a larger design question (the WAL would need to live inside the cover file itself via stego, which has its own chicken-and-egg — WAL writes are stego writes that need compensation that need H that needs calibration, etc.). Out of scope for this document; tracked separately.

## 9. Non-goals and what's explicitly not required

- **Synchronous-across-processes reader/writer locks.** External tools don't cooperate with LLMDB. We don't try to force them to; we just make the on-disk cover always look consistent to any reader that cares.
- **Zero-staleness external reads without snapshot.** Not achievable without either (a) forcing external readers to use an LLMDB-provided path (they won't) or (b) committing on every write on non-reflink FS (prohibitive cost). Snapshot is the explicit escape hatch.
- **WAL-backed durability.** Separate design thread. This document assumes sync-or-unmount-or-lose semantics.

## 10. Summary of decisions

| Decision | Choice |
|---|---|
| On-disk cover mutation | Only via atomic rename from a modified clone/copy |
| In-place edits to on-disk cover | Never, during mount |
| Background commit on reflink FS | Every 10s (default), plus sync + unmount |
| Background commit on non-reflink FS | Disabled; explicit sync + unmount only |
| Snapshot command | First-class, `llmdb snapshot <cover> <out>` |
| `ask` coupling to mount writes | Decoupled via internal snapshot at session start |
| External tool contract | Read on-disk cover for stale-OK; `llmdb snapshot` for fresh |
| Force-kill mid-mount | On-disk cover intact; uncommitted writes lost |
| WAL / durability | Separate design thread, not in this doc |
