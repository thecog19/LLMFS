# LLMDB V1 Task Backlog

This folder turns `DESIGN-NEW.MD` into an execution backlog for the V1 scope:
naive sequential stego, flat file storage, NBD mount, `ask` via llama-server.

V2 (AWQ/Hessian calibration, BPE compression, sensitivity-ordered allocation)
and V3 (GPU scatter map, VRAM hot/warm/cold tiering) are explicitly out of scope
for this backlog and will be specced separately per `DESIGN-NEW.MD §2`.

## Prior state: V0 base

The V0 base commit captures the pre-pivot prototype (SQLite VFS, BPE/NLQ
placeholders, 15-tier planner). The V1 tasks below reuse, rework, or delete
individual pieces of that tree — each task's "Existing code" section names what
survives and what dies. Nothing in V0 landed against `DESIGN-NEW.MD`; every V1
task file here is new.

## Implementation order

| # | Task | Maps to DESIGN-NEW § | Net effect on V0 tree |
|---|------|---------------------|---------------------|
| 00 | `00-testing-strategy.md` | §11 | Replaces V0 strategy; keeps synthetic-fixture policy |
| 01 | `01-repo-reset-and-dependency-baseline.md` | §2, §10 | Drops `rusqlite`, adds NBD + HTTP deps |
| 02 | `02-gguf-parser-extensions.md` | §3 | Reuses `src/gguf/parser.rs`, extends quant enum |
| 03 | `03-packing-modules.md` | §4 | Reuses all six packers as-is |
| 04 | `04-tensor-eligibility-and-map.md` | §5 (tensor map) | Reworks `src/stego/planner.rs` (15 tiers → 3) |
| 05 | `05-superblock-and-integrity.md` | §5 | Reworks `src/stego/integrity.rs` (new layout) |
| 06 | `06-redirection-table.md` | §5 | Net-new `src/stego/redirection.rs` |
| 07 | `07-stego-device-block-io.md` | §5 | Reworks `src/stego/device.rs` (redirection-based swap) |
| 08 | `08-shadow-copy-and-recovery.md` | §13 | Reworks atomic-write path; drops pending-metadata recovery |
| 09 | `09-file-table-and-file-ops.md` | §6 | Net-new `src/fs/` |
| 10 | `10-cli-file-commands.md` | §9 | Rewires `src/main.rs` for store/get/ls/rm/verify/dump/wipe |
| 11 | `11-nbd-protocol.md` | §7 | Net-new `src/nbd/protocol.rs` — **superseded post-v1; see DESIGN-NEW §7 (FUSE)** |
| 12 | `12-nbd-server.md` | §7 | Net-new `src/nbd/server.rs`, wires `mount`/`unmount` — **superseded post-v1; see DESIGN-NEW §7 (FUSE)** |
| 13 | `13-ask-bridge.md` | §8 | Net-new `src/ask/`, spawns `llama-server` |
| 14 | `14-diagnostics-and-status.md` | §9 | Real `src/diagnostics.rs`, wires `status` output |
| 15 | `15-benchmarks.md` | §11 | Net-new `benches/` |
| 16 | `16-v1-cleanup-and-docs.md` | — | Deletes dead V0 modules, writes README, archives DESIGN-OLD |

## Critical path

`01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10` is the shortest route to a
tested file-ops CLI: the user can `init`, `store`, `get`, `ls`, `rm`, `verify`,
`dump`, and `wipe` a GGUF-backed stego device without NBD or inference. That
surface is the smallest thing that proves the pivot landed. `11-12` adds NBD
and `mount`; `13` adds `ask`; `14-16` finishes V1 polish.

## Task file format

Every task file has these sections, in this order:

```markdown
# Task NN: <Title>

Status: todo | in-progress | active | done | blocked
Depends on: <task file refs, or "none">
Spec refs: DESIGN-NEW.MD sections "<section names>"

Objective:
<one-paragraph outcome>

Scope:
- <bullet>

Existing code to reuse / rework / delete:
- Reuse: <paths>
- Rework: <paths>
- Delete: <paths>

Acceptance criteria:
- <verifiable bullet>
```

The "Existing code" section is mandatory. The V0 backlog rotted because every
`Status:` said `todo` even after code shipped; naming which V0 paths each task
touches makes that rot visible.
