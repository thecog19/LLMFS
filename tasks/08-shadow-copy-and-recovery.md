# Task 08: Shadow-Copy Atomicity And Recovery

Status: done
Depends on: 07-stego-device-block-io.md
Spec refs: DESIGN-NEW.MD sections "13. Failure Modes" (Crash During Write, Unclean Shutdown Detection), "5. Stego Device" (Block Write Implementation)

Objective:
Guarantee the two-flush shadow-copy-then-swap sequence is crash-safe per §13,
and implement the orphan-block scan that replaces V0's pending-metadata-op
recovery. The guarantee: a crash at any point during `write_block` leaves
either the old or new block as the canonical value — never a torn block.

Scope:

- Add `dirty` flag to the superblock (bit 1 of the flags byte, already reserved in Task 05).
- On `StegoDevice::new` (open): if dirty flag is set, log "unclean shutdown detected" and run `recover()`. Otherwise, set dirty and flush, then proceed.
- On `StegoDevice::drop` / explicit `close()`: clear dirty flag and flush.
- `fn recover(&mut self)`:
  1. Walk the redirection table and collect the set of physical blocks currently in use as data blocks.
  2. Walk the free list and collect the set of physical blocks currently free.
  3. Compute orphans = all_physical_blocks − in_use − free − metadata_blocks.
  4. Push orphans back onto the free list (they were the shadow blocks from an interrupted write).
  5. Run `verify_all()` to flag any data blocks that fail CRC32.
  6. Clear the dirty flag and flush.
- Crash-point test API on `StegoDevice`: `#[cfg(test)] fn write_block_with_crash_after(&mut self, logical: u32, data: &[u8], crash_point: CrashPoint)` where `CrashPoint` is `AfterShadowWrite`, `AfterCrcUpdate`, `AfterRedirectionFlush`. This lets tests simulate an interrupt without spawning subprocesses.
- Tests in `tests/shadow_copy.rs` (net-new file):
  - `crash_before_first_flush_leaves_old_block_intact`: write a known value to logical 10, flush. Simulate a crash just after allocating a shadow and writing partial data, before any flush. Reopen, read logical 10 — expect the original value.
  - `crash_between_flushes_leaves_old_block_intact_and_shadow_reclaimed`: similar, but crash after the CRC update and shadow flush, before the redirection flip. Reopen, read — original value. Verify the orphan shadow block is back on the free list.
  - `crash_after_both_flushes_sees_new_value`: crash after the redirection flip flush. Reopen, read — new value. Old physical block should be on the free list.
  - `unclean_shutdown_sets_dirty_flag_and_triggers_recovery`: drop the device without calling `close()`; reopen and verify the dirty flag was set, recovery ran, and dirty is clear again.

Existing code to reuse / rework / delete:
- Reuse: CRC32 helper, integrity-block update routine, free-list push from Task 07
- Rework: `StegoDevice::new` and `Drop` (dirty-flag handling), `StegoDevice::recover`
- Delete: nothing (V0 pending-metadata code already dies in Task 07)

Acceptance criteria:
- `cargo test --offline tests::shadow_copy` passes all four crash-scenario tests.
- `Superblock.flags & 0x02` (dirty) is set exactly while a device handle is alive and clear after explicit close.
- Orphan blocks are reliably returned to the free list after a crash simulation; `device.used_blocks()` after recovery equals `used_blocks()` before the crash.
- Recovery is idempotent: running `recover()` twice in a row on the same (already-recovered) device is a no-op.

Deviations from spec (intentional):

- The write sequence uses 2 flushes, not 3 as in §13. Phase 1 flushes the
  shadow data + free-list pop + shadow intent (superblock); phase 2
  atomically flushes redirection flip AND integrity CRC update; phase 3
  pushes the old physical to the free list and clears the intent in memory,
  relying on the next superblock persist to make the clear durable. Why
  collapse phase 2: integrity is indexed by *logical* block, not physical
  (see `update_block_crc_for_data` in `src/stego/device.rs`). §13's literal
  sequence — integrity update then redirection flip in separate flushes —
  opens a window where `integrity[L] = CRC(new)` while `redirection[L] = O`,
  so a reader follows redirection to the old physical and fails the
  logical's CRC check against its own (old) data. The phase-2 mmap writes
  land under one flush, closing the window. V3's dirty-cache model dissolves
  the constraint (all writes live inside an atomic cache transaction);
  collapsing phase 2 is a V1-specific adaptation that can be un-collapsed
  once V3 lands.

- `write_block` is direct for the first write after `alloc_block` (the block
  has no live data to preserve) and shadow-copy for overwrites. This matches
  §13's File Store sequence ("write all data blocks, flush") for newly
  allocated blocks and §5's shadow-copy contract for in-place updates.
  "Written" state is carried in the high bit of each redirection entry; a
  bare identity entry (`entries[L] == L`) means "never written".

- On the first shadow-copy of a logical L, old_physical == L itself. Pushing
  that slot back to the free list would let `alloc_block` hand L out as a
  new logical while redirection[L] still referenced live shadow data. The
  slot is held as "wasted" (not on free list, not a referenced data location)
  until `free_block(L)` reclaims both the slot and the shadow physical.

- Recovery uses orphan scan as its reclamation engine, with the superblock
  intent fields (`shadow_block`, `shadow_target`) as the diagnostic signal
  rather than a separate reclamation path. §13 says "determine the old
  block by scanning" — the intent lets recovery *name* which side of the
  flip the crash landed on (aborted vs. committed), but the concrete block
  to reclaim is still found by scanning. This matches §13's spec; it is
  not a deviation.
