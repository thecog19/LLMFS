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
  shadow data + free-list pop; phase 2 atomically flushes redirection flip
  AND integrity CRC update; phase 3 flushes the old-physical free-list push.
  Collapsing CRC and redirection into one flush avoids a window in the spec's
  sequence where the integrity table describes the shadow block but the
  redirection still points at the old — any read of the logical in that
  window would report a CRC mismatch against its own (old) data. The phase-2
  mmap writes land together, so a crash mid-phase-2 leaves neither durable.

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

- The superblock still lacks the `generation`, `shadow_block`, and
  `shadow_target` fields that §5/§13 specify. Recovery falls back to orphan
  scan + dirty-flag detection, which is functionally sufficient for V1 but
  cannot distinguish "crash before redirection flip" from "crash after flip"
  by intent — only by reachability. Adding the fields is tracked as a
  Task 05 amendment, not in Task 08's scope.
