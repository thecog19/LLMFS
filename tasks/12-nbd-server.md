# Task 12: NBD Server And Mount

Status: superseded — NBD shipped in v1.0.0 then was deleted post-v1
in favor of FUSE (commits cd52c86 + 5c275ef). See DESIGN-NEW §7 for
the current mount architecture. This file is kept as a historical
record; nothing in the current tree implements or depends on it.

Depends on: 11-nbd-protocol.md, 10-cli-file-commands.md
Spec refs: DESIGN-NEW.MD section "7. NBD Server" (Implementation, Alignment, `llmdb mount`) — historical

Objective:
Serve the stego device as an NBD export on a Unix socket, attach it to
`/dev/nbdN` via `nbd-client`, and let the kernel mount an ext4 filesystem on
top. This is the "it actually looks like a disk" milestone.

Scope:

- Create `src/nbd/server.rs`:
  - `struct NbdServer { device: Arc<Mutex<StegoDevice>>, socket_path: PathBuf }`.
  - `fn bind(device: StegoDevice, socket_path: impl Into<PathBuf>) -> Result<Self, NbdError>` — create Unix socket, write the oldstyle handshake.
  - `fn run(&mut self) -> Result<(), NbdError>` — accept-loop: on each client, loop reading requests and writing replies. One connection at a time in V1 (NBD clients are single-connection).
  - `fn handle_request(&mut self, req: NbdRequest) -> NbdReply` — dispatch on command:
    - `Read`: compute the enclosing block range `[floor(offset / 4096), ceil((offset + length) / 4096))`, read each logical block, slice out the requested bytes.
    - `Write`: read-modify-write for unaligned requests. For each enclosing block: read, splice in the new bytes, write.
    - `Flush`: `device.flush()`.
    - `Disc`: clean disconnect, clear any per-connection state.
  - `fn stop(&mut self) -> Result<(), NbdError>` — shutdown the socket, close the device cleanly (clears dirty flag per Task 08).
- CLI command wiring in `src/main.rs`:
  - `mount <model.gguf> <mount_point> [--format]`:
    1. Open or format-then-open the stego device.
    2. `bind` the NBD server on a per-PID socket path (`/run/user/$UID/llmdb-<pid>.sock` or `/tmp/llmdb-<pid>.sock`).
    3. Spawn a thread running `server.run()`.
    4. Exec `nbd-client -unix /tmp/llmdb-<pid>.sock /dev/nbdN` (pick first free `/dev/nbdN`).
    5. If `--format`: run `mkfs.ext4 /dev/nbdN` (with confirmation prompt).
    6. `mount /dev/nbdN <mount_point>`.
    7. Write a pidfile under `<mount_point>/.llmdb.pid` or a sidecar state file so `unmount` can find the session.
    8. Block on SIGINT or `llmdb unmount <mount_point>`.
  - `unmount <mount_point>`:
    1. `umount <mount_point>`.
    2. `nbd-client -d /dev/nbdN`.
    3. Send shutdown signal to the server thread; wait for clean close (dirty flag clear).
- V1 accepts the root / `CAP_SYS_ADMIN` requirement. §14 names FUSE as a possible V1 fallback; defer to a follow-up task unless the implementer encounters unprivileged-use friction during implementation.
- Tests:
  - `tests/nbd_smoke.rs` (E2E, `#[ignore]` unless `LLMDB_E2E_NBD=1`): bind a server, connect a Rust client (raw socket + `src/nbd/protocol.rs`), issue a read at offset 0 length 4096, verify the returned bytes match `device.read_block(0)`.
  - `tests/nbd_alignment.rs`: in-process test — feed a `NbdRequest { offset: 100, length: 3000 }` to `handle_request` with a freshly-formatted device; verify the reply's data slice equals the expected bytes from the first two logical blocks.

Existing code to reuse / rework / delete:
- Reuse: `src/nbd/protocol.rs` from Task 11, `StegoDevice::read_block` / `write_block`
- Rework: `src/main.rs` `mount` / `unmount` subcommand bodies
- Delete: nothing

Acceptance criteria:
- `cargo test --offline tests::nbd_alignment` passes.
- With `LLMDB_E2E_NBD=1` and root on a Linux host: `cargo run -- mount <gguf> /mnt/llmdb --format`, then `echo hello > /mnt/llmdb/test.txt`, then `cargo run -- unmount /mnt/llmdb`, then re-mount, verify `cat /mnt/llmdb/test.txt` prints `hello`.
- A Ctrl-C during `mount` cleanly unmounts, disconnects NBD, and clears the dirty flag (so the next open does not trigger recovery).

Deviations from spec (intentional):

- Handshake is newstyle, not oldstyle — modern `nbd-client` (≥ 3.10)
  dropped oldstyle support entirely. The server emits the fixed
  newstyle banner (`NBDMAGIC` + `IHAVEOPT` + flags) and handles
  `NBD_OPT_EXPORT_NAME` / `NBD_OPT_GO` / `NBD_OPT_INFO` / `NBD_OPT_ABORT`;
  anything else replies with `NBD_REP_ERR_UNSUP`.
- Free-device discovery scans `/sys/block/nbdN/pid` (0..15). That file
  exists when an nbd-client has bound the device; absent or empty → free.
- mount/unmount communicate via a sidecar state file in
  `$XDG_RUNTIME_DIR/llmdb-mounts/<hex-encoded-mount-path>.state`
  when `XDG_RUNTIME_DIR` is available, otherwise
  `/tmp/llmdb-mounts-<uid>/<hex-encoded-mount-path>.state`
  (plain key=value).
  Contains the running mount PID, the chosen `/dev/nbdN`, the socket
  path, and the mount point. `unmount` scans/reads it to find the
  right nbd device; the in-process Ctrl-C handler runs the same
  cleanup via `AtomicBool`.
- `unmount` runs `umount` + `nbd-client -d` itself; the running mount
  process's server thread then returns naturally (client disconnected)
  and the mount shell exits. A second `umount` attempt may log
  "not mounted" — harmless.
- Wire-format diagnostics: `LLMDB_NBD_TRACE=<path>` env var emits a
  tab-separated per-request log (offset, length, CRC32 of payload, free
  list head, free count). Turned off by default; cheap to leave gated.
