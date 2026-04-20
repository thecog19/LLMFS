# Task 11: NBD Protocol

Status: superseded — NBD shipped in v1.0.0 then was deleted post-v1
in favor of FUSE (commits cd52c86 + 5c275ef). See DESIGN-NEW §7 for
the current mount architecture. This file is kept as a historical
record; nothing in the current tree implements or depends on it.

Depends on: 07-stego-device-block-io.md
Spec refs: DESIGN-NEW.MD section "7. NBD Server" (Protocol) — historical

Objective:
Implement the NBD wire protocol types so the server in Task 12 can parse kernel
requests and construct replies. Pure data-layer task — no sockets, no kernel,
no `StegoDevice` wiring. Keeping this separate makes the server implementation
testable against in-memory byte buffers.

Scope:

- Create `src/nbd/protocol.rs` with:
  - NBD magic constants: `NBD_REQUEST_MAGIC = 0x25609513`, `NBD_REPLY_MAGIC = 0x67446698`, plus the negotiation handshake magics (`NBDMAGIC = 0x4e42444d41474943`, `IHAVEOPT = 0x49484156454F5054`).
  - `enum NbdCommand { Read = 0, Write = 1, Disc = 2, Flush = 3, Trim = 4, Cache = 5, WriteZeroes = 6, BlockStatus = 7 }` (V1 only handles Read, Write, Disc, Flush).
  - `struct NbdRequest { command: NbdCommand, flags: u16, handle: u64, offset: u64, length: u32 }`.
  - `struct NbdReply { error: u32, handle: u64, data: Vec<u8> }`.
  - `fn parse_request(bytes: &[u8]) -> Result<(NbdRequest, usize), NbdProtoError>` — returns request + number of bytes consumed so the server can handle partial reads.
  - `fn encode_reply_header(error: u32, handle: u64) -> [u8; 16]` — encoded size only (the data payload for reads is sent separately to avoid copying).
  - Handshake: `fn encode_oldstyle_handshake(export_size: u64, flags: u16) -> [u8; 152]` — V1 uses oldstyle (simpler, no negotiation options needed for a single export).
- `NbdProtoError` variants: BadMagic, UnknownCommand, TruncatedHeader.
- Tests in `tests/nbd_protocol.rs` (net-new):
  - Parse a well-formed NBD_CMD_READ request; verify fields.
  - Parse a well-formed NBD_CMD_WRITE request; verify offset and length.
  - Reject bad magic with `NbdProtoError::BadMagic`.
  - Reject truncated headers (< 28 bytes) with `NbdProtoError::TruncatedHeader`.
  - Roundtrip an encoded reply header and re-parse its first 16 bytes.
  - Oldstyle handshake is exactly 152 bytes and contains the advertised export size.

Existing code to reuse / rework / delete:
- Reuse: nothing (net-new module)
- Rework: nothing
- Delete: nothing

Acceptance criteria:
- `cargo test --offline tests::nbd_protocol` passes.
- All protocol constants match the NBD spec on kernel.org (reference: https://github.com/NetworkBlockDevice/nbd/blob/master/doc/proto.md — implementor must verify against the spec, not copy from memory).
- No I/O in this module; everything is pure byte-slice parsing and encoding.
