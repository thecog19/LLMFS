use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crc32fast::Hasher;
use thiserror::Error;

use crate::nbd::protocol::{
    NBD_FLAG_C_NO_ZEROES, NBD_FLAG_FIXED_NEWSTYLE, NBD_FLAG_HAS_FLAGS, NBD_FLAG_NO_ZEROES,
    NBD_FLAG_SEND_FLUSH, NBD_OPT_ABORT, NBD_OPT_EXPORT_NAME, NBD_OPT_GO, NBD_OPT_INFO, NBD_REP_ACK,
    NBD_REP_ERR_UNSUP, NBD_REP_INFO, NbdCommand, NbdProtoError, NbdRequest, REQUEST_HEADER_BYTES,
    encode_export_name_reply, encode_info_export, encode_newstyle_header, encode_option_reply,
    encode_reply_header, parse_option_header, parse_request,
};
use crate::stego::device::{DeviceError, StegoDevice};

/// Bootstrap stub — preserved for the bootstrap_smoke path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NbdServerBootstrap {
    pub handles_partial_requests: bool,
}

impl Default for NbdServerBootstrap {
    fn default() -> Self {
        Self {
            handles_partial_requests: true,
        }
    }
}

/// Server-side errors surfaced by the NBD bridge.
#[derive(Debug, Error)]
pub enum NbdError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("device error: {0}")]
    Device(#[from] DeviceError),
    #[error("protocol error: {0}")]
    Protocol(#[from] NbdProtoError),
    #[error("request out of range: offset {offset}, length {length}, export {export_bytes}")]
    OutOfRange {
        offset: u64,
        length: u32,
        export_bytes: u64,
    },
    #[error("device lock poisoned")]
    LockPoisoned,
}

/// NBD server backed by a single `StegoDevice`. Exposes the raw data region
/// (blocks `[data_region_start .. total_blocks)`) as a flat byte-addressable
/// export. Metadata blocks are not reachable through NBD — the block device
/// view deliberately hides the stego-internal layout so ext4 can own its own
/// allocation without stepping on redirection / file-table pages.
///
/// Concurrency model: one NBD client connection at a time (matches the V1
/// kernel-side assumption). The device sits behind a `Mutex`; requests
/// serialize naturally.
pub struct NbdServer {
    device: Arc<Mutex<StegoDevice>>,
    data_region_start: u32,
    export_bytes: u64,
    trace: Option<Mutex<File>>,
}

impl NbdServer {
    pub fn new(device: StegoDevice) -> Self {
        let data_region_start = device.data_region_start();
        let total_blocks = device.total_blocks();
        let data_blocks = total_blocks.saturating_sub(data_region_start) as u64;
        let export_bytes = data_blocks * crate::BLOCK_SIZE as u64;

        // Optional per-request trace log: every handled command appends a
        // tab-separated line. Useful for diagnosing what a kernel client
        // (mkfs, mount, the ext4 driver) is actually pushing across the
        // wire and in what order.
        let trace = std::env::var("LLMDB_NBD_TRACE").ok().and_then(|path| {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .ok()
                .map(Mutex::new)
        });

        Self {
            device: Arc::new(Mutex::new(device)),
            data_region_start,
            export_bytes,
            trace,
        }
    }

    fn trace(&self, line: &str) {
        if let Some(file) = &self.trace
            && let Ok(mut guard) = file.lock()
        {
            let _ = writeln!(guard, "{line}");
        }
    }

    pub fn export_bytes(&self) -> u64 {
        self.export_bytes
    }

    pub fn data_region_start(&self) -> u32 {
        self.data_region_start
    }

    /// Obtain a shared handle to the device. Used by the mount CLI to call
    /// `close()` after the server loop exits.
    pub fn device(&self) -> Arc<Mutex<StegoDevice>> {
        self.device.clone()
    }

    // -- Per-command dispatch --

    pub fn handle_read(&self, offset: u64, length: u32) -> Result<Vec<u8>, NbdError> {
        self.check_range(offset, length)?;
        if length == 0 {
            return Ok(Vec::new());
        }

        let block_size = crate::BLOCK_SIZE as u64;
        let first = offset / block_size;
        let last_excl = (offset + length as u64).div_ceil(block_size);

        let mut full = Vec::with_capacity(((last_excl - first) * block_size) as usize);
        let device = self.device.lock().map_err(|_| NbdError::LockPoisoned)?;
        for nbd_block in first..last_excl {
            let logical = self.data_region_start + nbd_block as u32;
            full.extend_from_slice(&device.read_block(logical)?);
        }
        let free_after = device.free_blocks().unwrap_or(u32::MAX);
        drop(device);

        let start_in_full = (offset - first * block_size) as usize;
        let result = full[start_in_full..start_in_full + length as usize].to_vec();
        let free_str = match self
            .device
            .lock()
            .map_err(|_| NbdError::LockPoisoned)?
            .free_blocks()
        {
            Ok(n) => n.to_string(),
            Err(e) => format!("ERR({e})"),
        };
        let _ = free_after; // kept for future reuse if the diff matters
        self.trace(&format!(
            "READ\toffset={}\tlen={}\tcrc32={:08x}\tfree_after={}",
            offset,
            length,
            crc32(&result),
            free_str
        ));
        Ok(result)
    }

    pub fn handle_write(&self, offset: u64, data: &[u8]) -> Result<(), NbdError> {
        let length = data.len() as u32;
        self.check_range(offset, length)?;
        if data.is_empty() {
            return Ok(());
        }

        let payload_crc = crc32(data);
        let block_size = crate::BLOCK_SIZE as u64;
        let first = offset / block_size;
        let last_excl = (offset + length as u64).div_ceil(block_size);

        let mut device = self.device.lock().map_err(|_| NbdError::LockPoisoned)?;
        let mut cursor = 0_usize;
        let mut overwrites = 0_u32;
        for nbd_block in first..last_excl {
            let logical = self.data_region_start + nbd_block as u32;
            let block_byte_start = nbd_block * block_size;
            let in_start = offset.saturating_sub(block_byte_start) as usize;
            let in_end = ((offset + length as u64).saturating_sub(block_byte_start)).min(block_size)
                as usize;
            let copy_len = in_end - in_start;

            // Track whether this logical block has been touched before so
            // the trace makes "first write vs. overwrite" obvious.
            if device.is_logical_written(logical) {
                overwrites += 1;
            }

            let written_payload: Vec<u8> = if in_start == 0 && in_end == block_size as usize {
                let payload = data[cursor..cursor + copy_len].to_vec();
                device.write_block(logical, &payload)?;
                payload
            } else {
                let mut block = device.read_block(logical)?;
                block[in_start..in_end].copy_from_slice(&data[cursor..cursor + copy_len]);
                device.write_block(logical, &block)?;
                block
            };

            // Self-check: read the block back immediately and compare to
            // what we just wrote. If they differ, log it loudly — this
            // catches packer-roundtrip bugs and free-list/redirection
            // corruption before the kernel notices.
            let readback = device.read_block(logical)?;
            if readback != written_payload {
                let written_crc = crc32(&written_payload);
                let read_crc = crc32(&readback);
                self.trace(&format!(
                    "ROUNDTRIP_FAIL\tlogical={}\twrote_crc32={:08x}\tread_crc32={:08x}",
                    logical, written_crc, read_crc
                ));
            }

            cursor += copy_len;
        }
        let free_head = device.superblock().fields.free_list_head;
        let free_after = device.free_blocks();
        let free_str = match &free_after {
            Ok(n) => n.to_string(),
            Err(e) => format!("ERR({e})"),
        };
        drop(device);

        self.trace(&format!(
            "WRITE\toffset={}\tlen={}\tcrc32={:08x}\toverwrites={}\tfree_head={}\tfree_after={}",
            offset, length, payload_crc, overwrites, free_head, free_str
        ));
        Ok(())
    }

    pub fn handle_flush(&self) -> Result<(), NbdError> {
        let mut device = self.device.lock().map_err(|_| NbdError::LockPoisoned)?;
        device.flush()?;
        Ok(())
    }

    fn check_range(&self, offset: u64, length: u32) -> Result<(), NbdError> {
        let end = offset.saturating_add(length as u64);
        if end > self.export_bytes {
            return Err(NbdError::OutOfRange {
                offset,
                length,
                export_bytes: self.export_bytes,
            });
        }
        Ok(())
    }

    // -- Socket-level serve path --

    /// Bind a Unix socket at `socket_path`, accept one connection, run the
    /// oldstyle-handshake + request loop until the client disconnects, then
    /// remove the socket file. Intended to run on its own thread from the
    /// `mount` CLI.
    pub fn serve_on_unix_socket(&self, socket_path: &Path) -> Result<(), NbdError> {
        if socket_path.exists() {
            std::fs::remove_file(socket_path)?;
        }
        let listener = UnixListener::bind(socket_path)?;
        let (conn, _addr) = listener.accept()?;
        let result = self.serve_connection(conn);
        // Best-effort socket cleanup — we have already accepted; leaving
        // the socket file around would just be a stale entry.
        std::fs::remove_file(socket_path).ok();
        result
    }

    fn serve_connection(&self, mut conn: UnixStream) -> Result<(), NbdError> {
        let no_zeroes = match self.newstyle_handshake(&mut conn)? {
            Some(flag) => flag,
            None => return Ok(()), // client dropped after banner
        };
        match self.option_phase(&mut conn, no_zeroes)? {
            OptionOutcome::Transmission => self.transmission_phase(&mut conn),
            OptionOutcome::Abort => Ok(()),
            OptionOutcome::ClientGone => Ok(()),
        }
    }

    /// Send the fixed-newstyle banner and read the 4-byte client flags.
    /// Returns `Some(no_zeroes)` with the negotiated flag, or `None` if
    /// the client closed the connection between the banner and its flags
    /// reply.
    fn newstyle_handshake(&self, conn: &mut UnixStream) -> Result<Option<bool>, NbdError> {
        let handshake_flags = NBD_FLAG_FIXED_NEWSTYLE | NBD_FLAG_NO_ZEROES;
        conn.write_all(&encode_newstyle_header(handshake_flags))?;
        conn.flush()?;

        let mut client_flags_bytes = [0_u8; 4];
        match conn.read_exact(&mut client_flags_bytes) {
            Ok(()) => {}
            Err(e) if matches!(e.kind(), io::ErrorKind::UnexpectedEof) => {
                return Ok(None);
            }
            Err(e) => return Err(e.into()),
        }
        let client_flags = u32::from_be_bytes(client_flags_bytes);
        Ok(Some(client_flags & NBD_FLAG_C_NO_ZEROES != 0))
    }

    /// Option-haggling loop. Returns `Transmission` once the client has
    /// selected our (only) export via `EXPORT_NAME` or `GO`, or `Abort`
    /// on `NBD_OPT_ABORT`.
    fn option_phase(
        &self,
        conn: &mut UnixStream,
        no_zeroes: bool,
    ) -> Result<OptionOutcome, NbdError> {
        let transmission_flags = NBD_FLAG_HAS_FLAGS | NBD_FLAG_SEND_FLUSH;

        loop {
            let mut hdr = [0_u8; 16];
            match conn.read_exact(&mut hdr) {
                Ok(()) => {}
                Err(e) if matches!(e.kind(), io::ErrorKind::UnexpectedEof) => {
                    return Ok(OptionOutcome::ClientGone);
                }
                Err(e) => return Err(e.into()),
            }
            let parsed = parse_option_header(&hdr)?;

            let mut data = vec![0_u8; parsed.data_len as usize];
            if parsed.data_len > 0 {
                conn.read_exact(&mut data)?;
            }

            match parsed.option {
                NBD_OPT_EXPORT_NAME => {
                    // Raw (non-option-reply) frame — size + flags, optional
                    // 124-byte zero tail when the client did not ack
                    // NO_ZEROES.
                    let reply =
                        encode_export_name_reply(self.export_bytes, transmission_flags, no_zeroes);
                    conn.write_all(&reply)?;
                    conn.flush()?;
                    return Ok(OptionOutcome::Transmission);
                }
                NBD_OPT_GO | NBD_OPT_INFO => {
                    // Reply with NBD_INFO_EXPORT (size + flags), then ACK.
                    // The client's `data` field carries a requested-info
                    // list we ignore — INFO_EXPORT is mandatory to send.
                    let info = encode_info_export(self.export_bytes, transmission_flags);
                    let info_reply = encode_option_reply(parsed.option, NBD_REP_INFO, &info);
                    conn.write_all(&info_reply)?;

                    let ack = encode_option_reply(parsed.option, NBD_REP_ACK, &[]);
                    conn.write_all(&ack)?;
                    conn.flush()?;

                    if parsed.option == NBD_OPT_GO {
                        return Ok(OptionOutcome::Transmission);
                    }
                    // NBD_OPT_INFO: stay in the haggling loop.
                }
                NBD_OPT_ABORT => {
                    let ack = encode_option_reply(parsed.option, NBD_REP_ACK, &[]);
                    conn.write_all(&ack)?;
                    conn.flush()?;
                    return Ok(OptionOutcome::Abort);
                }
                other => {
                    let err = encode_option_reply(other, NBD_REP_ERR_UNSUP, &[]);
                    conn.write_all(&err)?;
                    conn.flush()?;
                }
            }
        }
    }

    fn transmission_phase(&self, conn: &mut UnixStream) -> Result<(), NbdError> {
        let mut header = [0_u8; REQUEST_HEADER_BYTES];
        loop {
            match conn.read_exact(&mut header) {
                Ok(()) => {}
                Err(e) if matches!(e.kind(), io::ErrorKind::UnexpectedEof) => {
                    // Client disconnected — treat as clean exit.
                    return Ok(());
                }
                Err(e) => return Err(e.into()),
            }
            let (req, _) = parse_request(&header)?;
            match req.command {
                NbdCommand::Disc => return Ok(()),
                NbdCommand::Read => self.reply_read(conn, &req)?,
                NbdCommand::Write => self.reply_write(conn, &req)?,
                NbdCommand::Flush => self.reply_flush(conn, &req)?,
                // Unsupported commands get a generic EINVAL (error=22).
                NbdCommand::Trim
                | NbdCommand::Cache
                | NbdCommand::WriteZeroes
                | NbdCommand::BlockStatus => {
                    conn.write_all(&encode_reply_header(22, req.handle))?;
                }
            }
        }
    }

    fn reply_read(&self, conn: &mut UnixStream, req: &NbdRequest) -> Result<(), NbdError> {
        match self.handle_read(req.offset, req.length) {
            Ok(data) => {
                conn.write_all(&encode_reply_header(0, req.handle))?;
                conn.write_all(&data)?;
            }
            Err(NbdError::OutOfRange { .. }) => {
                conn.write_all(&encode_reply_header(EINVAL, req.handle))?;
            }
            Err(NbdError::Device(e)) => {
                eprintln!("[llmdb:nbd] read error on offset {}: {e}", req.offset);
                conn.write_all(&encode_reply_header(EIO, req.handle))?;
            }
            Err(e) => return Err(e),
        }
        Ok(())
    }

    fn reply_write(&self, conn: &mut UnixStream, req: &NbdRequest) -> Result<(), NbdError> {
        let mut payload = vec![0_u8; req.length as usize];
        conn.read_exact(&mut payload)?;
        match self.handle_write(req.offset, &payload) {
            Ok(()) => conn.write_all(&encode_reply_header(0, req.handle))?,
            Err(NbdError::OutOfRange { .. }) => {
                conn.write_all(&encode_reply_header(EINVAL, req.handle))?;
            }
            Err(NbdError::Device(e)) => {
                eprintln!("[llmdb:nbd] write error on offset {}: {e}", req.offset);
                conn.write_all(&encode_reply_header(EIO, req.handle))?;
            }
            Err(e) => return Err(e),
        }
        Ok(())
    }

    fn reply_flush(&self, conn: &mut UnixStream, req: &NbdRequest) -> Result<(), NbdError> {
        match self.handle_flush() {
            Ok(()) => conn.write_all(&encode_reply_header(0, req.handle))?,
            Err(e) => return Err(e),
        }
        Ok(())
    }
}

/// Convenience alias so tests and callers can refer to the shared handle
/// without chasing imports.
pub type SharedDevice = Arc<Mutex<StegoDevice>>;

pub fn default_socket_path(pid: u32) -> PathBuf {
    PathBuf::from(format!("/tmp/llmdb-{pid}.sock"))
}

enum OptionOutcome {
    Transmission,
    Abort,
    ClientGone,
}

const EINVAL: u32 = 22;
const EIO: u32 = 5;

fn crc32(data: &[u8]) -> u32 {
    let mut h = Hasher::new();
    h.update(data);
    h.finalize()
}
