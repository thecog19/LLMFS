use std::io::{self, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use thiserror::Error;

use crate::nbd::protocol::{
    NbdCommand, NbdProtoError, NbdRequest, REQUEST_HEADER_BYTES, encode_oldstyle_handshake,
    encode_reply_header, parse_request,
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
}

impl NbdServer {
    pub fn new(device: StegoDevice) -> Self {
        let data_region_start = device.data_region_start();
        let total_blocks = device.total_blocks();
        let data_blocks = total_blocks.saturating_sub(data_region_start) as u64;
        let export_bytes = data_blocks * crate::BLOCK_SIZE as u64;
        Self {
            device: Arc::new(Mutex::new(device)),
            data_region_start,
            export_bytes,
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
        drop(device);

        let start_in_full = (offset - first * block_size) as usize;
        Ok(full[start_in_full..start_in_full + length as usize].to_vec())
    }

    pub fn handle_write(&self, offset: u64, data: &[u8]) -> Result<(), NbdError> {
        let length = data.len() as u32;
        self.check_range(offset, length)?;
        if data.is_empty() {
            return Ok(());
        }

        let block_size = crate::BLOCK_SIZE as u64;
        let first = offset / block_size;
        let last_excl = (offset + length as u64).div_ceil(block_size);

        let mut device = self.device.lock().map_err(|_| NbdError::LockPoisoned)?;
        let mut cursor = 0_usize;
        for nbd_block in first..last_excl {
            let logical = self.data_region_start + nbd_block as u32;
            let block_byte_start = nbd_block * block_size;
            let in_start = offset.saturating_sub(block_byte_start) as usize;
            let in_end = ((offset + length as u64).saturating_sub(block_byte_start))
                .min(block_size) as usize;
            let copy_len = in_end - in_start;

            if in_start == 0 && in_end == block_size as usize {
                // Aligned full-block write — no read needed.
                device.write_block(logical, &data[cursor..cursor + copy_len])?;
            } else {
                // Partial block — read-modify-write.
                let mut block = device.read_block(logical)?;
                block[in_start..in_end].copy_from_slice(&data[cursor..cursor + copy_len]);
                device.write_block(logical, &block)?;
            }
            cursor += copy_len;
        }
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
        let handshake = encode_oldstyle_handshake(self.export_bytes, 0);
        conn.write_all(&handshake)?;
        conn.flush()?;

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
                NbdCommand::Read => self.reply_read(&mut conn, &req)?,
                NbdCommand::Write => self.reply_write(&mut conn, &req)?,
                NbdCommand::Flush => self.reply_flush(&mut conn, &req)?,
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
                conn.write_all(&encode_reply_header(22, req.handle))?;
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
                conn.write_all(&encode_reply_header(22, req.handle))?;
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
