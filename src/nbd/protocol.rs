use thiserror::Error;

/// Bootstrap handle — retained so `bootstrap_smoke` can assert the crate
/// wires the NBD module. The real protocol surface is below.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NbdProtocolBootstrap {
    pub block_size: usize,
}

impl Default for NbdProtocolBootstrap {
    fn default() -> Self {
        Self {
            block_size: crate::BLOCK_SIZE,
        }
    }
}

// -- Protocol constants (see https://github.com/NetworkBlockDevice/nbd/blob/master/doc/proto.md) --

/// Magic in every request header (big-endian on the wire).
pub const NBD_REQUEST_MAGIC: u32 = 0x2560_9513;

/// Magic in every simple reply header (big-endian on the wire).
pub const NBD_REPLY_MAGIC: u32 = 0x6744_6698;

/// "NBDMAGIC" as ASCII, big-endian u64 — first 8 bytes of the oldstyle
/// negotiation banner.
pub const NBDMAGIC: u64 = 0x4e42_444d_4147_4943;

/// "IHAVEOPT" — option-negotiation magic used by newstyle handshakes. V1
/// uses oldstyle only; kept here for the eventual newstyle upgrade path.
pub const IHAVEOPT: u64 = 0x4948_4156_454F_5054;

/// Second oldstyle banner magic, immediately after "NBDMAGIC".
pub const CLISERV_MAGIC: u64 = 0x0000_4202_8186_1253;

/// On-wire size of a request header (no payload).
pub const REQUEST_HEADER_BYTES: usize = 28;

/// On-wire size of a simple reply header (data, if any, follows separately).
pub const REPLY_HEADER_BYTES: usize = 16;

/// On-wire size of the oldstyle negotiation banner the server emits on
/// connection accept.
pub const OLDSTYLE_HANDSHAKE_BYTES: usize = 152;

// -- Command type --

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum NbdCommand {
    Read = 0,
    Write = 1,
    Disc = 2,
    Flush = 3,
    Trim = 4,
    Cache = 5,
    WriteZeroes = 6,
    BlockStatus = 7,
}

impl NbdCommand {
    pub fn from_u16(value: u16) -> Result<Self, NbdProtoError> {
        Ok(match value {
            0 => Self::Read,
            1 => Self::Write,
            2 => Self::Disc,
            3 => Self::Flush,
            4 => Self::Trim,
            5 => Self::Cache,
            6 => Self::WriteZeroes,
            7 => Self::BlockStatus,
            other => return Err(NbdProtoError::UnknownCommand(other)),
        })
    }

    pub fn as_u16(self) -> u16 {
        self as u16
    }
}

// -- Request --

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NbdRequest {
    pub command: NbdCommand,
    pub flags: u16,
    pub handle: u64,
    pub offset: u64,
    pub length: u32,
}

/// Parse a request header from the front of `bytes`. Returns the parsed
/// request plus the number of bytes consumed (always `REQUEST_HEADER_BYTES`
/// on success), so the caller can drive a cursor across a stream that may
/// hold multiple pipelined requests. Does not consume the write-payload
/// bytes that follow a `Write` request — the server is expected to read
/// `request.length` additional bytes from the socket.
pub fn parse_request(bytes: &[u8]) -> Result<(NbdRequest, usize), NbdProtoError> {
    if bytes.len() < REQUEST_HEADER_BYTES {
        return Err(NbdProtoError::TruncatedHeader {
            have: bytes.len(),
            need: REQUEST_HEADER_BYTES,
        });
    }

    let magic = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
    if magic != NBD_REQUEST_MAGIC {
        return Err(NbdProtoError::BadMagic {
            expected: NBD_REQUEST_MAGIC,
            actual: magic,
        });
    }

    let flags = u16::from_be_bytes(bytes[4..6].try_into().unwrap());
    let cmd_raw = u16::from_be_bytes(bytes[6..8].try_into().unwrap());
    let command = NbdCommand::from_u16(cmd_raw)?;
    let handle = u64::from_be_bytes(bytes[8..16].try_into().unwrap());
    let offset = u64::from_be_bytes(bytes[16..24].try_into().unwrap());
    let length = u32::from_be_bytes(bytes[24..28].try_into().unwrap());

    Ok((
        NbdRequest {
            command,
            flags,
            handle,
            offset,
            length,
        },
        REQUEST_HEADER_BYTES,
    ))
}

/// Encode a request header. Used by the eventual `nbd_smoke` E2E test in
/// Task 12 to drive the server with a handcrafted client.
pub fn encode_request(req: &NbdRequest) -> [u8; REQUEST_HEADER_BYTES] {
    let mut bytes = [0_u8; REQUEST_HEADER_BYTES];
    bytes[0..4].copy_from_slice(&NBD_REQUEST_MAGIC.to_be_bytes());
    bytes[4..6].copy_from_slice(&req.flags.to_be_bytes());
    bytes[6..8].copy_from_slice(&req.command.as_u16().to_be_bytes());
    bytes[8..16].copy_from_slice(&req.handle.to_be_bytes());
    bytes[16..24].copy_from_slice(&req.offset.to_be_bytes());
    bytes[24..28].copy_from_slice(&req.length.to_be_bytes());
    bytes
}

// -- Reply --

/// Represents a fully-formed reply. The data payload is only meaningful for
/// successful `Read` responses; for other commands it is empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NbdReply {
    pub error: u32,
    pub handle: u64,
    pub data: Vec<u8>,
}

/// Encode only the 16-byte reply header. The data payload (for reads) is
/// sent separately so the server can avoid an extra allocation / copy — it
/// writes the header, then streams the bytes from wherever they live.
pub fn encode_reply_header(error: u32, handle: u64) -> [u8; REPLY_HEADER_BYTES] {
    let mut bytes = [0_u8; REPLY_HEADER_BYTES];
    bytes[0..4].copy_from_slice(&NBD_REPLY_MAGIC.to_be_bytes());
    bytes[4..8].copy_from_slice(&error.to_be_bytes());
    bytes[8..16].copy_from_slice(&handle.to_be_bytes());
    bytes
}

/// Parse the first 16 bytes of a reply. Returns `(error, handle)`.
pub fn parse_reply_header(bytes: &[u8]) -> Result<(u32, u64), NbdProtoError> {
    if bytes.len() < REPLY_HEADER_BYTES {
        return Err(NbdProtoError::TruncatedHeader {
            have: bytes.len(),
            need: REPLY_HEADER_BYTES,
        });
    }
    let magic = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
    if magic != NBD_REPLY_MAGIC {
        return Err(NbdProtoError::BadMagic {
            expected: NBD_REPLY_MAGIC,
            actual: magic,
        });
    }
    let error = u32::from_be_bytes(bytes[4..8].try_into().unwrap());
    let handle = u64::from_be_bytes(bytes[8..16].try_into().unwrap());
    Ok((error, handle))
}

// -- Oldstyle negotiation banner --
//
// Layout (big-endian), 152 bytes total:
//   0x00   8 bytes  "NBDMAGIC"
//   0x08   8 bytes  CLISERV_MAGIC (0x00420281861253)
//   0x10   8 bytes  export size in bytes
//   0x18   2 bytes  flags (u16)
//   0x1A   2 bytes  reserved (zero)
//   0x1C 124 bytes  reserved zero padding

pub fn encode_oldstyle_handshake(
    export_size: u64,
    flags: u16,
) -> [u8; OLDSTYLE_HANDSHAKE_BYTES] {
    let mut bytes = [0_u8; OLDSTYLE_HANDSHAKE_BYTES];
    bytes[0..8].copy_from_slice(&NBDMAGIC.to_be_bytes());
    bytes[8..16].copy_from_slice(&CLISERV_MAGIC.to_be_bytes());
    bytes[16..24].copy_from_slice(&export_size.to_be_bytes());
    bytes[24..26].copy_from_slice(&flags.to_be_bytes());
    // 26..28 reserved zero; 28..152 zero padding (already zero).
    bytes
}

// -- Errors --

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum NbdProtoError {
    #[error("bad magic: expected {expected:#010x}, got {actual:#010x}")]
    BadMagic { expected: u32, actual: u32 },
    #[error("unknown NBD command type: {0}")]
    UnknownCommand(u16),
    #[error("truncated header: have {have} bytes, need {need}")]
    TruncatedHeader { have: usize, need: usize },
}
