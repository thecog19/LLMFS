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

/// On-wire size of the oldstyle negotiation banner. Retained for parity
/// tests; modern `nbd-client` (≥ 3.10) dropped oldstyle, so the server
/// uses newstyle exclusively — see `encode_newstyle_header`.
pub const OLDSTYLE_HANDSHAKE_BYTES: usize = 152;

/// On-wire size of the fixed newstyle negotiation header the server emits
/// on connection accept: 8 bytes NBDMAGIC + 8 bytes IHAVEOPT + 2 bytes of
/// server handshake flags = 18 bytes.
pub const NEWSTYLE_HEADER_BYTES: usize = 18;

// -- Newstyle handshake flags --

/// Server flag: we speak fixed newstyle (option haggling with reply magic,
/// per NBD_PROTO_VERSION_FIXED_NEWSTYLE).
pub const NBD_FLAG_FIXED_NEWSTYLE: u16 = 1 << 0;

/// Server flag: the 124-byte zero tail after `NBD_OPT_EXPORT_NAME` is
/// omitted when the client acknowledges this flag.
pub const NBD_FLAG_NO_ZEROES: u16 = 1 << 1;

/// Client flag: client speaks fixed newstyle.
pub const NBD_FLAG_C_FIXED_NEWSTYLE: u32 = 1 << 0;
/// Client flag: client supports the no-zeroes reply to NBD_OPT_EXPORT_NAME.
pub const NBD_FLAG_C_NO_ZEROES: u32 = 1 << 1;

// -- Option codes --

pub const NBD_OPT_EXPORT_NAME: u32 = 1;
pub const NBD_OPT_ABORT: u32 = 2;
pub const NBD_OPT_LIST: u32 = 3;
pub const NBD_OPT_STARTTLS: u32 = 5;
pub const NBD_OPT_INFO: u32 = 6;
pub const NBD_OPT_GO: u32 = 7;
pub const NBD_OPT_STRUCTURED_REPLY: u32 = 8;

/// Magic prefix on every option reply (NBD proto `NBD_REP_MAGIC`).
pub const OPTION_REPLY_MAGIC: u64 = 0x0003_E889_0455_65A9;

// -- Option reply types --

pub const NBD_REP_ACK: u32 = 1;
pub const NBD_REP_INFO: u32 = 3;
pub const NBD_REP_ERR_UNSUP: u32 = 0x8000_0001;
pub const NBD_REP_ERR_POLICY: u32 = 0x8000_0002;
pub const NBD_REP_ERR_INVALID: u32 = 0x8000_0003;

// -- Info record types (inside NBD_REP_INFO) --

pub const NBD_INFO_EXPORT: u16 = 0;
pub const NBD_INFO_NAME: u16 = 1;
pub const NBD_INFO_DESCRIPTION: u16 = 2;
pub const NBD_INFO_BLOCK_SIZE: u16 = 3;

// -- Transmission flags (advertised per export) --

pub const NBD_FLAG_HAS_FLAGS: u16 = 1 << 0;
pub const NBD_FLAG_READ_ONLY: u16 = 1 << 1;
pub const NBD_FLAG_SEND_FLUSH: u16 = 1 << 2;
pub const NBD_FLAG_SEND_FUA: u16 = 1 << 3;
pub const NBD_FLAG_ROTATIONAL: u16 = 1 << 4;
pub const NBD_FLAG_SEND_TRIM: u16 = 1 << 5;

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

pub fn encode_oldstyle_handshake(export_size: u64, flags: u16) -> [u8; OLDSTYLE_HANDSHAKE_BYTES] {
    let mut bytes = [0_u8; OLDSTYLE_HANDSHAKE_BYTES];
    bytes[0..8].copy_from_slice(&NBDMAGIC.to_be_bytes());
    bytes[8..16].copy_from_slice(&CLISERV_MAGIC.to_be_bytes());
    bytes[16..24].copy_from_slice(&export_size.to_be_bytes());
    bytes[24..26].copy_from_slice(&flags.to_be_bytes());
    // 26..28 reserved zero; 28..152 zero padding (already zero).
    bytes
}

// -- Newstyle negotiation --
//
// Layout (big-endian), 18 bytes:
//   0x00   8 bytes  "NBDMAGIC"
//   0x08   8 bytes  "IHAVEOPT"
//   0x10   2 bytes  handshake flags
//
// The client then sends 4 bytes of client flags, and option haggling
// begins. After a successful `NBD_OPT_EXPORT_NAME` or `NBD_OPT_GO`
// reply the connection transitions to the transmission phase (same
// request/reply framing as oldstyle).

pub fn encode_newstyle_header(handshake_flags: u16) -> [u8; NEWSTYLE_HEADER_BYTES] {
    let mut bytes = [0_u8; NEWSTYLE_HEADER_BYTES];
    bytes[0..8].copy_from_slice(&NBDMAGIC.to_be_bytes());
    bytes[8..16].copy_from_slice(&IHAVEOPT.to_be_bytes());
    bytes[16..18].copy_from_slice(&handshake_flags.to_be_bytes());
    bytes
}

/// Reply to `NBD_OPT_EXPORT_NAME`. Not wrapped in the option-reply
/// envelope — this is a raw frame in the NBD spec. `no_zeroes` controls
/// whether the 124-byte reserved tail is omitted (when the client
/// advertised `NBD_FLAG_C_NO_ZEROES` in its client flags).
pub fn encode_export_name_reply(size: u64, flags: u16, no_zeroes: bool) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(if no_zeroes { 10 } else { 134 });
    bytes.extend_from_slice(&size.to_be_bytes());
    bytes.extend_from_slice(&flags.to_be_bytes());
    if !no_zeroes {
        bytes.extend_from_slice(&[0_u8; 124]);
    }
    bytes
}

/// Generic option-reply header + payload: magic + echoed option code +
/// reply type + payload length + payload bytes.
pub fn encode_option_reply(option: u32, reply_type: u32, payload: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(20 + payload.len());
    bytes.extend_from_slice(&OPTION_REPLY_MAGIC.to_be_bytes());
    bytes.extend_from_slice(&option.to_be_bytes());
    bytes.extend_from_slice(&reply_type.to_be_bytes());
    bytes.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

/// Encode the `NBD_INFO_EXPORT` info record that `NBD_OPT_GO` / `NBD_OPT_INFO`
/// replies wrap in an `NBD_REP_INFO` reply: 2 bytes info type + 8 bytes size
/// + 2 bytes transmission flags = 12 bytes of payload.
pub fn encode_info_export(size: u64, flags: u16) -> [u8; 12] {
    let mut bytes = [0_u8; 12];
    bytes[0..2].copy_from_slice(&NBD_INFO_EXPORT.to_be_bytes());
    bytes[2..10].copy_from_slice(&size.to_be_bytes());
    bytes[10..12].copy_from_slice(&flags.to_be_bytes());
    bytes
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NbdOptionHeader {
    pub option: u32,
    pub data_len: u32,
}

/// Parse the 16-byte option header sent by the client. Does not consume
/// the option payload bytes; the caller reads `data_len` bytes next.
pub fn parse_option_header(bytes: &[u8]) -> Result<NbdOptionHeader, NbdProtoError> {
    if bytes.len() < 16 {
        return Err(NbdProtoError::TruncatedHeader {
            have: bytes.len(),
            need: 16,
        });
    }
    let magic = u64::from_be_bytes(bytes[0..8].try_into().unwrap());
    if magic != IHAVEOPT {
        return Err(NbdProtoError::BadOptionMagic {
            expected: IHAVEOPT,
            actual: magic,
        });
    }
    let option = u32::from_be_bytes(bytes[8..12].try_into().unwrap());
    let data_len = u32::from_be_bytes(bytes[12..16].try_into().unwrap());
    Ok(NbdOptionHeader { option, data_len })
}

// -- Errors --

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum NbdProtoError {
    #[error("bad magic: expected {expected:#010x}, got {actual:#010x}")]
    BadMagic { expected: u32, actual: u32 },
    #[error("bad option magic: expected {expected:#018x}, got {actual:#018x}")]
    BadOptionMagic { expected: u64, actual: u64 },
    #[error("unknown NBD command type: {0}")]
    UnknownCommand(u16),
    #[error("truncated header: have {have} bytes, need {need}")]
    TruncatedHeader { have: usize, need: usize },
}
