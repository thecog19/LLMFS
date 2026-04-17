use llmdb::nbd::protocol::{
    CLISERV_MAGIC, NBDMAGIC, NBD_REPLY_MAGIC, NBD_REQUEST_MAGIC, NbdCommand, NbdProtoError,
    NbdRequest, OLDSTYLE_HANDSHAKE_BYTES, REPLY_HEADER_BYTES, REQUEST_HEADER_BYTES,
    encode_oldstyle_handshake, encode_reply_header, encode_request, parse_reply_header,
    parse_request,
};

#[test]
fn parse_read_request_extracts_every_field() {
    let req = NbdRequest {
        command: NbdCommand::Read,
        flags: 0,
        handle: 0xDEAD_BEEF_0BAD_F00D,
        offset: 0x1_0000,
        length: 4096,
    };
    let bytes = encode_request(&req);
    assert_eq!(bytes.len(), REQUEST_HEADER_BYTES);

    let (parsed, consumed) = parse_request(&bytes).expect("parse");
    assert_eq!(consumed, REQUEST_HEADER_BYTES);
    assert_eq!(parsed, req);
}

#[test]
fn parse_write_request_preserves_offset_and_length() {
    let req = NbdRequest {
        command: NbdCommand::Write,
        flags: 0x1,
        handle: 42,
        offset: 7 * 4096,
        length: 8192,
    };
    let bytes = encode_request(&req);
    let (parsed, _) = parse_request(&bytes).expect("parse");
    assert_eq!(parsed.command, NbdCommand::Write);
    assert_eq!(parsed.offset, 7 * 4096);
    assert_eq!(parsed.length, 8192);
    assert_eq!(parsed.handle, 42);
}

#[test]
fn parse_request_rejects_bad_magic() {
    let mut bytes = [0_u8; REQUEST_HEADER_BYTES];
    // Deliberately wrong magic in bytes[0..4]; everything else irrelevant.
    bytes[0..4].copy_from_slice(&0x1234_5678_u32.to_be_bytes());
    let result = parse_request(&bytes);
    match result {
        Err(NbdProtoError::BadMagic { expected, actual }) => {
            assert_eq!(expected, NBD_REQUEST_MAGIC);
            assert_eq!(actual, 0x1234_5678);
        }
        other => panic!("expected BadMagic, got {other:?}"),
    }
}

#[test]
fn parse_request_rejects_truncated_header() {
    let bytes = [0_u8; REQUEST_HEADER_BYTES - 1];
    let result = parse_request(&bytes);
    match result {
        Err(NbdProtoError::TruncatedHeader { have, need }) => {
            assert_eq!(have, REQUEST_HEADER_BYTES - 1);
            assert_eq!(need, REQUEST_HEADER_BYTES);
        }
        other => panic!("expected TruncatedHeader, got {other:?}"),
    }
}

#[test]
fn parse_request_rejects_unknown_command() {
    let mut bytes = encode_request(&NbdRequest {
        command: NbdCommand::Read,
        flags: 0,
        handle: 0,
        offset: 0,
        length: 0,
    });
    // Splice an invalid command type (99) into the command field.
    bytes[6..8].copy_from_slice(&99_u16.to_be_bytes());
    let result = parse_request(&bytes);
    assert!(matches!(result, Err(NbdProtoError::UnknownCommand(99))));
}

#[test]
fn reply_header_roundtrips_error_and_handle() {
    let bytes = encode_reply_header(0, 0xCAFEBABE);
    assert_eq!(bytes.len(), REPLY_HEADER_BYTES);
    let (error, handle) = parse_reply_header(&bytes).expect("parse");
    assert_eq!(error, 0);
    assert_eq!(handle, 0xCAFEBABE);

    // First 16 bytes contain magic + error + handle in network byte order.
    assert_eq!(
        u32::from_be_bytes(bytes[0..4].try_into().unwrap()),
        NBD_REPLY_MAGIC
    );
}

#[test]
fn reply_header_carries_nonzero_error_code() {
    let bytes = encode_reply_header(5, 17);
    let (error, handle) = parse_reply_header(&bytes).expect("parse");
    assert_eq!(error, 5);
    assert_eq!(handle, 17);
}

#[test]
fn parse_reply_header_rejects_short_buffer() {
    let bytes = [0_u8; REPLY_HEADER_BYTES - 1];
    assert!(matches!(
        parse_reply_header(&bytes),
        Err(NbdProtoError::TruncatedHeader { .. })
    ));
}

#[test]
fn oldstyle_handshake_is_exactly_152_bytes_with_advertised_size() {
    let size = 123_456_789_u64;
    let bytes = encode_oldstyle_handshake(size, 0);
    assert_eq!(bytes.len(), OLDSTYLE_HANDSHAKE_BYTES);

    // NBDMAGIC in bytes 0..8, CLISERV_MAGIC in 8..16, size in 16..24.
    assert_eq!(
        u64::from_be_bytes(bytes[0..8].try_into().unwrap()),
        NBDMAGIC
    );
    assert_eq!(
        u64::from_be_bytes(bytes[8..16].try_into().unwrap()),
        CLISERV_MAGIC
    );
    assert_eq!(
        u64::from_be_bytes(bytes[16..24].try_into().unwrap()),
        size
    );
    // Tail padding must be zero.
    assert!(bytes[28..].iter().all(|&b| b == 0));
}

#[test]
fn oldstyle_handshake_preserves_flags_field() {
    let bytes = encode_oldstyle_handshake(0, 0xBEEF);
    assert_eq!(
        u16::from_be_bytes(bytes[24..26].try_into().unwrap()),
        0xBEEF
    );
    // Reserved u16 after flags stays zero.
    assert_eq!(u16::from_be_bytes(bytes[26..28].try_into().unwrap()), 0);
}

#[test]
fn nbd_magic_constant_is_ascii_nbdmagic() {
    assert_eq!(&NBDMAGIC.to_be_bytes(), b"NBDMAGIC");
}
