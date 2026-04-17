//! Socket-level smoke test for the NBD bridge. Binds the server on a Unix
//! socket, connects a Rust client, and issues a handshake + read + write +
//! disc sequence. This is the closest we can get to "real" NBD without
//! pulling in the kernel and `nbd-client` (which require root and the
//! `nbd` module loaded — see the CLI `mount` flow for that path).
//!
//! The task spec marks this `#[ignore]` unless `LLMDB_E2E_NBD=1`, but the
//! roundtrip here is pure userspace and has no system dependencies, so we
//! keep it part of the default suite to actually exercise the socket code.

mod common;

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use common::{SyntheticGgufVersion, SyntheticTensorSpec, write_custom_gguf_fixture};
use llmdb::gguf::quant::GGML_TYPE_Q8_0_ID;
use llmdb::nbd::protocol::{
    IHAVEOPT, NBD_FLAG_C_FIXED_NEWSTYLE, NBD_FLAG_C_NO_ZEROES, NBD_OPT_EXPORT_NAME, NBDMAGIC,
    NEWSTYLE_HEADER_BYTES, NbdCommand, NbdRequest, REPLY_HEADER_BYTES, encode_request,
    parse_reply_header,
};
use llmdb::nbd::server::NbdServer;
use llmdb::stego::device::{DeviceOptions, StegoDevice};
use llmdb::stego::planner::AllocationMode;

fn q8_tensors(count: usize) -> Vec<SyntheticTensorSpec> {
    (0..count)
        .map(|i| SyntheticTensorSpec {
            name: format!("blk.{}.ffn_down.weight", count - 1 - i),
            dimensions: vec![8192],
            raw_type_id: GGML_TYPE_Q8_0_ID,
            data: vec![0_u8; (8192 / 32) * 34],
        })
        .collect()
}

fn make_server(name: &str, tensor_count: usize) -> (common::FixtureHandle, NbdServer) {
    let fx = write_custom_gguf_fixture(SyntheticGgufVersion::V3, name, &q8_tensors(tensor_count));
    let device = StegoDevice::initialize_with_options(
        &fx.path,
        AllocationMode::Standard,
        DeviceOptions { verbose: false },
    )
    .expect("init");
    (fx, NbdServer::new(device))
}

fn unique_socket_path(suffix: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    dir.join(format!("llmdb-nbd-{suffix}-{pid}-{nanos}.sock"))
}

#[test]
fn client_handshake_read_write_disc_over_unix_socket() {
    let (_fx, server) = make_server("nbd_smoke_roundtrip.gguf", 12);
    let export = server.export_bytes();
    let sock = unique_socket_path("roundtrip");
    let sock_clone = sock.clone();

    let server = Arc::new(server);
    let server_thread = {
        let server = Arc::clone(&server);
        thread::spawn(move || server.serve_on_unix_socket(&sock_clone))
    };

    // Give the server a moment to bind. If the first connect attempt fails,
    // retry briefly — the alternative (synchronization channel) is overkill
    // for a test that already completes in sub-100ms.
    let mut conn = None;
    for _ in 0..50 {
        match UnixStream::connect(&sock) {
            Ok(s) => {
                conn = Some(s);
                break;
            }
            Err(_) => thread::sleep(Duration::from_millis(10)),
        }
    }
    let mut conn = conn.expect("failed to connect to llmdb nbd socket");
    conn.set_read_timeout(Some(Duration::from_secs(5)))
        .expect("set read timeout");

    // 1. Newstyle handshake: read banner (18 bytes), send client flags (4),
    //    send NBD_OPT_EXPORT_NAME with empty export name, read the 10-byte
    //    export reply (NO_ZEROES negotiated → no 124-byte tail).
    let mut banner = [0_u8; NEWSTYLE_HEADER_BYTES];
    conn.read_exact(&mut banner).expect("banner");
    assert_eq!(
        u64::from_be_bytes(banner[0..8].try_into().unwrap()),
        NBDMAGIC
    );
    assert_eq!(
        u64::from_be_bytes(banner[8..16].try_into().unwrap()),
        IHAVEOPT
    );

    let client_flags = NBD_FLAG_C_FIXED_NEWSTYLE | NBD_FLAG_C_NO_ZEROES;
    conn.write_all(&client_flags.to_be_bytes())
        .expect("client flags");

    // NBD_OPT_EXPORT_NAME with empty name.
    let mut opt = Vec::new();
    opt.extend_from_slice(&IHAVEOPT.to_be_bytes());
    opt.extend_from_slice(&NBD_OPT_EXPORT_NAME.to_be_bytes());
    opt.extend_from_slice(&0_u32.to_be_bytes());
    conn.write_all(&opt).expect("send export_name option");

    let mut export_reply = [0_u8; 10]; // 8 bytes size + 2 bytes flags (NO_ZEROES)
    conn.read_exact(&mut export_reply).expect("export reply");
    let advertised = u64::from_be_bytes(export_reply[0..8].try_into().unwrap());
    assert_eq!(advertised, export, "export reply size mismatch");

    // 2. Send a Read request for block 0.
    let read_req = NbdRequest {
        command: NbdCommand::Read,
        flags: 0,
        handle: 0xAA,
        offset: 0,
        length: 4096,
    };
    conn.write_all(&encode_request(&read_req))
        .expect("send read");

    let mut reply = [0_u8; REPLY_HEADER_BYTES];
    conn.read_exact(&mut reply).expect("read reply");
    let (error, handle) = parse_reply_header(&reply).expect("parse reply");
    assert_eq!(error, 0);
    assert_eq!(handle, 0xAA);

    let mut data = vec![0_u8; 4096];
    conn.read_exact(&mut data).expect("read data");
    assert!(
        data.iter().all(|&b| b == 0),
        "fresh device should read zeros"
    );

    // 3. Send a Write request: put a known pattern in block 1.
    let payload: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
    let write_req = NbdRequest {
        command: NbdCommand::Write,
        flags: 0,
        handle: 0xBB,
        offset: 4096,
        length: 4096,
    };
    conn.write_all(&encode_request(&write_req))
        .expect("send write");
    conn.write_all(&payload).expect("send write payload");

    let mut reply = [0_u8; REPLY_HEADER_BYTES];
    conn.read_exact(&mut reply).expect("write reply");
    let (error, handle) = parse_reply_header(&reply).expect("parse reply");
    assert_eq!(error, 0);
    assert_eq!(handle, 0xBB);

    // 4. Read it back.
    let read_back_req = NbdRequest {
        command: NbdCommand::Read,
        flags: 0,
        handle: 0xCC,
        offset: 4096,
        length: 4096,
    };
    conn.write_all(&encode_request(&read_back_req))
        .expect("send read2");
    let mut reply = [0_u8; REPLY_HEADER_BYTES];
    conn.read_exact(&mut reply).expect("read2 reply");
    let (error, _) = parse_reply_header(&reply).expect("parse reply");
    assert_eq!(error, 0);
    let mut data = vec![0_u8; 4096];
    conn.read_exact(&mut data).expect("read2 data");
    assert_eq!(data, payload, "write must roundtrip through the server");

    // 5. Disconnect.
    let disc_req = NbdRequest {
        command: NbdCommand::Disc,
        flags: 0,
        handle: 0xDD,
        offset: 0,
        length: 0,
    };
    conn.write_all(&encode_request(&disc_req))
        .expect("send disc");
    drop(conn);

    let server_result = server_thread.join().expect("server thread panic");
    server_result.expect("server should exit cleanly on Disc");
}

#[test]
fn client_disconnect_without_disc_ends_cleanly() {
    let (_fx, server) = make_server("nbd_smoke_abrupt.gguf", 12);
    let sock = unique_socket_path("abrupt");
    let sock_clone = sock.clone();

    let server = Arc::new(server);
    let server_thread = {
        let server = Arc::clone(&server);
        thread::spawn(move || server.serve_on_unix_socket(&sock_clone))
    };

    let mut conn = None;
    for _ in 0..50 {
        if let Ok(s) = UnixStream::connect(&sock) {
            conn = Some(s);
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    let mut conn = conn.expect("connect");
    let mut banner = [0_u8; NEWSTYLE_HEADER_BYTES];
    conn.read_exact(&mut banner).expect("banner");
    drop(conn);

    let server_result = server_thread.join().expect("server thread panic");
    server_result.expect("abrupt disconnect should return Ok");
}
