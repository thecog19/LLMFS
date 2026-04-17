mod common;

use common::{SyntheticGgufVersion, SyntheticTensorSpec, write_custom_gguf_fixture};
use llmdb::gguf::quant::GGML_TYPE_Q8_0_ID;
use llmdb::nbd::server::NbdServer;
use llmdb::stego::device::{DeviceOptions, StegoDevice};
use llmdb::stego::planner::AllocationMode;

fn q8_tensors(count: usize) -> Vec<SyntheticTensorSpec> {
    (0..count)
        .map(|i| {
            let name = format!("blk.{}.ffn_down.weight", count - 1 - i);
            let weight_count = 8192_usize;
            let chunk_count = weight_count / 32;
            SyntheticTensorSpec {
                name,
                dimensions: vec![weight_count as u64],
                raw_type_id: GGML_TYPE_Q8_0_ID,
                data: vec![0_u8; chunk_count * 34],
            }
        })
        .collect()
}

fn open_fresh_server(name: &str, tensor_count: usize) -> (common::FixtureHandle, NbdServer) {
    let fx = write_custom_gguf_fixture(SyntheticGgufVersion::V3, name, &q8_tensors(tensor_count));
    let device = StegoDevice::initialize_with_options(
        &fx.path,
        AllocationMode::Standard,
        DeviceOptions { verbose: false },
    )
    .expect("init");
    let server = NbdServer::new(device);
    (fx, server)
}

#[test]
fn fresh_device_read_returns_zeros_across_data_region() {
    let (_fx, server) = open_fresh_server("nbd_align_zero.gguf", 12);
    let data = server.handle_read(0, 4096).expect("read full block");
    assert_eq!(data.len(), 4096);
    assert!(
        data.iter().all(|&b| b == 0),
        "fresh data block should read as zeros"
    );
}

#[test]
fn unaligned_read_at_offset_100_length_3000_spans_single_block() {
    let (_fx, server) = open_fresh_server("nbd_align_mid.gguf", 12);
    let data = server.handle_read(100, 3000).expect("unaligned read");
    assert_eq!(data.len(), 3000);
    assert!(
        data.iter().all(|&b| b == 0),
        "bytes 100..3100 of a fresh device should be zero"
    );
}

#[test]
fn unaligned_read_across_two_blocks_returns_concatenated_bytes() {
    // 2 blocks = 8192 bytes. Request bytes 100..5096 = 4996 bytes straddling
    // the 0/1 block boundary at byte 4096.
    let (_fx, server) = open_fresh_server("nbd_align_cross.gguf", 12);
    let data = server.handle_read(100, 4996).expect("cross-block read");
    assert_eq!(data.len(), 4996);
    assert!(data.iter().all(|&b| b == 0));
}

#[test]
fn write_then_read_roundtrips_unaligned_range() {
    let (_fx, server) = open_fresh_server("nbd_align_rmw.gguf", 12);

    let payload: Vec<u8> = (0..3000).map(|i| (i % 251) as u8).collect();
    server.handle_write(100, &payload).expect("unaligned write");

    let readback = server.handle_read(100, 3000).expect("read back");
    assert_eq!(readback, payload);

    // Bytes before and after the write should still read as zero.
    let head = server.handle_read(0, 100).expect("read head");
    assert!(head.iter().all(|&b| b == 0));
    let tail = server.handle_read(3100, 996).expect("read tail");
    assert!(tail.iter().all(|&b| b == 0));
}

#[test]
fn full_block_aligned_write_persists_across_subsequent_read() {
    let (_fx, server) = open_fresh_server("nbd_align_aligned.gguf", 12);

    let block: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
    server.handle_write(4096, &block).expect("write block 1");

    let readback = server.handle_read(4096, 4096).expect("read block 1");
    assert_eq!(readback, block);

    // Block 0 and block 2 still zero.
    let block0 = server.handle_read(0, 4096).expect("read block 0");
    assert!(block0.iter().all(|&b| b == 0));
}

#[test]
fn write_spanning_three_blocks_splices_middle_whole_block() {
    // Offset 100, length 8192+3000-100 = ... Actually craft a write that
    // covers tail of block 0, all of block 1, head of block 2.
    let (_fx, server) = open_fresh_server("nbd_align_three.gguf", 12);

    let first_partial = 4096 - 100; // 3996 bytes into block 0
    let middle_full = 4096;
    let last_partial = 500;
    let total = first_partial + middle_full + last_partial;

    let payload: Vec<u8> = (0..total).map(|i| ((i * 7) % 251) as u8).collect();
    server
        .handle_write(100, &payload)
        .expect("three-block write");

    let readback = server.handle_read(100, total as u32).expect("read back");
    assert_eq!(readback, payload);

    // Confirm the middle block is readable as a full aligned block.
    let middle = server.handle_read(4096, 4096).expect("middle aligned read");
    assert_eq!(
        &middle[..],
        &payload[first_partial..first_partial + middle_full]
    );
}

#[test]
fn read_past_end_of_export_is_rejected() {
    let (_fx, server) = open_fresh_server("nbd_align_oob.gguf", 12);
    let export = server.export_bytes();
    let result = server.handle_read(export - 100, 4096);
    assert!(result.is_err(), "read past end should return OutOfRange");
}

#[test]
fn export_bytes_matches_data_region_capacity() {
    let (_fx, server) = open_fresh_server("nbd_align_export.gguf", 12);
    // 12 tensors × 1 block each = 12 total blocks; metadata = 4 (super +
    // integrity + redirection + file-table); data = 8 blocks.
    assert_eq!(server.export_bytes(), 8 * 4096);
}
