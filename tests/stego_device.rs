mod common;

use common::{SyntheticGgufVersion, SyntheticTensorSpec, write_custom_gguf_fixture};

use llmdb::gguf::parser::parse_path;
use llmdb::gguf::quant::{
    GGML_TYPE_F32_ID, GGML_TYPE_Q3_K_ID, GGML_TYPE_Q4_K_ID, GGML_TYPE_Q5_K_ID, GGML_TYPE_Q6_K_ID,
    GGML_TYPE_Q8_0_ID,
};
use llmdb::stego::device::{DeviceOptions, StegoDevice};
use llmdb::stego::planner::{AllocationMode, build_allocation_plan};

#[test]
fn fresh_device_init_roundtrips_one_block() {
    // 8 tensors × 8192 Q8_0 weights = 8 blocks total.
    // Layout: 1 super + 1 integrity + 1 redir + 1 filetable = 4 metadata, 4 data blocks.
    let fixture = write_custom_gguf_fixture(
        SyntheticGgufVersion::V3,
        "device_fixture.gguf",
        &make_q8_tensors(8),
    );

    let mut device =
        StegoDevice::initialize(&fixture.path, AllocationMode::Standard).expect("init device");

    assert_eq!(device.total_blocks(), 8);
    assert_eq!(device.data_region_start(), 4);

    let sb = device.superblock().clone();
    assert_eq!(sb.fields.total_blocks, 8);
    assert_eq!(sb.fields.free_list_head, 4);
    assert_eq!(sb.fields.integrity_chain_head, 1);
    assert_eq!(sb.fields.redirection_table_start, 2);
    assert_eq!(sb.fields.redirection_table_length, 1);
    assert_eq!(sb.fields.file_table_start, 3);
    assert_eq!(sb.fields.file_table_length, 1);

    let allocated = device.alloc_block().expect("allocate block");
    assert_eq!(allocated, 4);

    let payload = vec![0x5A; llmdb::BLOCK_SIZE];
    device
        .write_block(allocated, &payload)
        .expect("write block");
    let read_back = device.read_block(allocated).expect("read block");
    assert_eq!(read_back, payload);

    device.free_block(allocated).expect("free block");
}

#[test]
fn alloc_write_free_cycle_returns_blocks_to_freelist() {
    let fixture = write_custom_gguf_fixture(
        SyntheticGgufVersion::V3,
        "device_alloc_free.gguf",
        &make_q8_tensors(10),
    );

    let mut device =
        StegoDevice::initialize(&fixture.path, AllocationMode::Standard).expect("init device");

    let data_start = device.data_region_start();
    let total = device.total_blocks();
    let expected_data_blocks = total - data_start;

    // Allocate all data blocks
    let mut allocated = Vec::new();
    for _ in 0..expected_data_blocks {
        allocated.push(device.alloc_block().expect("alloc"));
    }
    assert!(device.alloc_block().is_err(), "should be out of space");

    // Write to each
    for block_index in &allocated {
        let payload = vec![*block_index as u8; llmdb::BLOCK_SIZE];
        device.write_block(*block_index, &payload).expect("write");
    }

    // Read back each
    for block_index in &allocated {
        let read_back = device.read_block(*block_index).expect("read");
        assert_eq!(read_back[0], *block_index as u8);
    }

    // Free all
    for block_index in allocated.iter().rev() {
        device.free_block(*block_index).expect("free");
    }

    // Verify we can allocate them all again
    for _ in 0..expected_data_blocks {
        device.alloc_block().expect("re-alloc after free");
    }
}

#[test]
fn read_block_detects_crc_mismatch_after_out_of_band_corruption() {
    let fixture = write_custom_gguf_fixture(
        SyntheticGgufVersion::V3,
        "device_corruption.gguf",
        &make_q8_tensors(8),
    );

    let data_start;
    {
        let mut device =
            StegoDevice::initialize(&fixture.path, AllocationMode::Standard).expect("init device");
        data_start = device.data_region_start();
        let allocated = device.alloc_block().expect("allocate block");
        let payload = vec![0xA5; llmdb::BLOCK_SIZE];
        device
            .write_block(allocated, &payload)
            .expect("write block");
        device.flush().expect("flush device");
    }

    // Corrupt a weight byte that contributes to the first data block.
    let parsed = parse_path(&fixture.path).expect("parse fixture");
    let plan = build_allocation_plan(&parsed.tensors, AllocationMode::Standard);
    // data_start-th tensor in plan order covers the first data block's stego bytes.
    let target_tensor = parsed
        .tensors
        .iter()
        .find(|tensor| tensor.name == plan.tensors[data_start as usize].name)
        .expect("find tensor covering data block");
    let corrupt_offset = target_tensor
        .absolute_offset(parsed.tensor_data_offset)
        .expect("tensor offset") as usize
        + 2;

    let mut bytes = std::fs::read(&fixture.path).expect("read gguf");
    bytes[corrupt_offset] ^= 0x01;
    std::fs::write(&fixture.path, bytes).expect("rewrite gguf");

    let device = StegoDevice::open_with_options(
        &fixture.path,
        AllocationMode::Standard,
        DeviceOptions { verbose: false },
    )
    .expect("open device");

    let result = device.read_block(data_start);
    assert!(
        result.is_err(),
        "reading a corrupted block should return an integrity error"
    );
}

#[test]
fn device_roundtrips_across_reopen() {
    let fixture = write_custom_gguf_fixture(
        SyntheticGgufVersion::V3,
        "device_reopen.gguf",
        &make_q8_tensors(10),
    );

    let data_start;
    let payload = patterned_payload(0x42);
    let block_index;

    {
        let mut device = StegoDevice::initialize_with_options(
            &fixture.path,
            AllocationMode::Standard,
            DeviceOptions { verbose: true },
        )
        .expect("init device");

        data_start = device.data_region_start();
        block_index = device.alloc_block().expect("allocate");
        device.write_block(block_index, &payload).expect("write");
        device.flush().expect("flush");
    }

    let reopened = StegoDevice::open_with_options(
        &fixture.path,
        AllocationMode::Standard,
        DeviceOptions { verbose: true },
    )
    .expect("reopen device");

    assert_eq!(reopened.data_region_start(), data_start);
    let read_back = reopened.read_block(block_index).expect("read after reopen");
    assert_eq!(read_back, payload);
    assert_eq!(
        reopened.verify_integrity().expect("verify"),
        Vec::<u32>::new()
    );
}

#[test]
fn mixed_quant_device_roundtrips_across_reopen_and_integrity_scan() {
    let fixture = write_custom_gguf_fixture(
        SyntheticGgufVersion::V3,
        "device_mixed_quant.gguf",
        &mixed_quant_tensors(),
    );

    let data_start;
    let payloads = [
        vec![0x11; llmdb::BLOCK_SIZE],
        vec![0x22; llmdb::BLOCK_SIZE],
        vec![0x33; llmdb::BLOCK_SIZE],
        patterned_payload(0x40),
        patterned_payload(0x80),
    ];

    {
        let mut device = StegoDevice::initialize_with_options(
            &fixture.path,
            AllocationMode::Standard,
            DeviceOptions { verbose: true },
        )
        .expect("init mixed-quant device");

        data_start = device.data_region_start();
        let data_blocks = device.total_blocks() - data_start;
        assert!(data_blocks >= 5, "need >=5 data blocks, got {data_blocks}");

        let mut allocated = Vec::new();
        for payload in &payloads {
            let block_index = device.alloc_block().expect("allocate block");
            device.write_block(block_index, payload).expect("write");
            allocated.push(block_index);
        }

        assert_eq!(
            device.verify_integrity().expect("verify"),
            Vec::<u32>::new()
        );
        device.flush().expect("flush");
    }

    let reopened = StegoDevice::open_with_options(
        &fixture.path,
        AllocationMode::Standard,
        DeviceOptions { verbose: true },
    )
    .expect("reopen mixed device");

    assert_eq!(
        reopened.verify_integrity().expect("verify reopened"),
        Vec::<u32>::new()
    );

    for (index, payload) in payloads.iter().enumerate() {
        let block_index = data_start + index as u32;
        let read_back = reopened.read_block(block_index).expect("read back");
        assert_eq!(read_back, *payload);
    }
}

/// Generate N Q8_0 tensors alternating FFN/attention names, each with
/// 8192 weights = 4096 stego bytes = 1 block.
fn make_q8_tensors(count: usize) -> Vec<SyntheticTensorSpec> {
    let names = [
        "blk.{i}.ffn_down.weight",
        "blk.{i}.ffn_up.weight",
        "blk.{i}.attn_q.weight",
        "blk.{i}.attn_k.weight",
    ];

    (0..count)
        .map(|index| {
            let template = names[index % names.len()];
            let layer = count - 1 - index; // deepest first
            let name = template.replace("{i}", &layer.to_string());
            q8_tensor(&name, 8192)
        })
        .collect()
}

fn q8_tensor(name: &str, weight_count: usize) -> SyntheticTensorSpec {
    let chunk_count = weight_count / 32;
    SyntheticTensorSpec {
        name: name.to_owned(),
        dimensions: vec![weight_count as u64],
        raw_type_id: GGML_TYPE_Q8_0_ID,
        data: vec![0_u8; chunk_count * 34],
    }
}

fn mixed_quant_tensors() -> Vec<SyntheticTensorSpec> {
    vec![
        q8_tensor("blk.10.ffn_down.weight", 24_576),
        SyntheticTensorSpec {
            name: "blk.9.ffn_up.weight".to_owned(),
            dimensions: vec![4_096],
            raw_type_id: GGML_TYPE_F32_ID,
            data: vec![0_u8; 16_384],
        },
        SyntheticTensorSpec {
            name: "blk.8.attn_q.weight".to_owned(),
            dimensions: vec![16_384],
            raw_type_id: GGML_TYPE_Q6_K_ID,
            data: vec![0_u8; 13_440],
        },
        SyntheticTensorSpec {
            name: "blk.7.attn_v.weight".to_owned(),
            dimensions: vec![16_384],
            raw_type_id: GGML_TYPE_Q5_K_ID,
            data: vec![0_u8; 11_264],
        },
        SyntheticTensorSpec {
            name: "blk.6.ffn_gate.weight".to_owned(),
            dimensions: vec![16_384],
            raw_type_id: GGML_TYPE_Q4_K_ID,
            data: vec![0_u8; 9_216],
        },
        SyntheticTensorSpec {
            name: "blk.5.attn_output.weight".to_owned(),
            dimensions: vec![32_768],
            raw_type_id: GGML_TYPE_Q3_K_ID,
            data: vec![0_u8; 14_080],
        },
        q8_tensor("blk.4.attn_k.weight", 8_192),
        q8_tensor("blk.3.attn_v.weight", 8_192),
    ]
}

fn patterned_payload(seed: u8) -> Vec<u8> {
    let mut payload = vec![0_u8; llmdb::BLOCK_SIZE];
    for (index, byte) in payload.iter_mut().enumerate() {
        *byte = seed.wrapping_add((index % 251) as u8);
    }
    payload
}
