mod common;

use common::{SyntheticGgufVersion, SyntheticTensorSpec, write_custom_gguf_fixture};

use llmdb::gguf::parser::parse_path;
use llmdb::gguf::quant::{
    GGML_TYPE_F32_ID, GGML_TYPE_Q3_K_ID, GGML_TYPE_Q4_K_ID, GGML_TYPE_Q5_K_ID, GGML_TYPE_Q6_K_ID,
    GGML_TYPE_Q8_0_ID,
};
use llmdb::stego::device::{DeviceOptions, StegoDevice};
use llmdb::stego::integrity::NO_BLOCK;
use llmdb::stego::planner::{AllocationMode, build_allocation_plan};

#[test]
fn fresh_device_init_chains_free_blocks_and_roundtrips_one_block() {
    let fixture = write_custom_gguf_fixture(
        SyntheticGgufVersion::V3,
        "device_fixture.gguf",
        &[
            q8_tensor("blk.4.ffn_down.weight", 8192),
            q8_tensor("blk.3.ffn_up.weight", 8192),
            q8_tensor("blk.2.attn_q.weight", 8192),
            q8_tensor("blk.1.attn_k.weight", 8192),
            q8_tensor("blk.0.attn_v.weight", 8192),
        ],
    );

    let mut device =
        StegoDevice::initialize(&fixture.path, AllocationMode::Standard).expect("init device");

    // Layout: block 0 = superblock, block 1 = integrity, blocks 2..4 = data (free list)
    assert_eq!(device.total_blocks(), 5);
    assert_eq!(device.integrity_block_count(), 1);
    assert_eq!(device.data_region_start(), 2);

    let superblock = device.superblock().clone();
    assert_eq!(superblock.fields.total_blocks, 5);
    assert_eq!(superblock.fields.free_list_head, 2);
    assert_eq!(superblock.fields.integrity_chain_head, 1);

    let allocated = device.alloc_block().expect("allocate block");
    assert_eq!(allocated, 2);

    let payload = vec![0x5A; llmdb::BLOCK_SIZE];
    device
        .write_block(allocated, &payload)
        .expect("write logical block");
    let read_back = device.read_block(allocated).expect("read logical block");
    assert_eq!(read_back, payload);

    device.free_block(allocated).expect("free block");
}

#[test]
fn read_block_detects_crc_mismatch_after_out_of_band_corruption() {
    let fixture = write_custom_gguf_fixture(
        SyntheticGgufVersion::V3,
        "device_corruption.gguf",
        &[
            q8_tensor("blk.4.ffn_down.weight", 8192),
            q8_tensor("blk.3.ffn_up.weight", 8192),
            q8_tensor("blk.2.attn_q.weight", 8192),
            q8_tensor("blk.1.attn_k.weight", 8192),
            q8_tensor("blk.0.attn_v.weight", 8192),
        ],
    );

    {
        let mut device =
            StegoDevice::initialize(&fixture.path, AllocationMode::Standard).expect("init device");
        let allocated = device.alloc_block().expect("allocate block");
        let payload = vec![0xA5; llmdb::BLOCK_SIZE];
        device
            .write_block(allocated, &payload)
            .expect("write block");
        device.flush().expect("flush device");
    }

    let parsed = parse_path(&fixture.path).expect("parse fixture");
    let plan = build_allocation_plan(&parsed.tensors, AllocationMode::Standard);
    // Block 2 = first data block. Each Q8_0 tensor with 8192 weights holds
    // exactly one 4096-byte logical block. The plan orders Tier1 (FFN) first,
    // so the third tensor in plan order (index 2 = first Tier2 attention
    // tensor) maps to logical block 2.
    let target_tensor = parsed
        .tensors
        .iter()
        .find(|tensor| tensor.name == plan.tensors[2].name)
        .expect("find tensor covering block 2");
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

    let result = device.read_block(2);
    assert!(
        result.is_err(),
        "reading a corrupted block should return an integrity error"
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
    let total;

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

        total = device.total_blocks();
        data_start = device.data_region_start();
        assert!(total > data_start + 5, "need >=5 data blocks");

        let mut allocated = Vec::new();
        for payload in &payloads {
            let block_index = device.alloc_block().expect("allocate block");
            device.write_block(block_index, payload).expect("write block");
            allocated.push(block_index);
        }

        assert_eq!(
            device.verify_integrity().expect("verify integrity"),
            Vec::<u32>::new()
        );
        device.flush().expect("flush mixed device");
    }

    let reopened = StegoDevice::open_with_options(
        &fixture.path,
        AllocationMode::Standard,
        DeviceOptions { verbose: true },
    )
    .expect("reopen mixed device");

    assert_eq!(
        reopened.verify_integrity().expect("verify reopened integrity"),
        Vec::<u32>::new()
    );

    for (index, payload) in payloads.iter().enumerate() {
        let block_index = data_start + index as u32;
        let read_back = reopened.read_block(block_index).expect("read back block");
        assert_eq!(read_back, *payload);
    }
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
