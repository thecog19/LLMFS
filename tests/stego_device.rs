mod common;

use common::{SyntheticGgufVersion, SyntheticTensorSpec, write_custom_gguf_fixture};

use llmdb::gguf::parser::parse_path;
use llmdb::gguf::quant::{
    GGML_TYPE_F32_ID, GGML_TYPE_Q3_K_ID, GGML_TYPE_Q4_K_ID, GGML_TYPE_Q5_K_ID, GGML_TYPE_Q6_K_ID,
    GGML_TYPE_Q8_0_ID,
};
use llmdb::stego::device::{DeviceError, DeviceOptions, StegoDevice};
use llmdb::stego::integrity::{NO_BLOCK, PendingMetadataOp};
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

    assert_eq!(device.total_blocks(), 5);
    assert_eq!(device.used_blocks().expect("used blocks"), 4);

    let superblock = device.superblock().clone();
    assert_eq!(superblock.fields.total_blocks, 5);
    assert_eq!(superblock.fields.free_list_head, 3);
    assert_eq!(superblock.fields.table_directory_block, NO_BLOCK);
    assert_eq!(superblock.fields.integrity_chain_head, 2);
    assert_eq!(superblock.fields.shadow_block, 4);
    assert_eq!(superblock.fields.pending_target_block, NO_BLOCK);
    assert_eq!(
        superblock.fields.pending_metadata_op,
        PendingMetadataOp::None
    );

    let allocated = device.alloc_block().expect("allocate block");
    assert_eq!(allocated, 3);
    assert_eq!(device.used_blocks().expect("used after alloc"), 5);

    let payload = vec![0x5A; llmdb::BLOCK_SIZE];
    device
        .write_block(allocated, &payload)
        .expect("write logical block");
    let read_back = device.read_block(allocated).expect("read logical block");
    assert_eq!(read_back, payload);

    device.free_block(allocated).expect("free block");
    assert_eq!(device.used_blocks().expect("used after free"), 4);
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
    let target_tensor_name = &plan.tensors[3].name;
    let target_tensor = parsed
        .tensors
        .iter()
        .find(|tensor| &tensor.name == target_tensor_name)
        .expect("find data-block tensor");
    let corrupt_offset = target_tensor
        .absolute_offset(parsed.tensor_data_offset)
        .expect("tensor offset") as usize
        + 2;

    let mut bytes = std::fs::read(&fixture.path).expect("read gguf");
    bytes[corrupt_offset] ^= 0x01;
    std::fs::write(&fixture.path, bytes).expect("rewrite gguf");

    let device = StegoDevice::open(&fixture.path, AllocationMode::Standard).expect("reopen device");
    let error = device.read_block(3).expect_err("crc mismatch should fail");

    assert!(matches!(
        error,
        DeviceError::IntegrityMismatch { block_index: 3, .. }
    ));
}

#[test]
fn reopen_recovers_pending_shadow_write_after_interrupted_commit() {
    let fixture = write_custom_gguf_fixture(
        SyntheticGgufVersion::V3,
        "device_recovery.gguf",
        &[
            q8_tensor("blk.4.ffn_down.weight", 8192),
            q8_tensor("blk.3.ffn_up.weight", 8192),
            q8_tensor("blk.2.attn_q.weight", 8192),
            q8_tensor("blk.1.attn_k.weight", 8192),
            q8_tensor("blk.0.attn_v.weight", 8192),
        ],
    );
    let payload = patterned_payload(0x61);

    {
        let mut device = StegoDevice::initialize_with_options(
            &fixture.path,
            AllocationMode::Standard,
            DeviceOptions { verbose: true },
        )
        .expect("init recovery device");
        let allocated = device.alloc_block().expect("allocate block");
        assert_eq!(allocated, 3);

        device
            .stage_pending_write_for_test(allocated, &payload)
            .expect("stage pending write");
        assert_eq!(device.superblock().fields.pending_target_block, 3);
    }

    let reopened = StegoDevice::open_with_options(
        &fixture.path,
        AllocationMode::Standard,
        DeviceOptions { verbose: true },
    )
    .expect("reopen recovery device");

    assert_eq!(reopened.superblock().fields.pending_target_block, NO_BLOCK);
    assert_eq!(
        reopened
            .verify_integrity()
            .expect("verify recovered integrity"),
        Vec::<u32>::new()
    );
    assert_eq!(
        reopened.read_block(3).expect("read recovered block"),
        payload
    );
}

#[test]
fn reopen_rolls_back_pending_alloc_before_head_advance() {
    let fixture = write_custom_gguf_fixture(
        SyntheticGgufVersion::V3,
        "device_alloc_recovery.gguf",
        &[
            q8_tensor("blk.4.ffn_down.weight", 8192),
            q8_tensor("blk.3.ffn_up.weight", 8192),
            q8_tensor("blk.2.attn_q.weight", 8192),
            q8_tensor("blk.1.attn_k.weight", 8192),
            q8_tensor("blk.0.attn_v.weight", 8192),
        ],
    );

    {
        let mut device = StegoDevice::initialize_with_options(
            &fixture.path,
            AllocationMode::Standard,
            DeviceOptions { verbose: true },
        )
        .expect("init alloc-recovery device");

        let staged = device
            .stage_pending_alloc_for_test()
            .expect("stage pending alloc");
        assert_eq!(staged, 3);
        assert_eq!(
            device.superblock().fields.pending_metadata_op,
            PendingMetadataOp::AllocHeadAdvance
        );
    }

    let mut reopened = StegoDevice::open_with_options(
        &fixture.path,
        AllocationMode::Standard,
        DeviceOptions { verbose: true },
    )
    .expect("reopen alloc-recovery device");

    assert_eq!(
        reopened.superblock().fields.pending_metadata_op,
        PendingMetadataOp::None
    );
    assert_eq!(reopened.superblock().fields.free_list_head, 3);
    assert_eq!(reopened.alloc_block().expect("allocate after recovery"), 3);
}

#[test]
fn reopen_finalizes_pending_free_after_block_rewrite() {
    let fixture = write_custom_gguf_fixture(
        SyntheticGgufVersion::V3,
        "device_free_recovery.gguf",
        &[
            q8_tensor("blk.4.ffn_down.weight", 8192),
            q8_tensor("blk.3.ffn_up.weight", 8192),
            q8_tensor("blk.2.attn_q.weight", 8192),
            q8_tensor("blk.1.attn_k.weight", 8192),
            q8_tensor("blk.0.attn_v.weight", 8192),
        ],
    );

    {
        let mut device = StegoDevice::initialize_with_options(
            &fixture.path,
            AllocationMode::Standard,
            DeviceOptions { verbose: true },
        )
        .expect("init free-recovery device");
        let allocated = device.alloc_block().expect("allocate block");
        assert_eq!(allocated, 3);
        device
            .write_block(allocated, &patterned_payload(0x23))
            .expect("write allocated block");

        device
            .stage_pending_free_for_test(allocated)
            .expect("stage pending free");
        assert_eq!(
            device.superblock().fields.pending_metadata_op,
            PendingMetadataOp::FreeHeadPush
        );
    }

    let mut reopened = StegoDevice::open_with_options(
        &fixture.path,
        AllocationMode::Standard,
        DeviceOptions { verbose: true },
    )
    .expect("reopen free-recovery device");

    assert_eq!(
        reopened.superblock().fields.pending_metadata_op,
        PendingMetadataOp::None
    );
    assert_eq!(reopened.superblock().fields.free_list_head, 3);
    assert_eq!(reopened.used_blocks().expect("used after free recovery"), 4);
    assert_eq!(reopened.alloc_block().expect("reallocate freed block"), 3);
}

#[test]
fn reopen_uses_backup_superblock_when_primary_is_corrupt() {
    let fixture = write_custom_gguf_fixture(
        SyntheticGgufVersion::V3,
        "device_superblock_mirror.gguf",
        &[
            q8_tensor("blk.4.ffn_down.weight", 8192),
            q8_tensor("blk.3.ffn_up.weight", 8192),
            q8_tensor("blk.2.attn_q.weight", 8192),
            q8_tensor("blk.1.attn_k.weight", 8192),
            q8_tensor("blk.0.attn_v.weight", 8192),
        ],
    );

    {
        let mut device = StegoDevice::initialize_with_options(
            &fixture.path,
            AllocationMode::Standard,
            DeviceOptions { verbose: true },
        )
        .expect("init mirrored-superblock device");
        device.flush().expect("flush mirrored device");
    }

    let parsed = parse_path(&fixture.path).expect("parse fixture");
    let plan = build_allocation_plan(&parsed.tensors, AllocationMode::Standard);
    let primary_tensor_name = &plan.tensors[0].name;
    let primary_tensor = parsed
        .tensors
        .iter()
        .find(|tensor| &tensor.name == primary_tensor_name)
        .expect("find primary superblock tensor");
    let corrupt_offset = primary_tensor
        .absolute_offset(parsed.tensor_data_offset)
        .expect("primary tensor offset") as usize
        + 3;

    let mut bytes = std::fs::read(&fixture.path).expect("read gguf");
    bytes[corrupt_offset] ^= 0x80;
    std::fs::write(&fixture.path, bytes).expect("rewrite gguf");

    let mut reopened = StegoDevice::open_with_options(
        &fixture.path,
        AllocationMode::Standard,
        DeviceOptions { verbose: true },
    )
    .expect("reopen with backup superblock");

    assert_eq!(reopened.total_blocks(), 5);
    assert_eq!(reopened.superblock().fields.free_list_head, 3);
    assert_eq!(
        reopened.alloc_block().expect("allocate after backup load"),
        3
    );
}

#[test]
fn mixed_quant_device_roundtrips_across_reopen_and_integrity_scan() {
    let fixture = write_custom_gguf_fixture(
        SyntheticGgufVersion::V3,
        "device_mixed_quant.gguf",
        &mixed_quant_tensors(),
    );

    let payloads = [
        (3_u32, vec![0x11; llmdb::BLOCK_SIZE]),
        (4_u32, vec![0x22; llmdb::BLOCK_SIZE]),
        (5_u32, vec![0x33; llmdb::BLOCK_SIZE]),
        (6_u32, patterned_payload(0x40)),
        (7_u32, patterned_payload(0x80)),
    ];

    {
        let mut device = StegoDevice::initialize_with_options(
            &fixture.path,
            AllocationMode::Standard,
            DeviceOptions { verbose: true },
        )
        .expect("init mixed-quant device");

        assert_eq!(device.total_blocks(), 9);
        assert_eq!(device.integrity_block_count(), 1);
        assert_eq!(device.data_region_start(), 3);
        assert_eq!(device.shadow_block(), 8);

        for (expected_block, payload) in &payloads {
            let allocated = device.alloc_block().expect("allocate block");
            assert_eq!(allocated, *expected_block);
            device.write_block(allocated, payload).expect("write block");
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
        reopened
            .verify_integrity()
            .expect("verify reopened integrity"),
        Vec::<u32>::new()
    );

    for (block_index, payload) in &payloads {
        let read_back = reopened.read_block(*block_index).expect("read back block");
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
