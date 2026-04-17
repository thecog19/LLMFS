mod common;

use common::{SyntheticGgufVersion, write_custom_gguf_fixture};

use llmdb::stego::device::{CrashPoint, DeviceOptions, StegoDevice};
use llmdb::stego::planner::AllocationMode;

fn verbose() -> DeviceOptions {
    DeviceOptions { verbose: true }
}

/// 12 tensors × 8192 Q8_0 weights = 12 blocks. Layout: 4 metadata + 8 data.
/// Enough room for shadow-copy operations.
fn make_fixture(name: &str) -> common::FixtureHandle {
    write_custom_gguf_fixture(
        SyntheticGgufVersion::V3,
        name,
        &make_q8_tensors(12),
    )
}

#[test]
fn crash_after_shadow_write_leaves_old_block_intact() {
    let fixture = make_fixture("shadow_crash_1.gguf");

    let old_payload = vec![0xAA; llmdb::BLOCK_SIZE];
    let new_payload = vec![0xBB; llmdb::BLOCK_SIZE];
    let block_index;
    let used_before;

    {
        let mut device =
            StegoDevice::initialize_with_options(&fixture.path, AllocationMode::Standard, verbose())
                .expect("init");

        block_index = device.alloc_block().expect("alloc");
        device
            .write_block(block_index, &old_payload)
            .expect("write old");

        used_before = device.used_blocks().expect("used");

        // Simulate crash after shadow data is written + flushed but BEFORE
        // the redirection flip. Old data should survive.
        device
            .write_block_with_crash_after(block_index, &new_payload, CrashPoint::AfterShadowFlush)
            .expect("crash write");

        // Drop WITHOUT close → dirty flag stays set
        std::mem::forget(device);
    }

    // Reopen — dirty flag triggers recovery
    let device =
        StegoDevice::open_with_options(&fixture.path, AllocationMode::Standard, verbose())
            .expect("reopen");

    let read_back = device.read_block(block_index).expect("read after crash");
    assert_eq!(
        read_back, old_payload,
        "old data must survive crash before redirection flip"
    );

    // Orphan shadow block should be reclaimed
    assert_eq!(
        device.used_blocks().expect("used after recovery"),
        used_before,
        "shadow block should be returned to free list by recovery"
    );
}

#[test]
fn crash_after_redirection_flip_sees_new_value() {
    let fixture = make_fixture("shadow_crash_2.gguf");

    let old_payload = vec![0xCC; llmdb::BLOCK_SIZE];
    let new_payload = vec![0xDD; llmdb::BLOCK_SIZE];
    let block_index;

    {
        let mut device =
            StegoDevice::initialize_with_options(&fixture.path, AllocationMode::Standard, verbose())
                .expect("init");

        block_index = device.alloc_block().expect("alloc");
        device
            .write_block(block_index, &old_payload)
            .expect("write old");

        // Crash AFTER the redirection flip is flushed but before the old
        // physical is returned to the free list.
        device
            .write_block_with_crash_after(
                block_index,
                &new_payload,
                CrashPoint::AfterRedirectionFlush,
            )
            .expect("crash write");

        std::mem::forget(device);
    }

    let device =
        StegoDevice::open_with_options(&fixture.path, AllocationMode::Standard, verbose())
            .expect("reopen");

    let read_back = device.read_block(block_index).expect("read after crash");
    assert_eq!(
        read_back, new_payload,
        "new data must be visible after redirection flip"
    );
}

#[test]
fn unclean_shutdown_sets_dirty_and_recovery_clears_it() {
    let fixture = make_fixture("shadow_dirty.gguf");

    {
        let mut device =
            StegoDevice::initialize_with_options(&fixture.path, AllocationMode::Standard, verbose())
                .expect("init");

        // Device open should have set dirty flag
        let block = device.alloc_block().expect("alloc");
        let payload = vec![0xEE; llmdb::BLOCK_SIZE];
        device.write_block(block, &payload).expect("write");

        // forget → no close → dirty stays set
        std::mem::forget(device);
    }

    // Reopen — recovery should run. Device is dirty while open (that's correct).
    {
        let device =
            StegoDevice::open_with_options(&fixture.path, AllocationMode::Standard, verbose())
                .expect("reopen after dirty");

        // Dirty is set while open — that's the point (tracks liveness).
        assert!(
            device.superblock().is_dirty(),
            "dirty flag must be set while device is open"
        );

        // Explicit close clears it.
        device.close().expect("close");
    }

    // Reopen again — should NOT trigger recovery (clean shutdown left dirty=false on disk).
    // After open, dirty is set again (liveness tracking), but no recovery log.
    {
        let device =
            StegoDevice::open_with_options(&fixture.path, AllocationMode::Standard, verbose())
                .expect("reopen after clean close");

        // Clean close means the on-disk superblock had dirty=false before
        // this open. open_with_options set dirty=true for liveness. But no
        // recovery should have run (verified by the absence of "unclean
        // shutdown" log in verbose output — not assertable here, but the
        // data integrity check below confirms correctness).
        let block_data = device.read_block(4).expect("read stored block");
        assert_eq!(block_data, vec![0xEE; llmdb::BLOCK_SIZE]);
    }
}

#[test]
fn overwrite_after_overwrite_reclaims_previous_shadow() {
    // Covers the two-stage shadow transition: the first overwrite leaves the
    // old physical in place (old == logical key), the second overwrite pushes
    // the previous shadow back to the free list.
    let fixture = make_fixture("shadow_double_overwrite.gguf");

    let mut device =
        StegoDevice::initialize_with_options(&fixture.path, AllocationMode::Standard, verbose())
            .expect("init");

    let block = device.alloc_block().expect("alloc");

    let v1 = vec![0x01_u8; llmdb::BLOCK_SIZE];
    let v2 = vec![0x02_u8; llmdb::BLOCK_SIZE];
    let v3 = vec![0x03_u8; llmdb::BLOCK_SIZE];

    device.write_block(block, &v1).expect("write v1 (direct)");
    let used_after_v1 = device.used_blocks().expect("used v1");

    device.write_block(block, &v2).expect("write v2 (first shadow)");
    let used_after_v2 = device.used_blocks().expect("used v2");
    assert_eq!(
        used_after_v2,
        used_after_v1 + 1,
        "first shadow consumes one block — old physical is kept as the logical key"
    );

    device.write_block(block, &v3).expect("write v3 (second shadow)");
    let used_after_v3 = device.used_blocks().expect("used v3");
    assert_eq!(
        used_after_v3, used_after_v2,
        "second shadow reclaims the previous shadow into the free list"
    );

    let read_back = device.read_block(block).expect("read v3");
    assert_eq!(read_back, v3);

    assert_eq!(
        device.verify_integrity().expect("verify"),
        Vec::<u32>::new(),
        "integrity scan must be clean after overwrites"
    );

    device.close().expect("close");
}

#[test]
fn alloc_returns_distinct_indices_after_shadow_copy() {
    // Regression for the bug where pushing `old_physical == logical_key` to
    // the free list on the first shadow caused alloc_block to hand out the
    // same logical index twice.
    let fixture = make_fixture("shadow_alloc_uniqueness.gguf");

    let mut device =
        StegoDevice::initialize_with_options(&fixture.path, AllocationMode::Standard, verbose())
            .expect("init");

    let first = device.alloc_block().expect("alloc first");
    device
        .write_block(first, &vec![0xAA_u8; llmdb::BLOCK_SIZE])
        .expect("write first (direct)");
    device
        .write_block(first, &vec![0xBB_u8; llmdb::BLOCK_SIZE])
        .expect("overwrite first (shadow)");

    let second = device.alloc_block().expect("alloc second");
    assert_ne!(
        second, first,
        "second alloc must not return the same logical as the shadow'd first"
    );

    device
        .write_block(second, &vec![0xCC_u8; llmdb::BLOCK_SIZE])
        .expect("write second");

    assert_eq!(
        device.read_block(first).expect("read first"),
        vec![0xBB_u8; llmdb::BLOCK_SIZE]
    );
    assert_eq!(
        device.read_block(second).expect("read second"),
        vec![0xCC_u8; llmdb::BLOCK_SIZE]
    );

    device.close().expect("close");
}

#[test]
fn recovery_is_idempotent() {
    let fixture = make_fixture("shadow_idempotent.gguf");

    {
        let mut device =
            StegoDevice::initialize_with_options(&fixture.path, AllocationMode::Standard, verbose())
                .expect("init");

        let block = device.alloc_block().expect("alloc");
        device
            .write_block(block, &vec![0xFF; llmdb::BLOCK_SIZE])
            .expect("write");

        std::mem::forget(device);
    }

    // First recovery
    let used_after_first;
    {
        let device =
            StegoDevice::open_with_options(&fixture.path, AllocationMode::Standard, verbose())
                .expect("first recovery");
        used_after_first = device.used_blocks().expect("used");
        std::mem::forget(device);
    }

    // Second recovery (device was forgotten again, dirty set again)
    {
        let device =
            StegoDevice::open_with_options(&fixture.path, AllocationMode::Standard, verbose())
                .expect("second recovery");
        let used_after_second = device.used_blocks().expect("used");
        assert_eq!(
            used_after_first, used_after_second,
            "recovery must be idempotent"
        );
    }
}

fn make_q8_tensors(count: usize) -> Vec<common::SyntheticTensorSpec> {
    let names = [
        "blk.{i}.ffn_down.weight",
        "blk.{i}.ffn_up.weight",
        "blk.{i}.attn_q.weight",
        "blk.{i}.attn_k.weight",
    ];
    (0..count)
        .map(|index| {
            let template = names[index % names.len()];
            let layer = count - 1 - index;
            let name = template.replace("{i}", &layer.to_string());
            common::SyntheticTensorSpec {
                name,
                dimensions: vec![8192],
                raw_type_id: llmdb::gguf::quant::GGML_TYPE_Q8_0_ID,
                data: vec![0_u8; (8192 / 32) * 34],
            }
        })
        .collect()
}
