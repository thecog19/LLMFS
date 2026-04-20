mod common;

use std::io::Write;

use common::{SyntheticGgufVersion, SyntheticTensorSpec, write_custom_gguf_fixture};
use llmdb::fs::file_ops::FsError;
use llmdb::gguf::quant::GGML_TYPE_Q8_0_ID;
use llmdb::stego::device::{DeviceOptions, StegoDevice};
use llmdb::stego::planner::AllocationMode;

fn q8_tensor(name: &str, weight_count: usize) -> SyntheticTensorSpec {
    let chunk_count = weight_count / 32;
    SyntheticTensorSpec {
        name: name.to_owned(),
        dimensions: vec![weight_count as u64],
        raw_type_id: GGML_TYPE_Q8_0_ID,
        data: vec![0_u8; chunk_count * 34],
    }
}

fn make_q8_tensors(count: usize) -> Vec<SyntheticTensorSpec> {
    (0..count)
        .map(|i| q8_tensor(&format!("blk.{}.ffn_down.weight", count - 1 - i), 8192))
        .collect()
}

fn open_device(name: &str, tensor_count: usize) -> (common::FixtureHandle, StegoDevice) {
    let fixture = write_custom_gguf_fixture(
        SyntheticGgufVersion::V3,
        name,
        &make_q8_tensors(tensor_count),
    );
    let device = StegoDevice::initialize_with_options(
        &fixture.path,
        AllocationMode::Standard,
        DeviceOptions { verbose: false },
    )
    .expect("init device");
    (fixture, device)
}

fn patterned_bytes(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| seed.wrapping_add((i % 251) as u8))
        .collect()
}

#[test]
fn store_list_get_roundtrips_small_file() {
    let (_fixture, mut device) = open_device("file_ops_small.gguf", 12);

    let data = patterned_bytes(1024, 0x17);
    let host = tempfile::NamedTempFile::new().expect("host temp");
    host.as_file().write_all(&data).expect("write host");

    let stored = device
        .store_file(host.path(), "notes.txt", 0o644)
        .expect("store");
    assert_eq!(stored.filename, "notes.txt");
    assert_eq!(stored.size_bytes, 1024);
    assert_eq!(stored.block_count, 1);

    let listed = device.list_files().expect("list");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].filename, "notes.txt");

    let out_path = tempfile::NamedTempFile::new().expect("out temp");
    device.get_file("notes.txt", out_path.path()).expect("get");
    let readback = std::fs::read(out_path.path()).expect("read out");
    assert_eq!(readback, data, "bytewise roundtrip");
}

#[test]
fn ten_kb_file_spans_three_blocks_and_roundtrips() {
    let (_fixture, mut device) = open_device("file_ops_10kb.gguf", 12);

    let data = patterned_bytes(10 * 1024, 0x42);
    let stored = device.store_bytes(&data, "big.bin", 0o644).expect("store");
    assert_eq!(
        stored.block_count, 3,
        "10KB should span ceil(10240/4096)=3 blocks"
    );
    assert_eq!(stored.inline_blocks.len(), 3);
    assert_eq!(
        stored.overflow_block,
        llmdb::stego::integrity::NO_BLOCK,
        "3 blocks fits inline, no overflow needed"
    );

    let readback = device.read_file_bytes("big.bin").expect("read");
    assert_eq!(readback, data);
}

#[test]
fn delete_returns_all_data_blocks_to_free_list() {
    let (_fixture, mut device) = open_device("file_ops_delete.gguf", 12);

    let used_pre = device.used_blocks().expect("used pre");

    let data = patterned_bytes(9000, 0x55);
    device
        .store_bytes(&data, "deleteme.bin", 0o644)
        .expect("store");
    let used_post_store = device.used_blocks().expect("used post store");
    assert!(used_post_store > used_pre);

    device.delete_file("deleteme.bin").expect("delete");
    let used_post_delete = device.used_blocks().expect("used post delete");
    assert_eq!(
        used_post_delete, used_pre,
        "delete should return all data blocks to the free list"
    );

    let live = device.list_files().expect("list after delete");
    assert!(
        live.iter().all(|e| e.filename != "deleteme.bin"),
        "tombstoned entry should not appear in list_files"
    );
}

#[test]
fn delete_releases_name_for_reuse() {
    // Regression: tombstones used to shadow the name in find_entry_by_name,
    // so delete-then-recreate with the same name failed with DuplicateName.
    // A shell user deleting `notes.txt` and then re-storing `notes.txt`
    // hits this — no reason to keep the name reserved past the delete.
    let (_fixture, mut device) = open_device("file_ops_dupe.gguf", 12);

    let data_a = patterned_bytes(256, 0xA1);
    device
        .store_bytes(&data_a, "once.txt", 0o644)
        .expect("first store");
    device.delete_file("once.txt").expect("delete");

    let data_b = patterned_bytes(256, 0xB2);
    device
        .store_bytes(&data_b, "once.txt", 0o644)
        .expect("second store after delete must succeed");

    let read_back = device.read_file_bytes("once.txt").expect("read");
    assert_eq!(read_back, data_b, "recreated file must carry new contents");
}

#[test]
fn live_duplicate_is_still_rejected() {
    let (_fixture, mut device) = open_device("file_ops_live_dupe.gguf", 12);
    let data = patterned_bytes(256, 0xA1);
    device
        .store_bytes(&data, "live.txt", 0o644)
        .expect("first store");
    let result = device.store_bytes(&data, "live.txt", 0o644);
    assert!(
        matches!(result, Err(FsError::DuplicateName(ref n)) if n == "live.txt"),
        "overwriting a live file without delete must still fail, got: {result:?}"
    );
}

#[test]
fn churning_deletes_reuses_slots_instead_of_leaking() {
    // File table capacity is bounded; if delete+recreate leaked a slot
    // per cycle, a long-running workload (ext4 journal, CI log rotation)
    // would exhaust the table. Verify reuse by churning well past the
    // per-block slot count.
    let (_fixture, mut device) = open_device("file_ops_churn.gguf", 12);

    let data = patterned_bytes(128, 0x33);
    for i in 0..64 {
        let name = format!("slot_churn_{}.bin", i % 4);
        if device.read_file_bytes(&name).is_ok() {
            device.delete_file(&name).expect("delete churn");
        }
        device
            .store_bytes(&data, &name, 0o644)
            .expect("store churn");
    }

    // At the end there are at most 4 live files.
    let live = device.list_files().expect("list");
    assert!(
        live.len() <= 4,
        "expected ≤4 live files after churn, got {}",
        live.len()
    );
}

#[test]
fn get_file_with_unknown_name_returns_not_found() {
    let (_fixture, device) = open_device("file_ops_missing.gguf", 12);

    let out_path = tempfile::NamedTempFile::new().expect("out temp");
    let result = device.get_file("ghost.txt", out_path.path());
    assert!(
        matches!(result, Err(FsError::FileNotFound(ref n)) if n == "ghost.txt"),
        "expected FileNotFound, got: {result:?}"
    );
}

#[test]
fn list_files_skips_tombstones_and_free_slots() {
    let (_fixture, mut device) = open_device("file_ops_list.gguf", 14);

    device
        .store_bytes(&patterned_bytes(100, 0x01), "a.txt", 0o644)
        .expect("store a");
    device
        .store_bytes(&patterned_bytes(100, 0x02), "b.txt", 0o644)
        .expect("store b");
    device
        .store_bytes(&patterned_bytes(100, 0x03), "c.txt", 0o644)
        .expect("store c");

    device.delete_file("b.txt").expect("delete b");

    let live = device.list_files().expect("list");
    let names: Vec<_> = live.iter().map(|e| e.filename.clone()).collect();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"a.txt".to_string()));
    assert!(names.contains(&"c.txt".to_string()));
}

#[test]
fn store_beyond_free_capacity_returns_file_too_large_without_partial_alloc() {
    let (_fixture, mut device) = open_device("file_ops_toolarge.gguf", 12);

    let used_pre = device.used_blocks().expect("used pre");

    // 12 tensors → 8 data blocks; a 40 KB file needs 10 blocks.
    let data = vec![0xCC_u8; 40 * 1024];
    let result = device.store_bytes(&data, "too_big.bin", 0o644);
    assert!(
        matches!(result, Err(FsError::FileTooLarge { .. })),
        "expected FileTooLarge, got: {result:?}"
    );

    let used_post = device.used_blocks().expect("used post");
    assert_eq!(
        used_pre, used_post,
        "failed store must not leave any blocks allocated"
    );
}

#[test]
fn corrupted_file_block_triggers_crc_mismatch_on_get() {
    // Store a file, then corrupt one of its logical data blocks via
    // write_block so the end-to-end CRC fails.
    let (_fixture, mut device) = open_device("file_ops_crc.gguf", 12);

    let data = patterned_bytes(4096 * 2, 0xEE);
    let entry = device
        .store_bytes(&data, "target.bin", 0o644)
        .expect("store");

    // Tamper with the first data block's bytes (via the public write path).
    let victim = entry.inline_blocks[0];
    let tampered = vec![0_u8; llmdb::BLOCK_SIZE];
    device.write_block(victim, &tampered).expect("tamper");

    let result = device.read_file_bytes("target.bin");
    assert!(
        matches!(result, Err(FsError::Crc32Mismatch { .. })),
        "expected Crc32Mismatch, got: {result:?}"
    );
}

#[test]
fn invalid_filenames_are_rejected() {
    let (_fixture, mut device) = open_device("file_ops_badname.gguf", 12);

    let data = vec![0; 16];

    let empty = device.store_bytes(&data, "", 0o644);
    assert!(matches!(empty, Err(FsError::InvalidFilename { .. })));

    let too_long = "x".repeat(96);
    let too_long_result = device.store_bytes(&data, &too_long, 0o644);
    assert!(matches!(
        too_long_result,
        Err(FsError::InvalidFilename { .. })
    ));

    let with_slash = device.store_bytes(&data, "a/b.txt", 0o644);
    assert!(matches!(with_slash, Err(FsError::InvalidFilename { .. })));

    let with_null = device.store_bytes(&data, "a\0b.txt", 0o644);
    assert!(matches!(with_null, Err(FsError::InvalidFilename { .. })));
}

#[test]
fn store_more_than_15_files_extends_chain_and_roundtrips() {
    // 15 slots per file-table block (slot 15 is reserved for the chain
    // pointer). Storing the 16th file must allocate a new chain block,
    // which comes from the data region via `alloc_metadata_physical`.
    // Needs enough data-region blocks to host 20 one-block files AND one
    // chain block: 20 + 1 + metadata = ~25. Fixture: 32 tensors → 32 blocks
    // → 28 data blocks available.
    let (_fixture, mut device) = open_device("file_ops_chain.gguf", 32);

    for i in 0..20 {
        let name = format!("file_{i:02}.txt");
        let payload = patterned_bytes(16, i as u8);
        device.store_bytes(&payload, &name, 0o644).expect("store");
    }

    let listed = device.list_files().expect("list");
    assert_eq!(listed.len(), 20);

    // Roundtrip every file to confirm the chain walk covers both blocks.
    for i in 0..20 {
        let name = format!("file_{i:02}.txt");
        let got = device.read_file_bytes(&name).expect("read");
        let expected = patterned_bytes(16, i as u8);
        assert_eq!(got, expected, "file {name} mismatch");
    }
}

#[test]
fn chain_survives_close_and_reopen() {
    let fixture = write_custom_gguf_fixture(
        SyntheticGgufVersion::V3,
        "chain_reopen.gguf",
        &make_q8_tensors(32),
    );

    {
        let mut device = StegoDevice::initialize_with_options(
            &fixture.path,
            AllocationMode::Standard,
            DeviceOptions { verbose: false },
        )
        .expect("init");
        for i in 0..18 {
            let name = format!("persist_{i:02}.bin");
            device
                .store_bytes(&patterned_bytes(16, i as u8 + 0x80), &name, 0o644)
                .expect("store");
        }
        device.close().expect("close");
    }

    let device = StegoDevice::open_with_options(
        &fixture.path,
        AllocationMode::Standard,
        DeviceOptions { verbose: false },
    )
    .expect("reopen");

    let listed = device.list_files().expect("list after reopen");
    assert_eq!(
        listed.len(),
        18,
        "chain walk must pick up all entries across extended blocks"
    );
    for i in 0..18 {
        let name = format!("persist_{i:02}.bin");
        let got = device.read_file_bytes(&name).expect("read after reopen");
        assert_eq!(got, patterned_bytes(16, i as u8 + 0x80));
    }
}

#[test]
fn unclean_shutdown_preserves_chain_blocks_from_orphan_scan() {
    // Without the recover() fix, the orphan scan would treat chain blocks
    // (which live in the data region, not in redirection) as unreferenced
    // and reclaim them on every dirty reopen. Verify by forcing a dirty
    // shutdown after chain extension.
    let fixture = write_custom_gguf_fixture(
        SyntheticGgufVersion::V3,
        "chain_recovery.gguf",
        &make_q8_tensors(32),
    );

    let file_count = 18;
    {
        let mut device = StegoDevice::initialize_with_options(
            &fixture.path,
            AllocationMode::Standard,
            DeviceOptions { verbose: false },
        )
        .expect("init");
        for i in 0..file_count {
            let name = format!("crash_{i:02}.bin");
            device
                .store_bytes(&patterned_bytes(16, i as u8), &name, 0o644)
                .expect("store");
        }
        // Simulate a crash: don't call close, skip Drop by forgetting.
        std::mem::forget(device);
    }

    let device = StegoDevice::open_with_options(
        &fixture.path,
        AllocationMode::Standard,
        DeviceOptions { verbose: false },
    )
    .expect("reopen after dirty shutdown");

    let listed = device.list_files().expect("list after recovery");
    assert_eq!(
        listed.len(),
        file_count,
        "recovery must keep chain blocks out of the orphan set"
    );
    for i in 0..file_count {
        let name = format!("crash_{i:02}.bin");
        let got = device
            .read_file_bytes(&name)
            .unwrap_or_else(|e| panic!("read {name} after recovery: {e}"));
        assert_eq!(got, patterned_bytes(16, i as u8));
    }
}

#[test]
fn recovery_preserves_live_entry_overflow_blocks() {
    // The pre-chaining recover() didn't mark overflow blocks as in-use
    // either (a latent bug spec §13 listed). Verify they survive a dirty
    // reopen now.
    let fixture = write_custom_gguf_fixture(
        SyntheticGgufVersion::V3,
        "overflow_recovery.gguf",
        &make_q8_tensors(48),
    );

    // Store a file that needs an overflow block (30 blocks of data > 28
    // inline capacity → 1 overflow block).
    let data_len = 30 * 4096 - 1;
    let payload = patterned_bytes(data_len, 0x33);
    {
        let mut device = StegoDevice::initialize_with_options(
            &fixture.path,
            AllocationMode::Standard,
            DeviceOptions { verbose: false },
        )
        .expect("init");
        let entry = device
            .store_bytes(&payload, "needs_overflow.bin", 0o644)
            .expect("store");
        assert_ne!(entry.overflow_block, llmdb::stego::integrity::NO_BLOCK);
        std::mem::forget(device);
    }

    let device = StegoDevice::open_with_options(
        &fixture.path,
        AllocationMode::Standard,
        DeviceOptions { verbose: false },
    )
    .expect("reopen after dirty shutdown");

    let got = device
        .read_file_bytes("needs_overflow.bin")
        .expect("read after recovery — overflow block must survive orphan scan");
    assert_eq!(got, payload);
}

#[test]
fn large_file_uses_overflow_block_and_roundtrips() {
    // Need > 28 data blocks to exercise overflow. 40 tensors → 40 raw blocks,
    // of which 4 are metadata → 36 data blocks. A file of 30 × 4096 = ~123 KB
    // needs 30 data blocks + 1 overflow block = 31 blocks (fits).
    let (_fixture, mut device) = open_device("file_ops_overflow.gguf", 40);

    let data_len = 30 * 4096 - 17;
    let data = patterned_bytes(data_len, 0x99);
    let entry = device
        .store_bytes(&data, "overflow.bin", 0o644)
        .expect("store overflow");
    assert_eq!(entry.block_count, 30);
    assert_eq!(entry.inline_blocks.len(), 28);
    assert_ne!(entry.overflow_block, llmdb::stego::integrity::NO_BLOCK);

    let readback = device
        .read_file_bytes("overflow.bin")
        .expect("read overflow");
    assert_eq!(readback, data);
}
