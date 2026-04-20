//! Tests for the generation counter and shadow-copy intent fields on the
//! on-disk superblock. Pairs with `tests/shadow_copy.rs`, which covers
//! intent-based recovery semantics; this file focuses on what the write
//! path writes (observable via `peek_superblock_on_disk`).

mod common;

use common::{SyntheticGgufVersion, write_custom_gguf_fixture};

use llmdb::stego::device::{CrashPoint, DeviceOptions, StegoDevice};
use llmdb::stego::integrity::NO_BLOCK;
use llmdb::stego::planner::AllocationMode;

fn verbose() -> DeviceOptions {
    DeviceOptions { verbose: true }
}

/// 12 tensors × 8192 Q8_0 weights = 12 blocks.
fn make_fixture(name: &str) -> common::FixtureHandle {
    write_custom_gguf_fixture(SyntheticGgufVersion::V3, name, &make_q8_tensors(12))
}

// ============================================================================
// R2 — Generation counter
// ============================================================================

/// Each persist of the superblock bumps the on-disk generation by at least 1.
/// We verify by comparing the on-disk generation across opens separated by
/// a single explicit state change.
#[test]
fn generation_increments_on_each_persist() {
    let fixture = make_fixture("gen_increment.gguf");

    // Fresh init: format() sets gen=0 in memory, then persist bumps to 1.
    // open_with_options() then sets dirty=true and persists again → 2.
    let gen_after_init = {
        let device = StegoDevice::initialize_with_options(
            &fixture.path,
            AllocationMode::Standard,
            verbose(),
        )
        .expect("init");
        let g = device.superblock().fields.generation;
        device.close().expect("close"); // close bumps again
        g
    };
    assert!(
        gen_after_init >= 1,
        "expected at least one bump after initialize, got {gen_after_init}"
    );

    // Peek after close.
    let after_close = StegoDevice::peek_superblock_on_disk(&fixture.path, AllocationMode::Standard)
        .expect("peek after close");
    assert!(
        after_close.fields.generation > gen_after_init,
        "close must persist a generation strictly higher than the in-memory \
         value seen while open: open={}, on-disk={}",
        gen_after_init,
        after_close.fields.generation
    );

    // Open + close again: at least two more bumps (open sets dirty, close clears).
    let prev = after_close.fields.generation;
    {
        let device =
            StegoDevice::open_with_options(&fixture.path, AllocationMode::Standard, verbose())
                .expect("reopen");
        device.close().expect("close 2");
    }
    let after_reopen_close =
        StegoDevice::peek_superblock_on_disk(&fixture.path, AllocationMode::Standard)
            .expect("peek 2");
    assert!(
        after_reopen_close.fields.generation >= prev + 2,
        "open+close pair must produce at least two bumps: prev={}, now={}",
        prev,
        after_reopen_close.fields.generation
    );
}

/// N block writes produce a generation advance of at least N. We're liberal
/// with "at least" because bookkeeping (redirection persist, dirty toggles,
/// free-list push) each persist the superblock too.
#[test]
fn n_writes_advance_generation_by_at_least_n() {
    let fixture = make_fixture("gen_writes.gguf");
    const N: usize = 5;

    let mut device =
        StegoDevice::initialize_with_options(&fixture.path, AllocationMode::Standard, verbose())
            .expect("init");

    let gen_before_writes = device.superblock().fields.generation;

    let block = device.alloc_block().expect("alloc");
    for i in 0..N {
        device
            .write_block(block, &vec![i as u8; llmdb::BLOCK_SIZE])
            .expect("write");
    }

    let gen_after_writes = device.superblock().fields.generation;
    assert!(
        gen_after_writes >= gen_before_writes + N as u64,
        "expected at least +{N} after {N} writes: before={gen_before_writes}, after={gen_after_writes}"
    );

    device.close().expect("close");
}

/// Generation survives close → reopen: the on-disk value is the source of
/// truth, and a fresh open starts from that value (not from zero).
#[test]
fn generation_survives_close_and_reopen() {
    let fixture = make_fixture("gen_persist.gguf");

    let gen_after_first_session = {
        let mut device = StegoDevice::initialize_with_options(
            &fixture.path,
            AllocationMode::Standard,
            verbose(),
        )
        .expect("init");

        let block = device.alloc_block().expect("alloc");
        device
            .write_block(block, &vec![0xAA; llmdb::BLOCK_SIZE])
            .expect("write");

        device.close().expect("close");
        StegoDevice::peek_superblock_on_disk(&fixture.path, AllocationMode::Standard)
            .expect("peek after session 1")
            .fields
            .generation
    };
    assert!(gen_after_first_session > 0, "gen must have advanced");

    // Reopen: in-memory gen starts at on-disk gen, then open_with_options
    // bumps dirty=true, so we expect >= first-session + 1 in memory.
    let device = StegoDevice::open_with_options(&fixture.path, AllocationMode::Standard, verbose())
        .expect("reopen");
    assert!(
        device.superblock().fields.generation > gen_after_first_session,
        "reopen must see at least one bump beyond the on-disk value: \
         on-disk={}, in-memory after open={}",
        gen_after_first_session,
        device.superblock().fields.generation
    );
    device.close().expect("close 2");
}

// ============================================================================
// R3 — Shadow intent fields
// ============================================================================

/// After a normal shadow-copy completes, the on-disk superblock must have
/// shadow_block and shadow_target reset to NO_BLOCK. No leftover intent.
#[test]
fn normal_shadow_copy_clears_intent_on_completion() {
    let fixture = make_fixture("intent_clean.gguf");

    let mut device =
        StegoDevice::initialize_with_options(&fixture.path, AllocationMode::Standard, verbose())
            .expect("init");

    let block = device.alloc_block().expect("alloc");
    device
        .write_block(block, &vec![0xA1; llmdb::BLOCK_SIZE])
        .expect("first write");
    device
        .write_block(block, &vec![0xB2; llmdb::BLOCK_SIZE])
        .expect("overwrite (shadow-copy)");

    device.close().expect("close");

    let on_disk = StegoDevice::peek_superblock_on_disk(&fixture.path, AllocationMode::Standard)
        .expect("peek");
    assert_eq!(
        on_disk.fields.shadow_block, NO_BLOCK,
        "shadow_block must be cleared after normal completion"
    );
    assert_eq!(
        on_disk.fields.shadow_target, NO_BLOCK,
        "shadow_target must be cleared after normal completion"
    );
}

/// Crash at AfterShadowFlush — shadow data durable, redirection not flipped.
/// On-disk superblock must record the in-flight shadow as (S, L) so recovery
/// can discriminate by intent.
#[test]
fn crash_after_shadow_flush_preserves_intent_on_disk() {
    let fixture = make_fixture("intent_phase1.gguf");

    let block;
    let shadow_head_at_crash;

    {
        let mut device = StegoDevice::initialize_with_options(
            &fixture.path,
            AllocationMode::Standard,
            verbose(),
        )
        .expect("init");

        block = device.alloc_block().expect("alloc");
        device
            .write_block(block, &vec![0xAA; llmdb::BLOCK_SIZE])
            .expect("first write");

        // Peek at the free-list head: the shadow pop will take this physical.
        shadow_head_at_crash = device.superblock().fields.free_list_head;
        assert_ne!(shadow_head_at_crash, NO_BLOCK);

        device
            .write_block_with_crash_after(
                block,
                &vec![0xBB; llmdb::BLOCK_SIZE],
                CrashPoint::AfterShadowFlush,
            )
            .expect("crash at phase 1");

        std::mem::forget(device);
    }

    let on_disk = StegoDevice::peek_superblock_on_disk(&fixture.path, AllocationMode::Standard)
        .expect("peek after crash");
    assert_eq!(
        on_disk.fields.shadow_block, shadow_head_at_crash,
        "on-disk shadow_block must equal the physical that was popped"
    );
    assert_eq!(
        on_disk.fields.shadow_target, block,
        "on-disk shadow_target must equal the logical being shadowed"
    );
    assert!(
        on_disk.is_dirty(),
        "dirty flag must still be set after a crash mid-write"
    );
}

/// Crash at AfterRedirectionFlush — shadow installed, old physical not yet
/// freed, intent not yet cleared. On-disk superblock must still carry the
/// intent so recovery knows which side the commit landed on.
#[test]
fn crash_after_redirection_flush_preserves_intent_on_disk() {
    let fixture = make_fixture("intent_phase2.gguf");

    let block;
    let shadow_head_at_crash;

    {
        let mut device = StegoDevice::initialize_with_options(
            &fixture.path,
            AllocationMode::Standard,
            verbose(),
        )
        .expect("init");

        block = device.alloc_block().expect("alloc");
        device
            .write_block(block, &vec![0xCC; llmdb::BLOCK_SIZE])
            .expect("first write");

        shadow_head_at_crash = device.superblock().fields.free_list_head;
        assert_ne!(shadow_head_at_crash, NO_BLOCK);

        device
            .write_block_with_crash_after(
                block,
                &vec![0xDD; llmdb::BLOCK_SIZE],
                CrashPoint::AfterRedirectionFlush,
            )
            .expect("crash at phase 2");

        std::mem::forget(device);
    }

    let on_disk = StegoDevice::peek_superblock_on_disk(&fixture.path, AllocationMode::Standard)
        .expect("peek after crash");
    assert_eq!(
        on_disk.fields.shadow_block, shadow_head_at_crash,
        "intent must remain on disk across phase 2 until phase 3 clears it"
    );
    assert_eq!(on_disk.fields.shadow_target, block);
}

/// First write (unmapped logical → fresh physical) is NOT a shadow-copy:
/// no old data to preserve. The intent fields must not be touched.
#[test]
fn first_write_does_not_set_shadow_intent() {
    let fixture = make_fixture("intent_first_write.gguf");

    let block;
    {
        let mut device = StegoDevice::initialize_with_options(
            &fixture.path,
            AllocationMode::Standard,
            verbose(),
        )
        .expect("init");

        block = device.alloc_block().expect("alloc");
        device
            .write_block(block, &vec![0xEE; llmdb::BLOCK_SIZE])
            .expect("first write");

        // First write path is NOT shadow-copy. On-disk intent must be clean.
        std::mem::forget(device);
    }

    let on_disk = StegoDevice::peek_superblock_on_disk(&fixture.path, AllocationMode::Standard)
        .expect("peek");
    assert_eq!(
        on_disk.fields.shadow_block, NO_BLOCK,
        "first-write path must not set shadow_block"
    );
    assert_eq!(
        on_disk.fields.shadow_target, NO_BLOCK,
        "first-write path must not set shadow_target"
    );
}

// ============================================================================
// R4 — Intent-based recovery
// ============================================================================

/// Recovery must clear the intent fields on disk when it finishes, regardless
/// of which side of the flip the crash landed on.
#[test]
fn recovery_clears_intent_on_disk() {
    let fixture = make_fixture("intent_cleared_by_recovery.gguf");

    let block;
    {
        let mut device = StegoDevice::initialize_with_options(
            &fixture.path,
            AllocationMode::Standard,
            verbose(),
        )
        .expect("init");

        block = device.alloc_block().expect("alloc");
        device
            .write_block(block, &vec![0x10; llmdb::BLOCK_SIZE])
            .expect("first write");

        device
            .write_block_with_crash_after(
                block,
                &vec![0x20; llmdb::BLOCK_SIZE],
                CrashPoint::AfterShadowFlush,
            )
            .expect("crash phase 1");

        std::mem::forget(device);
    }

    // Sanity: intent was set on disk immediately after the crash.
    let pre_recovery =
        StegoDevice::peek_superblock_on_disk(&fixture.path, AllocationMode::Standard)
            .expect("peek pre-recovery");
    assert_ne!(pre_recovery.fields.shadow_block, NO_BLOCK);
    assert_ne!(pre_recovery.fields.shadow_target, NO_BLOCK);

    // Reopen → recovery runs → clears intent + dirty, then close cleanly.
    {
        let device =
            StegoDevice::open_with_options(&fixture.path, AllocationMode::Standard, verbose())
                .expect("reopen");
        device.close().expect("close");
    }

    let post_recovery =
        StegoDevice::peek_superblock_on_disk(&fixture.path, AllocationMode::Standard)
            .expect("peek post-recovery");
    assert_eq!(
        post_recovery.fields.shadow_block, NO_BLOCK,
        "recovery must clear shadow_block on disk"
    );
    assert_eq!(
        post_recovery.fields.shadow_target, NO_BLOCK,
        "recovery must clear shadow_target on disk"
    );
    assert!(
        !post_recovery.is_dirty(),
        "close after recovery must leave dirty=false on disk"
    );
}

/// Crash after phase 2 (redirection flipped, intent still set on disk).
/// Recovery must see intent=(S,L) and redirection[L]=S → conclude committed
/// → reclaim the old physical → new value visible on next read.
#[test]
fn recovery_at_phase_2_commits_via_intent_check() {
    let fixture = make_fixture("intent_recovery_phase2.gguf");

    let block;
    let used_after_first_write;
    {
        let mut device = StegoDevice::initialize_with_options(
            &fixture.path,
            AllocationMode::Standard,
            verbose(),
        )
        .expect("init");

        block = device.alloc_block().expect("alloc");
        device
            .write_block(block, &vec![0x30; llmdb::BLOCK_SIZE])
            .expect("first write");
        used_after_first_write = device.used_blocks().expect("used");

        device
            .write_block_with_crash_after(
                block,
                &vec![0x40; llmdb::BLOCK_SIZE],
                CrashPoint::AfterRedirectionFlush,
            )
            .expect("crash phase 2");

        std::mem::forget(device);
    }

    let device = StegoDevice::open_with_options(&fixture.path, AllocationMode::Standard, verbose())
        .expect("reopen → recovery");

    // The new value is visible (phase 2 committed before crash).
    let read_back = device.read_block(block).expect("read");
    assert_eq!(read_back, vec![0x40; llmdb::BLOCK_SIZE]);

    // The old physical was reclaimed by recovery — used count is back to
    // pre-shadow-copy baseline (one block used, which is the shadow that
    // now holds the committed data).
    assert_eq!(
        device.used_blocks().expect("used"),
        used_after_first_write,
        "recovery must reclaim the orphaned old physical"
    );
}

/// Crash before phase 2 (shadow written, redirection not flipped).
/// Recovery must see intent=(S,L) and redirection[L]≠S → conclude aborted
/// → reclaim the shadow → old value still visible.
#[test]
fn recovery_at_phase_1_aborts_via_intent_check() {
    let fixture = make_fixture("intent_recovery_phase1.gguf");

    let block;
    let used_after_first_write;
    {
        let mut device = StegoDevice::initialize_with_options(
            &fixture.path,
            AllocationMode::Standard,
            verbose(),
        )
        .expect("init");

        block = device.alloc_block().expect("alloc");
        device
            .write_block(block, &vec![0x50; llmdb::BLOCK_SIZE])
            .expect("first write");
        used_after_first_write = device.used_blocks().expect("used");

        device
            .write_block_with_crash_after(
                block,
                &vec![0x60; llmdb::BLOCK_SIZE],
                CrashPoint::AfterShadowFlush,
            )
            .expect("crash phase 1");

        std::mem::forget(device);
    }

    let device = StegoDevice::open_with_options(&fixture.path, AllocationMode::Standard, verbose())
        .expect("reopen → recovery");

    // Old value still canonical — phase 2 never landed.
    let read_back = device.read_block(block).expect("read");
    assert_eq!(read_back, vec![0x50; llmdb::BLOCK_SIZE]);

    // Used count reverts to pre-shadow: shadow was reclaimed.
    assert_eq!(
        device.used_blocks().expect("used"),
        used_after_first_write,
        "recovery must reclaim the orphaned shadow"
    );
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
