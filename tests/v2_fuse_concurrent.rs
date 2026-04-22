//! V2 FUSE concurrent-dispatch tests.
//!
//! Verifies that multiple in-flight FUSE requests run in parallel
//! rather than head-of-line blocking on a single dispatch thread.
//! The driver achieves this by spawning each request's work onto a
//! worker thread and returning the synchronous fuser handler
//! immediately — fuser 0.15's dispatch loop is itself serial, so
//! parallelism has to come from inside our handlers.
//!
//! Skipped gracefully if `/dev/fuse` is missing (sandbox / container
//! without FUSE) or `fusermount{3,}` isn't on PATH.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};
use llmdb::v2::cdc::FastCdcParams;
use llmdb::v2::fs::Filesystem;
use llmdb::v2::fuse::{LlmdbV2Fs, MountConfig, spawn_background};

// ─── env probing ───────────────────────────────────────────────────────

fn fuse_available() -> bool {
    if !Path::new("/dev/fuse").exists() {
        return false;
    }
    ["fusermount3", "fusermount"]
        .iter()
        .any(|prog| which(prog).is_some())
}

fn which(prog: &str) -> Option<std::path::PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|p| {
            let candidate = p.join(prog);
            if candidate.is_file() {
                Some(candidate)
            } else {
                None
            }
        })
    })
}

// ─── F16 cover fixture ─────────────────────────────────────────────────

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp32 = ((bits >> 23) & 0xFF) as i32;
    let mantissa32 = bits & 0x7FFFFF;
    if exp32 == 0 {
        return sign << 15;
    }
    let exp16 = exp32 - 127 + 15;
    if exp16 <= 0 {
        return sign << 15;
    }
    if exp16 >= 31 {
        return (sign << 15) | (0x1F << 10);
    }
    let mantissa16 = (mantissa32 >> 13) as u16;
    (sign << 15) | ((exp16 as u16) << 10) | mantissa16
}

/// Cover sized for ~10 MB of stealable storage capacity (F16 → 4
/// stealable bits/weight → 10 MB needs ~20 M weights).
fn make_cover() -> (Vec<u8>, TensorMap) {
    let weight_count = 20_000_000_u64;
    let mut bytes = Vec::with_capacity((weight_count * 2) as usize);
    for i in 0..weight_count {
        let sign = if i % 3 == 0 { -1.0 } else { 1.0 };
        let v = sign * ((i + 1) as f32) * 0.000_001;
        bytes.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
    }
    let bits = GgufQuantType::F16.stealable_bits_hint() as u64;
    let slot = TensorSlot {
        name: "fuse_concurrent.f16".to_owned(),
        quant_type: GgufQuantType::F16,
        tier: TensorTier::Tier1,
        data_offset: 0,
        weight_count,
        stealable_bits_per_weight: bits as usize,
        capacity_bits: weight_count * bits,
        bit_start: 0,
        bit_end: weight_count * bits,
    };
    let map = TensorMap {
        slots: vec![slot.clone()],
        total_capacity_bits: slot.capacity_bits,
        total_capacity_bytes: slot.capacity_bits / 8,
    };
    (bytes, map)
}

fn cdc() -> FastCdcParams {
    FastCdcParams::default()
}

/// Boot a V2 filesystem with two large-ish files at known paths.
/// Returns `None` when FUSE isn't usable.
fn spawn_mount_with_two_files(
    bytes_per_file: usize,
) -> Option<(tempfile::TempDir, fuser::BackgroundSession, Vec<u8>, Vec<u8>)> {
    if !fuse_available() {
        eprintln!("skipping: /dev/fuse or fusermount missing");
        return None;
    }
    let (cover, map) = make_cover();
    let mut fs = Filesystem::init_with_cdc_params(cover, map, cdc()).expect("init v2");

    // Two files with distinct content so dedup doesn't collapse them
    // to the same chunks (which would make the test unrepresentative).
    let payload_a: Vec<u8> = (0..bytes_per_file).map(|i| (i % 251) as u8).collect();
    let payload_b: Vec<u8> = (0..bytes_per_file).map(|i| ((i * 31 + 7) % 251) as u8).collect();
    fs.create_file("/a.bin", &payload_a).expect("create /a.bin");
    fs.create_file("/b.bin", &payload_b).expect("create /b.bin");

    let driver = LlmdbV2Fs::new(fs);
    let mount = tempfile::tempdir().expect("tempdir");
    let session = spawn_background(driver, mount.path(), &MountConfig { allow_other: false })
        .expect("spawn mount");
    thread::sleep(Duration::from_millis(150));
    Some((mount, session, payload_a, payload_b))
}

/// Open a file via std::fs and read the whole thing. Repeats `iters`
/// times, returning total wall time.
fn read_loop(path: &Path, iters: usize, expected_len: usize) -> Duration {
    let start = Instant::now();
    for _ in 0..iters {
        let bytes = std::fs::read(path).expect("read");
        assert_eq!(bytes.len(), expected_len);
    }
    start.elapsed()
}

// ─── Tests ─────────────────────────────────────────────────────────────

/// Two threads each read a separate file in a tight loop. With
/// concurrent dispatch, total wall time should be meaningfully less
/// than running them back-to-back.
///
/// Threshold: parallel < 0.7 × sequential. With perfect 2-way
/// parallelism we'd expect 0.5×; the 0.7× margin tolerates dispatch
/// overhead, kernel scheduling, and CI jitter while still failing
/// loudly if the work serializes.
#[test]
fn two_readers_on_separate_files_run_in_parallel() {
    const BYTES_PER_FILE: usize = 256 * 1024; // 256 KB each
    const ITERS: usize = 60;

    let Some((mount, session, payload_a, payload_b)) = spawn_mount_with_two_files(BYTES_PER_FILE)
    else {
        return;
    };
    let mount_path = mount.path().to_owned();
    let path_a = mount_path.join("a.bin");
    let path_b = mount_path.join("b.bin");

    // Sequential baseline: do both workloads back-to-back on the
    // calling thread.
    let sequential = {
        let t = Instant::now();
        let _ = read_loop(&path_a, ITERS, payload_a.len());
        let _ = read_loop(&path_b, ITERS, payload_b.len());
        t.elapsed()
    };

    // Parallel: same workload on two threads concurrently.
    let parallel = {
        let path_a = path_a.clone();
        let path_b = path_b.clone();
        let len_a = payload_a.len();
        let len_b = payload_b.len();
        let t = Instant::now();
        let h_a = thread::spawn(move || read_loop(&path_a, ITERS, len_a));
        let h_b = thread::spawn(move || read_loop(&path_b, ITERS, len_b));
        h_a.join().expect("join a");
        h_b.join().expect("join b");
        t.elapsed()
    };

    eprintln!(
        "two_readers: sequential={:?}, parallel={:?}, ratio={:.2}",
        sequential,
        parallel,
        parallel.as_secs_f64() / sequential.as_secs_f64()
    );

    drop(session);

    // The strict assertion: parallel must be at least 30 % faster
    // than sequential. Without spawn-and-async this is ~1.0 (no
    // gain).
    let ratio = parallel.as_secs_f64() / sequential.as_secs_f64();
    assert!(
        ratio < 0.7,
        "expected parallel/sequential < 0.7 (real concurrency), \
         got {ratio:.2} (sequential={sequential:?}, parallel={parallel:?})"
    );
}

/// A long-running reader on `/big.bin` must not block a quick
/// `getattr` on `/small.bin`. Without concurrent dispatch the
/// quick op queues behind the slow read. With concurrent
/// dispatch, the quick op completes promptly.
///
/// We approximate "long read" by hammering reads in a loop on one
/// thread and checking that a single `metadata()` on the small
/// file from another thread returns within a small wall-clock
/// budget. The threshold is intentionally generous (200 ms) — we're
/// testing for non-blocking, not hard latency.
#[test]
fn fast_op_does_not_wait_behind_slow_read() {
    const BIG_BYTES: usize = 1024 * 1024; // 1 MB
    const ITERS: usize = 200;

    let Some((mount, session, _payload_a, _payload_b)) = spawn_mount_with_two_files(BIG_BYTES)
    else {
        return;
    };
    let mount_path = mount.path().to_owned();
    let big = mount_path.join("a.bin");
    let small = mount_path.join("b.bin");

    let stop = Arc::new(AtomicBool::new(false));
    let stop_for_reader = Arc::clone(&stop);
    let big_for_reader = big.clone();
    let reader = thread::spawn(move || {
        for _ in 0..ITERS {
            if stop_for_reader.load(Ordering::Relaxed) {
                break;
            }
            let _ = std::fs::read(&big_for_reader).expect("read big");
        }
    });

    // Give the reader a tick to get into its loop.
    thread::sleep(Duration::from_millis(20));

    // Time a single metadata() on the small file. With async
    // dispatch this should land in tens of ms; with serial dispatch
    // it queues behind whatever big read is in flight.
    let probe_start = Instant::now();
    let _ = std::fs::metadata(&small).expect("metadata small");
    let probe = probe_start.elapsed();

    stop.store(true, Ordering::Relaxed);
    reader.join().expect("join reader");
    drop(session);

    eprintln!("fast_op_under_load: probe={probe:?}");
    assert!(
        probe < Duration::from_millis(200),
        "metadata() under concurrent reader took {probe:?} \
         (expected < 200ms with concurrent dispatch)"
    );
}
