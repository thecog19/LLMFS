//! End-to-end file storage: `store_bytes` → `read_file_bytes` →
//! `delete_file` for 1 KB / 100 KB / 1 MB payloads. Exercises the
//! file-table + redirection + shadow-copy path as a unit.

mod common;

use common::{q8_tensors, write_fixture_v3};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use llmdb::stego::device::StegoDevice;
use llmdb::stego::planner::AllocationMode;

const SIZES: &[(usize, &str)] = &[
    (1024, "1KiB"),
    (100 * 1024, "100KiB"),
    (1024 * 1024, "1MiB"),
];

fn bench_file_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("file_roundtrip");

    for &(size, label) in SIZES {
        // Size the fixture so the largest file + overhead + overwrite
        // headroom fits comfortably. 512 tensors × 16384 Q8_0 ≈ 1024
        // stego blocks ≈ 4 MiB usable.
        let fixture = write_fixture_v3(
            &format!("bench_file_{label}.gguf"),
            &q8_tensors(512, 16_384),
        );
        let mut device =
            StegoDevice::initialize(&fixture.path, AllocationMode::Standard).expect("init");
        let payload = vec![0xA5_u8; size];

        group.throughput(Throughput::Bytes(size as u64));
        let mut counter: u64 = 0;
        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter(|| {
                // Tombstones hold onto their filename slot, so each
                // iter uses a unique name — keeps the bench honest
                // about store+read+delete cost without tripping the
                // duplicate-name guard.
                counter += 1;
                let name = format!("b{counter}.bin");
                device.store_bytes(&payload, &name, 0o644).expect("store");
                let read_back = device.read_file_bytes(&name).expect("read");
                std::hint::black_box(&read_back);
                device.delete_file(&name).expect("delete");
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_file_roundtrip);
criterion_main!(benches);
