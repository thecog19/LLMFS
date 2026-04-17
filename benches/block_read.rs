//! Block-level read throughput. Warms the device with a known payload
//! pattern, then times random-index `read_block` calls. Reports
//! blocks/s and MB/s via `Throughput::Bytes`.

mod common;

use common::{FixtureHandle, Xorshift, q8_tensors, write_fixture_v3};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use llmdb::stego::device::StegoDevice;
use llmdb::stego::planner::AllocationMode;

fn prepare_device(fixture: &FixtureHandle, fill_ratio_pct: u32) -> (StegoDevice, Vec<u32>) {
    let mut device =
        StegoDevice::initialize(&fixture.path, AllocationMode::Standard).expect("init");
    let total = device.total_blocks();
    let data_start = device.data_region_start();
    let data_blocks = total - data_start;
    let to_fill = (data_blocks * fill_ratio_pct / 100) as usize;

    let mut logicals = Vec::with_capacity(to_fill);
    for i in 0..to_fill {
        let logical = device.alloc_block().expect("alloc");
        let payload = vec![(i as u8).wrapping_add(1); llmdb::BLOCK_SIZE];
        device.write_block(logical, &payload).expect("write");
        logicals.push(logical);
    }
    device.flush().expect("flush");
    (device, logicals)
}

fn bench_block_read(c: &mut Criterion) {
    // 64 Q8_0 tensors × 8192 weights = 64 × 4096 stego bytes = 64 KiB →
    // 16 data blocks after metadata. We want enough room to bench
    // random reads across a nontrivial working set, so bump to 128
    // tensors × 16384 weights = ~512 stego blocks → big enough.
    let fixture = write_fixture_v3("bench_read.gguf", &q8_tensors(128, 16_384));
    let (device, logicals) = prepare_device(&fixture, 50);

    let mut group = c.benchmark_group("block_read");
    group.throughput(Throughput::Bytes(llmdb::BLOCK_SIZE as u64));

    let mut rng = Xorshift(0x9E37_79B9_7F4A_7C15);
    let handle_count = logicals.len() as u32;
    group.bench_function(BenchmarkId::from_parameter("4KiB_random"), |b| {
        b.iter(|| {
            let idx = logicals[rng.next_range(handle_count) as usize];
            let block = device.read_block(idx).expect("read");
            std::hint::black_box(block);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_block_read);
criterion_main!(benches);
