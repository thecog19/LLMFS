//! Block-level write throughput. Overwrites random pre-allocated
//! logicals to exercise the shadow-copy path (which is what an NBD
//! client actually hits; a fresh allocate-first-write is strictly
//! cheaper than an overwrite, so we bench the hotter case).

mod common;

use common::{Xorshift, q8_tensors, write_fixture_v3};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use llmdb::stego::device::StegoDevice;
use llmdb::stego::planner::AllocationMode;

fn bench_block_write(c: &mut Criterion) {
    let fixture = write_fixture_v3("bench_write.gguf", &q8_tensors(128, 16_384));
    let mut device =
        StegoDevice::initialize(&fixture.path, AllocationMode::Standard).expect("init");

    let total = device.total_blocks();
    let data_start = device.data_region_start();
    let data_blocks = total - data_start;
    // Reserve half the data blocks as live logicals we'll overwrite —
    // the other half stays free so the shadow-copy always has a target.
    let live = (data_blocks / 2) as usize;
    let mut logicals = Vec::with_capacity(live);
    for i in 0..live {
        let logical = device.alloc_block().expect("alloc");
        let payload = vec![(i as u8).wrapping_add(1); llmdb::BLOCK_SIZE];
        device.write_block(logical, &payload).expect("prime");
        logicals.push(logical);
    }
    device.flush().expect("flush");

    let mut group = c.benchmark_group("block_write");
    group.throughput(Throughput::Bytes(llmdb::BLOCK_SIZE as u64));

    let mut rng = Xorshift(0xC3A5_C85C_97CB_3127);
    let handle_count = logicals.len() as u32;
    let payload = vec![0xA5_u8; llmdb::BLOCK_SIZE];
    group.bench_function(BenchmarkId::from_parameter("4KiB_overwrite"), |b| {
        b.iter(|| {
            let idx = logicals[rng.next_range(handle_count) as usize];
            device.write_block(idx, &payload).expect("write");
        });
    });
    group.finish();
}

criterion_group!(benches, bench_block_write);
criterion_main!(benches);
