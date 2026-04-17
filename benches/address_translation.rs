//! The per-byte logical→physical walk. DESIGN-NEW §14 flags this as
//! the top optimization candidate — it runs once per stego byte.

mod common;

use common::{q8_tensors, write_fixture_v3};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use llmdb::gguf::parser::parse_path;
use llmdb::stego::planner::{AllocationMode, build_allocation_plan};
use llmdb::stego::tensor_map::TensorMap;

fn bench_address_translation(c: &mut Criterion) {
    let fixture = write_fixture_v3("bench_addr.gguf", &q8_tensors(128, 16_384));
    let parsed = parse_path(&fixture.path).expect("parse");
    let plan = build_allocation_plan(&parsed.tensors, AllocationMode::Standard);
    let map = TensorMap::from_allocation_plan(&plan);

    let total = map.total_capacity_bytes;
    assert!(total > 0);

    let mut group = c.benchmark_group("address_translation");
    group.throughput(Throughput::Elements(1));

    // Stepping by a prime ensures we touch different tensor slots
    // instead of hitting the same cache line every iter.
    let stride = 104_729_u64;
    let mut cursor = 0_u64;
    group.bench_function(BenchmarkId::from_parameter("map_logical_byte"), |b| {
        b.iter(|| {
            let idx = cursor % total;
            cursor = cursor.wrapping_add(stride);
            let mapping = map.map_logical_byte(idx).expect("map");
            std::hint::black_box(mapping);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_address_translation);
criterion_main!(benches);
