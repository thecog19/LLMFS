//! Cross-implementation validation: our K-quant decoders against
//! ggml-encoded blocks. Without this test the K-quant decoders are
//! tested only against synthetic inputs we built from our own reading
//! of the spec — same algorithm in writer and reader, trivially
//! passes, doesn't catch bugs in our spec interpretation.
//!
//! Strategy: compare the dequantized values to the original F16
//! source, asserting per-tensor max and mean errors are within the
//! known representational envelope of the target K-quant. A wrong
//! decoder produces nonsense outside the envelope (we already
//! caught this kind of bug — the magnitude estimator was reading
//! tensor data using relative offsets, producing values 4-5 orders
//! of magnitude off; this test would have caught it instantly).
//!
//! Fixture: `benches/fixtures/k-quant/smollm2-135m-q4_k_m.gguf` is
//! produced by `scripts/regen-kquant-fixture.sh`. It's gitignored
//! (~100 MB). The test skips if the fixture is missing.

use std::fs::File;
use std::path::Path;

use llmdb::gguf::parser::parse_path;
use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::packing::{float, q3_k, q4_k, q5_k, q6_k};
use memmap2::Mmap;

const SRC_F16: &str = "models/pristine/smollm2-135m-f16.gguf";
// Q3_K_M on SmolLM2-135M yields zero Q3_K tensors in practice — the
// model's ffn_down rows pick Q4_K/Q5_K instead because of the mixing
// heuristic. Q3_K_S pushes ffn_down rows to Q3_K, so we use it as the
// reference fixture for Q3_K validation.
const Q3_K_S_FIXTURE: &str = "benches/fixtures/k-quant/smollm2-135m-q3_k_s.gguf";
const Q4_K_M_FIXTURE: &str = "benches/fixtures/k-quant/smollm2-135m-q4_k_m.gguf";
const Q5_K_M_FIXTURE: &str = "benches/fixtures/k-quant/smollm2-135m-q5_k_m.gguf";

/// Q4_K_M ships F16 weights as Q4_K, Q5_K, Q6_K, Q5_0 mixed; Q5_K_M
/// pushes most tensors to Q5_K specifically. Q3_K_M pushes down to
/// Q3_K where possible. Different K-quant types surface in different
/// fixtures, so the validation tests pick the fixture that's most
/// likely to contain the target type.
fn fixture_for(target: GgufQuantType) -> &'static str {
    match target {
        GgufQuantType::Q3K => Q3_K_S_FIXTURE,
        GgufQuantType::Q5K => Q5_K_M_FIXTURE,
        _ => Q4_K_M_FIXTURE,
    }
}

struct ParsedCover {
    parsed: llmdb::gguf::parser::GgufFile,
    mmap: Mmap,
}

fn open_cover(path: &str) -> Option<ParsedCover> {
    if !Path::new(path).exists() {
        return None;
    }
    let parsed = parse_path(path).expect("parse cover");
    let mmap = unsafe { Mmap::map(&File::open(path).expect("open cover")).expect("mmap") };
    Some(ParsedCover { parsed, mmap })
}

#[test]
#[ignore = "needs ~100MB regen fixture; run with --ignored"]
fn q3_k_decoder_matches_ggml_quantization() {
    validate_kquant_against_f16(GgufQuantType::Q3K, q3_k_decode);
}

#[test]
#[ignore = "needs ~100MB regen fixture; run with --ignored"]
fn q4_k_decoder_matches_ggml_quantization() {
    validate_kquant_against_f16(GgufQuantType::Q4K, q4_k_decode);
}

#[test]
#[ignore = "needs ~100MB regen fixture; run with --ignored"]
fn q5_k_decoder_matches_ggml_quantization() {
    validate_kquant_against_f16(GgufQuantType::Q5K, q5_k_decode);
}

#[test]
#[ignore = "needs ~100MB regen fixture; run with --ignored"]
fn q6_k_decoder_matches_ggml_quantization() {
    validate_kquant_against_f16(GgufQuantType::Q6K, q6_k_decode);
}

fn q3_k_decode(block: &[u8], i: usize) -> f32 {
    q3_k::read_weight_value(block, i).expect("q3_k decode")
}

fn q4_k_decode(block: &[u8], i: usize) -> f32 {
    q4_k::read_weight_value(block, i).expect("q4_k decode")
}

fn q5_k_decode(block: &[u8], i: usize) -> f32 {
    q5_k::read_weight_value(block, i).expect("q5_k decode")
}

fn q6_k_decode(block: &[u8], i: usize) -> f32 {
    q6_k::read_weight_value(block, i).expect("q6_k decode")
}

/// Walk every tensor of the requested K-quant type in the fixture,
/// dequantize each weight via the supplied decoder, and assert the
/// per-tensor error against the F16 source falls inside the
/// representational envelope of that K-quant.
fn validate_kquant_against_f16(target: GgufQuantType, decode: fn(&[u8], usize) -> f32) {
    let Some(src) = open_cover(SRC_F16) else {
        eprintln!("skipping: {SRC_F16} not present");
        return;
    };
    let fixture = fixture_for(target);
    let Some(quant) = open_cover(fixture) else {
        eprintln!("skipping: {fixture} not present — run scripts/regen-kquant-fixture.sh");
        return;
    };

    let src_base = src.parsed.tensor_data_offset as u64;
    let quant_base = quant.parsed.tensor_data_offset as u64;

    let block_bytes = match target {
        GgufQuantType::Q3K => q3_k::BLOCK_BYTES,
        GgufQuantType::Q4K => q4_k::BLOCK_BYTES,
        GgufQuantType::Q5K => q5_k::BLOCK_BYTES,
        GgufQuantType::Q6K => q6_k::BLOCK_BYTES,
        _ => unreachable!(),
    };
    let weights_per_block = match target {
        GgufQuantType::Q3K => q3_k::WEIGHTS_PER_BLOCK,
        GgufQuantType::Q4K => q4_k::WEIGHTS_PER_BLOCK,
        GgufQuantType::Q5K => q5_k::WEIGHTS_PER_BLOCK,
        GgufQuantType::Q6K => q6_k::WEIGHTS_PER_BLOCK,
        _ => unreachable!(),
    };

    let mut tensors_checked = 0;
    let mut total_weights = 0_u64;
    let mut total_max_err = 0_f32;
    let mut total_mean_err_sum = 0_f64;

    for q_tensor in &quant.parsed.tensors {
        let tag = GgufQuantType::from_raw_ggml_type(q_tensor.raw_type_id);
        if tag != Some(target) {
            continue;
        }
        let src_tensor = src
            .parsed
            .tensors
            .iter()
            .find(|t| t.name == q_tensor.name)
            .expect("matching source tensor");
        // Source must be F16 — that's the only quant type our F16 cover
        // contains besides F32 norms (which never re-quantize).
        let src_q = GgufQuantType::from_raw_ggml_type(src_tensor.raw_type_id);
        if src_q != Some(GgufQuantType::F16) {
            continue;
        }
        let weight_count = q_tensor.element_count() as usize;
        let n_blocks = weight_count / weights_per_block;
        if n_blocks == 0 {
            continue;
        }

        let q_off = quant_base + q_tensor.data_offset;
        let f16_off = src_base + src_tensor.data_offset;

        let mut max_err = 0_f32;
        let mut sum_err = 0_f64;
        for w in 0..(n_blocks * weights_per_block) {
            let block_idx = w / weights_per_block;
            let in_block = w % weights_per_block;
            let bs = (q_off + (block_idx as u64) * (block_bytes as u64)) as usize;
            let block = &quant.mmap[bs..bs + block_bytes];
            let decoded = decode(block, in_block);

            let off = (f16_off + (w as u64) * 2) as usize;
            let bits = u16::from_le_bytes([src.mmap[off], src.mmap[off + 1]]);
            let f16_val = float::f16_to_f32(bits);

            let err = (decoded - f16_val).abs();
            if err > max_err {
                max_err = err;
            }
            sum_err += err as f64;
        }
        let mean_err = sum_err / (n_blocks * weights_per_block) as f64;
        tensors_checked += 1;
        total_weights += (n_blocks * weights_per_block) as u64;
        if max_err > total_max_err {
            total_max_err = max_err;
        }
        total_mean_err_sum += mean_err * (n_blocks * weights_per_block) as f64;
    }

    assert!(
        tensors_checked > 0,
        "no {target:?} tensors found in fixture"
    );
    let aggregate_mean_err = total_mean_err_sum / total_weights as f64;
    println!(
        "{target:?}: {tensors_checked} tensors, {total_weights} weights, max_err={:.4}, mean_err={:.5}",
        total_max_err, aggregate_mean_err
    );

    // Envelope: K-quants encode 4-6 bits per weight with per-32-or-64
    // sub-block scales. Mean reconstruction error is bounded by ~half a
    // step of the smallest sub-block scale; on trained transformer
    // weights with magnitudes ~0.1 this is roughly 0.005-0.02. We
    // assert generous bounds — the real signal is "is it in the right
    // ballpark?" not "is it pixel-perfect?". A wrong decoder produces
    // values 100×-1000× off, which these bounds would catch instantly.
    let mean_bound = match target {
        // Q3_K has 3 bits/weight so the reconstruction envelope is
        // wider than Q4_K's; empirically mean_err on transformer
        // weights is ~0.024 with max up to ~0.66 due to the hmask
        // high-bit stepping through ±4×scale.
        GgufQuantType::Q3K => 0.05,
        GgufQuantType::Q4K => 0.05,
        GgufQuantType::Q5K => 0.025,
        GgufQuantType::Q6K => 0.012,
        _ => unreachable!(),
    };
    let max_bound = match target {
        GgufQuantType::Q3K => 1.0,
        GgufQuantType::Q4K => 0.5,
        GgufQuantType::Q5K => 0.3,
        GgufQuantType::Q6K => 0.2,
        _ => unreachable!(),
    };
    assert!(
        aggregate_mean_err <= mean_bound,
        "{target:?} mean_err {aggregate_mean_err} exceeds bound {mean_bound}"
    );
    assert!(
        total_max_err <= max_bound,
        "{target:?} max_err {total_max_err} exceeds bound {max_bound}"
    );
}
