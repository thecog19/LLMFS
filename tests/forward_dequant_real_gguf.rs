//! Integration tests for `llmdb::forward::dequant`.
//!
//! The unit tests in `src/forward/dequant.rs` cover correctness
//! with hand-constructed blocks. These integration tests exercise
//! the same code path against a *real* GGUF tensor row pulled out
//! of `models/smollm2-135m-f16.gguf` — no more, no less. They're
//! skipped when the model isn't present (gitignored — see
//! `.gitignore`'s `/models/*` rule).
//!
//! We cannot compare output to a reference binary per the no-
//! patching constraint, so the assertions are **structural**:
//!
//! - Every dequantized weight is finite (no NaN, no Inf).
//! - Magnitudes fall inside a plausible range for a real
//!   transformer layer (≤ ~10, not uniformly zero, non-trivial
//!   standard deviation).
//! - Dequantizing the same bytes twice gives bit-identical output
//!   (determinism).
//!
//! Catching "did we wire up f16 decoding correctly" is the goal —
//! numerical parity with `llama.cpp` is the A8 end-to-end gate.

use std::path::Path;

use memmap2::Mmap;

use llmdb::forward::dequant::{dequantize_row, dequantize_row_into};
use llmdb::gguf::parser::parse_path;
use llmdb::gguf::quant::GgufQuantType;

const SMOLLM2: &str = "models/smollm2-135m-f16.gguf";

fn mmap_or_skip(path: &str) -> Option<Mmap> {
    if !Path::new(path).exists() {
        eprintln!("skipping: {path} not present");
        return None;
    }
    let file = std::fs::File::open(path).expect("open gguf");
    // SAFETY: file is read-only for the duration of the test.
    let mmap = unsafe { Mmap::map(&file).expect("mmap gguf") };
    Some(mmap)
}

#[test]
fn smollm2_token_embed_first_row_dequantizes_cleanly() {
    let Some(mmap) = mmap_or_skip(SMOLLM2) else {
        return;
    };
    let gguf = parse_path(SMOLLM2).expect("parse gguf");

    // `token_embd.weight` is always present in a Llama GGUF. For
    // SmolLM2-135M-F16 it's F16, shape [hidden_dim, vocab].
    let tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .expect("token_embd.weight present");
    assert_eq!(tensor.quant_type(), Some(GgufQuantType::F16));

    // First "row" = first hidden_dim worth of weights, i.e. the
    // embedding vector for token id 0.
    let hidden_dim = tensor.dimensions[0] as usize;
    let abs = tensor
        .absolute_offset(gguf.tensor_data_offset)
        .expect("absolute offset fits u64") as usize;
    let row_bytes = &mmap[abs..abs + hidden_dim * 2]; // f16 = 2 bytes

    let row = dequantize_row(GgufQuantType::F16, row_bytes).expect("dequant row");
    assert_eq!(row.len(), hidden_dim);

    // Structural bounds: finite, non-trivial, not absurdly large.
    assert!(
        row.iter().all(|v| v.is_finite()),
        "token_embd row 0 had a non-finite weight",
    );
    let max_abs = row.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
    assert!(
        max_abs < 10.0,
        "token_embd row 0 max |w| = {max_abs}; real SmolLM2 embeddings are O(1)",
    );
    let mean: f32 = row.iter().sum::<f32>() / row.len() as f32;
    let var: f32 =
        row.iter().map(|v| (*v - mean).powi(2)).sum::<f32>() / row.len() as f32;
    let std = var.sqrt();
    assert!(std > 1e-4, "embedding row collapsed to constant (std {std})");
}

#[test]
fn smollm2_block0_attn_q_first_row_is_well_behaved() {
    let Some(mmap) = mmap_or_skip(SMOLLM2) else {
        return;
    };
    let gguf = parse_path(SMOLLM2).expect("parse gguf");

    // Layer-0 Q projection weight — sanity that non-embedding
    // tensors dequantize too.
    let tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_q.weight")
        .expect("blk.0.attn_q.weight present");
    assert_eq!(tensor.quant_type(), Some(GgufQuantType::F16));

    let cols = tensor.dimensions[0] as usize;
    let abs = tensor
        .absolute_offset(gguf.tensor_data_offset)
        .expect("abs offset") as usize;
    let row_bytes = &mmap[abs..abs + cols * 2];

    let mut row = vec![0.0_f32; cols];
    dequantize_row_into(GgufQuantType::F16, row_bytes, &mut row).expect("dequant");

    assert!(row.iter().all(|v| v.is_finite()));
    let max_abs = row.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
    // Attention projections are typically initialized / trained so
    // that weights are ≤ ~1, but be generous here — the contract
    // is "real values, not exploded", not a precise bound.
    assert!(max_abs < 5.0, "attn_q row 0 max |w| {max_abs} looks wrong");
}

#[test]
fn smollm2_f16_dequant_is_deterministic() {
    // Bit-for-bit identical on repeated calls against the same
    // bytes. Sanity against any accidental nondeterminism (e.g.
    // iteration order over a HashMap) in the dequant path.
    let Some(mmap) = mmap_or_skip(SMOLLM2) else {
        return;
    };
    let gguf = parse_path(SMOLLM2).expect("parse gguf");
    let tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .expect("token_embd.weight present");
    let hidden_dim = tensor.dimensions[0] as usize;
    let abs = tensor
        .absolute_offset(gguf.tensor_data_offset)
        .expect("abs offset") as usize;
    let bytes = &mmap[abs..abs + hidden_dim * 2];

    let a = dequantize_row(GgufQuantType::F16, bytes).unwrap();
    let b = dequantize_row(GgufQuantType::F16, bytes).unwrap();
    assert_eq!(a, b, "dequant is not deterministic");
}
