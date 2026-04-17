//! Synthetic GGUF fixture builder for benches. Mirrors the logic in
//! `tests/common/mod.rs` — kept separate because Criterion benches
//! cannot `use` paths from the `tests/` tree.

#![allow(dead_code)]

use std::fs;
use std::path::PathBuf;

use tempfile::TempDir;

use llmdb::gguf::quant::GGML_TYPE_Q8_0_ID;

pub struct FixtureHandle {
    _root: TempDir,
    pub path: PathBuf,
}

pub struct SyntheticTensorSpec {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub raw_type_id: u32,
    pub data: Vec<u8>,
}

pub fn write_fixture_v3(name: &str, tensors: &[SyntheticTensorSpec]) -> FixtureHandle {
    let root = tempfile::tempdir().expect("temp dir");
    let path = root.path().join(name);

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&3_u64.to_le_bytes());

    write_kv_u32(&mut bytes, "general.alignment", 32);
    write_kv_string(&mut bytes, "general.architecture", "llama");
    write_kv_string(&mut bytes, "tokenizer.ggml.model", "gpt2");

    let mut data_offsets = Vec::with_capacity(tensors.len());
    let mut blob_cursor = 0_usize;
    for tensor in tensors {
        blob_cursor = align_to(blob_cursor, 32);
        data_offsets.push(blob_cursor as u64);
        blob_cursor += tensor.data.len();
    }

    for (tensor, data_offset) in tensors.iter().zip(data_offsets.iter().copied()) {
        write_string(&mut bytes, &tensor.name);
        bytes.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
        for dimension in &tensor.dimensions {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        bytes.extend_from_slice(&tensor.raw_type_id.to_le_bytes());
        bytes.extend_from_slice(&data_offset.to_le_bytes());
    }

    let padding = align_to(bytes.len(), 32) - bytes.len();
    bytes.resize(bytes.len() + padding, 0);

    let tensor_blob_start = bytes.len();
    for (tensor, data_offset) in tensors.iter().zip(data_offsets.iter().copied()) {
        let absolute_offset = tensor_blob_start + data_offset as usize;
        if bytes.len() < absolute_offset {
            bytes.resize(absolute_offset, 0);
        }
        bytes.extend_from_slice(&tensor.data);
    }

    fs::write(&path, bytes).expect("write fixture");
    FixtureHandle { _root: root, path }
}

/// `count` Q8_0 tensors each of `weights_per_tensor` weights — enough
/// to give roughly `count * weights_per_tensor / 2 / 4096` stego blocks.
pub fn q8_tensors(count: usize, weights_per_tensor: usize) -> Vec<SyntheticTensorSpec> {
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
            let chunk_count = weights_per_tensor / 32;
            SyntheticTensorSpec {
                name,
                dimensions: vec![weights_per_tensor as u64],
                raw_type_id: GGML_TYPE_Q8_0_ID,
                data: vec![0_u8; chunk_count * 34],
            }
        })
        .collect()
}

/// Deterministic xorshift for repeatable random indices inside a bench.
pub struct Xorshift(pub u64);

impl Xorshift {
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn next_range(&mut self, bound: u32) -> u32 {
        (self.next() % bound as u64) as u32
    }
}

fn write_kv_u32(bytes: &mut Vec<u8>, key: &str, value: u32) {
    write_string(bytes, key);
    bytes.extend_from_slice(&4_u32.to_le_bytes());
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_kv_string(bytes: &mut Vec<u8>, key: &str, value: &str) {
    write_string(bytes, key);
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    write_string(bytes, value);
}

fn write_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn align_to(value: usize, alignment: usize) -> usize {
    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value + (alignment - remainder)
    }
}
