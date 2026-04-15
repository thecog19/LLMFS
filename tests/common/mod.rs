#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

use tempfile::TempDir;

use llmdb::gguf::quant::GGML_TYPE_Q8_0_ID;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyntheticGgufVersion {
    V2,
    V3,
}

impl SyntheticGgufVersion {
    fn as_u32(self) -> u32 {
        match self {
            Self::V2 => 2,
            Self::V3 => 3,
        }
    }
}

#[derive(Debug)]
pub struct FixtureHandle {
    root: TempDir,
    pub path: PathBuf,
}

impl FixtureHandle {
    pub fn root(&self) -> &Path {
        self.root.path()
    }
}

pub fn write_synthetic_gguf_fixture(version: SyntheticGgufVersion, name: &str) -> FixtureHandle {
    let root = tempfile::tempdir().expect("temp dir");
    let path = root.path().join(name);

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&version.as_u32().to_le_bytes());
    bytes.extend_from_slice(&(0_u64).to_le_bytes());
    bytes.extend_from_slice(&(0_u64).to_le_bytes());

    fs::write(&path, bytes).expect("write synthetic gguf fixture");

    FixtureHandle { root, path }
}

#[derive(Debug, Clone)]
pub struct SyntheticTensorSpec {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub raw_type_id: u32,
    pub data: Vec<u8>,
}

pub fn write_parser_fixture(version: SyntheticGgufVersion, name: &str) -> FixtureHandle {
    write_custom_gguf_fixture(
        version,
        name,
        &[SyntheticTensorSpec {
            name: "blk.0.ffn_down.weight".to_owned(),
            dimensions: vec![4, 8],
            raw_type_id: GGML_TYPE_Q8_0_ID,
            data: vec![0_u8; 34],
        }],
    )
}

pub fn write_custom_gguf_fixture(
    version: SyntheticGgufVersion,
    name: &str,
    tensors: &[SyntheticTensorSpec],
) -> FixtureHandle {
    let root = tempfile::tempdir().expect("temp dir");
    let path = root.path().join(name);

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&version.as_u32().to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(3_u64).to_le_bytes());

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

    fs::write(&path, bytes).expect("write custom gguf fixture");

    FixtureHandle { root, path }
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
