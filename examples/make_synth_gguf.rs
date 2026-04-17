//! Write a minimal Q8_0 GGUF v3 file to disk. Intended for smoke-testing
//! the llmdb stack end-to-end without having to download or convert a real
//! model.
//!
//! Usage:
//!     cargo run --example make_synth_gguf -- <output_path> [tensor_count]
//!
//! The generated file has `tensor_count` Q8_0 tensors, each of 8192 weights
//! (= 4096 stego bytes = 1 logical block of capacity). Metadata consumes
//! 4 blocks (superblock + integrity + redirection + file table), so after
//! `llmdb init` the data region holds `tensor_count - 4` usable blocks.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

const GGML_TYPE_Q8_0_ID: u32 = 8;
const ALIGNMENT: usize = 32;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: {} <output_path> [tensor_count]\n\
             \n\
             Generates a synthetic Q8_0 GGUF v3 with `tensor_count` tensors\n\
             (default 16), each 8192 weights. Each tensor gives one block\n\
             of stego capacity.",
            args.first().map(String::as_str).unwrap_or("make_synth_gguf")
        );
        return ExitCode::from(1);
    }

    let output = PathBuf::from(&args[1]);
    let tensor_count: usize = args
        .get(2)
        .map(|s| s.parse().unwrap_or(16))
        .unwrap_or(16);

    let tensors = build_tensor_specs(tensor_count);
    let bytes = encode_gguf_v3(&tensors);
    match fs::write(&output, &bytes) {
        Ok(()) => {
            println!(
                "wrote {} ({} bytes, {} tensors → ~{} data blocks after init)",
                output.display(),
                bytes.len(),
                tensor_count,
                tensor_count.saturating_sub(4)
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error writing {}: {e}", output.display());
            ExitCode::from(2)
        }
    }
}

struct TensorSpec {
    name: String,
    dimensions: Vec<u64>,
    raw_type_id: u32,
    data: Vec<u8>,
}

fn build_tensor_specs(count: usize) -> Vec<TensorSpec> {
    (0..count)
        .map(|i| {
            let name = format!("blk.{}.ffn_down.weight", count - 1 - i);
            let weight_count = 8192_usize;
            let chunk_count = weight_count / 32;
            TensorSpec {
                name,
                dimensions: vec![weight_count as u64],
                raw_type_id: GGML_TYPE_Q8_0_ID,
                // Q8_0 block: 2 bytes scale (fp16) + 32 bytes of weights = 34 bytes.
                data: vec![0_u8; chunk_count * 34],
            }
        })
        .collect()
}

fn encode_gguf_v3(tensors: &[TensorSpec]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(3_u64).to_le_bytes()); // metadata kv count

    write_kv_u32(&mut bytes, "general.alignment", ALIGNMENT as u32);
    write_kv_string(&mut bytes, "general.architecture", "llama");
    write_kv_string(&mut bytes, "tokenizer.ggml.model", "gpt2");

    let mut data_offsets = Vec::with_capacity(tensors.len());
    let mut cursor = 0_usize;
    for t in tensors {
        cursor = align_to(cursor, ALIGNMENT);
        data_offsets.push(cursor as u64);
        cursor += t.data.len();
    }

    for (t, data_offset) in tensors.iter().zip(data_offsets.iter().copied()) {
        write_string(&mut bytes, &t.name);
        bytes.extend_from_slice(&(t.dimensions.len() as u32).to_le_bytes());
        for d in &t.dimensions {
            bytes.extend_from_slice(&d.to_le_bytes());
        }
        bytes.extend_from_slice(&t.raw_type_id.to_le_bytes());
        bytes.extend_from_slice(&data_offset.to_le_bytes());
    }

    let padding = align_to(bytes.len(), ALIGNMENT) - bytes.len();
    bytes.resize(bytes.len() + padding, 0);

    let blob_start = bytes.len();
    for (t, data_offset) in tensors.iter().zip(data_offsets.iter().copied()) {
        let absolute = blob_start + data_offset as usize;
        if bytes.len() < absolute {
            bytes.resize(absolute, 0);
        }
        bytes.extend_from_slice(&t.data);
    }

    bytes
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
