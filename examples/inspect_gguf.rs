//! Dump GGUF metadata relevant to the forward-pass calibration build.
//!
//! Scoped to: tokenizer type, architecture, and the Llama config keys
//! the forward pass (`src/forward/`) will need. Not a general-purpose
//! GGUF browser — just what Phase A0 / A1 need to discover.
//!
//! Usage: cargo run --example inspect_gguf -- <model.gguf>

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use llmdb::gguf::parser::{GgufFile, GgufMetadataValue, parse_path};

fn main() -> ExitCode {
    let Some(path) = env::args().nth(1).map(PathBuf::from) else {
        eprintln!("usage: inspect_gguf <model.gguf>");
        return ExitCode::from(2);
    };

    let gguf = match parse_path(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("parse {}: {e}", path.display());
            return ExitCode::from(1);
        }
    };

    println!("file:               {}", path.display());
    println!("gguf version:       {}", gguf.header.version);
    println!("tensors:            {}", gguf.tensors.len());
    println!("metadata entries:   {}", gguf.metadata.len());
    println!("tensor_data_offset: {}", gguf.tensor_data_offset);
    println!();

    print_string(&gguf, "general.architecture");
    print_string(&gguf, "general.name");
    println!();

    println!("-- tokenizer --");
    print_string(&gguf, "tokenizer.ggml.model");
    print_string(&gguf, "tokenizer.ggml.pre");
    print_u32(&gguf, "tokenizer.ggml.bos_token_id");
    print_u32(&gguf, "tokenizer.ggml.eos_token_id");
    print_u32(&gguf, "tokenizer.ggml.padding_token_id");
    print_u32(&gguf, "tokenizer.ggml.unknown_token_id");
    print_bool(&gguf, "tokenizer.ggml.add_bos_token");
    print_bool(&gguf, "tokenizer.ggml.add_eos_token");
    print_array_len(&gguf, "tokenizer.ggml.tokens");
    print_array_len(&gguf, "tokenizer.ggml.scores");
    print_array_len(&gguf, "tokenizer.ggml.token_type");
    print_array_len(&gguf, "tokenizer.ggml.merges");
    println!();

    // Architecture-specific keys. We prefix-search so this works for
    // any `<arch>.*` set — llama.*, qwen2.*, etc.
    println!("-- architecture config (first matching arch prefix) --");
    let arch_prefix = arch_key_prefix(&gguf);
    for key in [
        "context_length",
        "embedding_length",
        "block_count",
        "feed_forward_length",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.layer_norm_rms_epsilon",
        "rope.dimension_count",
        "rope.freq_base",
        "rope.scaling.type",
        "rope.scaling.factor",
    ] {
        let full = format!("{arch_prefix}.{key}");
        print_auto(&gguf, &full);
    }

    ExitCode::SUCCESS
}

fn arch_key_prefix(gguf: &GgufFile) -> String {
    match gguf.find_metadata_value("general.architecture") {
        Some(GgufMetadataValue::String(s)) => s.clone(),
        _ => "llama".to_owned(),
    }
}

fn print_string(gguf: &GgufFile, key: &str) {
    match gguf.find_metadata_value(key) {
        Some(GgufMetadataValue::String(s)) => println!("{key:>40}: {s}"),
        Some(_) => println!("{key:>40}: <wrong type>"),
        None => println!("{key:>40}: <absent>"),
    }
}

fn print_u32(gguf: &GgufFile, key: &str) {
    match gguf.find_metadata_value(key) {
        Some(GgufMetadataValue::Uint32(v)) => println!("{key:>40}: {v}"),
        Some(GgufMetadataValue::Uint64(v)) => println!("{key:>40}: {v}"),
        Some(GgufMetadataValue::Int32(v)) => println!("{key:>40}: {v}"),
        Some(GgufMetadataValue::Int64(v)) => println!("{key:>40}: {v}"),
        Some(_) => println!("{key:>40}: <non-int>"),
        None => {}
    }
}

fn print_bool(gguf: &GgufFile, key: &str) {
    match gguf.find_metadata_value(key) {
        Some(GgufMetadataValue::Bool(v)) => println!("{key:>40}: {v}"),
        Some(_) => println!("{key:>40}: <non-bool>"),
        None => {}
    }
}

fn print_array_len(gguf: &GgufFile, key: &str) {
    match gguf.find_metadata_value(key) {
        Some(GgufMetadataValue::Array { values, .. }) => {
            println!("{key:>40}: <array, {} entries>", values.len())
        }
        Some(_) => println!("{key:>40}: <non-array>"),
        None => {}
    }
}

fn print_auto(gguf: &GgufFile, key: &str) {
    let Some(v) = gguf.find_metadata_value(key) else {
        return;
    };
    match v {
        GgufMetadataValue::String(s) => println!("{key:>40}: {s}"),
        GgufMetadataValue::Uint32(x) => println!("{key:>40}: {x}"),
        GgufMetadataValue::Uint64(x) => println!("{key:>40}: {x}"),
        GgufMetadataValue::Int32(x) => println!("{key:>40}: {x}"),
        GgufMetadataValue::Int64(x) => println!("{key:>40}: {x}"),
        GgufMetadataValue::Float32(x) => println!("{key:>40}: {x}"),
        GgufMetadataValue::Float64(x) => println!("{key:>40}: {x}"),
        GgufMetadataValue::Bool(x) => println!("{key:>40}: {x}"),
        GgufMetadataValue::Array { values, .. } => {
            println!("{key:>40}: <array, {} entries>", values.len())
        }
        _ => println!("{key:>40}: <?>"),
    }
}
