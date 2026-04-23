//! D0 measurement harness — dumps full per-(site, layer) Hessians
//! to disk for offline analysis.
//!
//! `#[ignore]`'d by default: both tests load a real model, run a
//! multi-thousand-token forward pass with [`HessianAccumulator`],
//! and write ~1.3 GB of F32 upper-triangle data per cover (Qwen
//! 0.5B case; SmolLM2 is ~65 MB). Run them only when you're ready
//! to re-measure for the Phase D decision.
//!
//! ```
//! cargo test --release --test hessian_measure -- --ignored --nocapture
//! ```
//!
//! Output layout (under `target/hessian-dump/<cover-tag>/`):
//!
//!   * `<layer_idx>_<site_tag>.f32`  — raw little-endian F32 upper
//!     triangle, row-major within the triangle, length
//!     `N*(N+1)/2`. No per-file header; shape metadata lives in the
//!     sibling `manifest.json`.
//!   * `manifest.json` — list of (site, layer, N, file) tuples plus
//!     corpus + model metadata. Consumed by
//!     `scripts/analyze-hessian.py`.
//!
//! The dump is the *mean* Hessian (accumulated `Σ_t x_t x_t^T`
//! divided by token count), matching what Phase E's compensation
//! math expects as the activation second-moment statistic. F64
//! accumulation → F32 at dump time keeps disk size tractable.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use llmdb::forward::{
    ActivationSite, ForwardModel, HessianAccumulator, KvCache, ModelScratch,
};

const SMOLLM2_Q8_0: &str = "models/smollm2-135m-q8_0.gguf";
const SMOLLM2_F16: &str = "models/smollm2-135m-f16.gguf";
const CORPUS_PATH: &str = "benches/fixtures/wiki.test.raw";
const DUMP_ROOT: &str = "target/hessian-dump";

/// Target token count. Picked to exceed `N_max` at every observation
/// site so H is full-rank — i.e., the eigenvalue tail reflects real
/// structure rather than rank truncation. SmolLM2-135M: `N_max =
/// ffn_dim = 1536`, target 2048.
///
/// Note on scope: D0 originally planned to also dump Qwen-2.5-0.5B
/// for a cross-architecture sanity check, but the forward pass only
/// supports llama-arch GGUFs with gpt2-style tokenizers (see
/// `src/forward/config.rs:UnsupportedArch` and the matching
/// tokenizer guard). Of the locally available models, only SmolLM2
/// satisfies both. The two SmolLM2 quantizations (F16 + Q8_0)
/// stand in as a second data point: they share architecture but
/// differ in the per-weight quantization noise, so identical H
/// structure across the two also validates that the dequant path
/// preserves the activation second-moment statistic.
const TARGET_TOKENS: usize = 2048;

fn site_tag(site: ActivationSite) -> &'static str {
    match site {
        ActivationSite::QkvInput => "qkv_input",
        ActivationSite::AttnOutputInput => "attn_output_input",
        ActivationSite::FfnGateUpInput => "ffn_gate_up_input",
        ActivationSite::FfnDownInput => "ffn_down_input",
    }
}

fn tensor_names_for(site: ActivationSite, layer: usize) -> Vec<String> {
    // Mirrors `src/forward/awq.rs::tensor_names_for`. Kept local to
    // the test to avoid pulling internal helpers into the crate's
    // public surface just for D0.
    match site {
        ActivationSite::QkvInput => vec![
            format!("blk.{layer}.attn_q.weight"),
            format!("blk.{layer}.attn_k.weight"),
            format!("blk.{layer}.attn_v.weight"),
        ],
        ActivationSite::AttnOutputInput => {
            vec![format!("blk.{layer}.attn_output.weight")]
        }
        ActivationSite::FfnGateUpInput => vec![
            format!("blk.{layer}.ffn_gate.weight"),
            format!("blk.{layer}.ffn_up.weight"),
        ],
        ActivationSite::FfnDownInput => vec![format!("blk.{layer}.ffn_down.weight")],
    }
}

/// Run one cover end-to-end: load model, tokenize a prefix of
/// `wiki.test.raw` of length `target_tokens` (post-tokenize),
/// forward + accumulate, dump per-(site, layer) upper triangles and
/// a manifest.
fn run_dump(model_path: &str, cover_tag: &str, target_tokens: usize) {
    if !Path::new(model_path).exists() {
        eprintln!("skipping {cover_tag}: {model_path} not present");
        return;
    }
    if !Path::new(CORPUS_PATH).exists() {
        eprintln!(
            "skipping {cover_tag}: corpus {CORPUS_PATH} not present (run \
             from repo root)"
        );
        return;
    }

    let corpus = fs::read_to_string(CORPUS_PATH).expect("read wiki.test.raw");

    eprintln!("=== {cover_tag}: loading {model_path} ===");
    let t0 = std::time::Instant::now();
    let model = ForwardModel::load(model_path).expect("load model");
    eprintln!("loaded in {:.1?}", t0.elapsed());

    // Take a chunk generous enough to exceed target_tokens at any
    // tokenization rate. 6× the target in characters covers even
    // single-char-per-token worst cases with headroom.
    let char_budget = (target_tokens * 6).min(corpus.len());
    let chunk = &corpus[..char_budget];

    eprintln!("tokenizing ({char_budget} chars, target {target_tokens} tokens)...");
    let t0 = std::time::Instant::now();
    let mut tokens = model.encode(chunk).expect("encode corpus");
    eprintln!("encoded {} tokens in {:.1?}", tokens.len(), t0.elapsed());
    assert!(
        tokens.len() >= target_tokens,
        "corpus produced only {} tokens, need ≥ {}",
        tokens.len(),
        target_tokens,
    );
    tokens.truncate(target_tokens);

    // One forward pass, one HessianAccumulator, many observe calls.
    let ctx_len = tokens.len();
    eprintln!(
        "forward pass with HessianAccumulator (ctx_len = {ctx_len}, \
         {} layers)...",
        model.config.n_layers,
    );
    let mut cache = KvCache::new(&model.config, ctx_len);
    let mut scratch = ModelScratch::new(&model.config, ctx_len, ctx_len);
    let mut acc = HessianAccumulator::new();
    let t0 = std::time::Instant::now();
    let _ = model.forward_all_logits_with_observer(&tokens, &mut cache, &mut scratch, &mut acc);
    eprintln!("forward + accumulate in {:.1?}", t0.elapsed());

    // Finalize to mean-H F32 upper triangles per (site, layer).
    let finalized = acc.finalize();
    eprintln!("finalized {} (site, layer) entries", finalized.len());

    // Prepare dump directory.
    let dump_dir: PathBuf = [DUMP_ROOT, cover_tag].iter().collect();
    fs::create_dir_all(&dump_dir).expect("create dump dir");
    // Clean out any stale files from a prior run.
    for entry in fs::read_dir(&dump_dir).expect("read dump dir") {
        let entry = entry.expect("dir entry");
        let _ = fs::remove_file(entry.path());
    }

    // Dump each (site, layer) and collect manifest entries.
    let mut manifest_entries: Vec<serde_json::Value> = Vec::with_capacity(finalized.len());
    let mut total_bytes: u64 = 0;
    let mut sorted: Vec<_> = finalized.iter().collect();
    sorted.sort_by_key(|((site, layer), _)| (*layer, site_tag(*site)));
    for ((site, layer), (n, tri)) in sorted {
        let fname = format!("{:02}_{}.f32", layer, site_tag(*site));
        let fpath = dump_dir.join(&fname);
        let mut f = fs::File::create(&fpath).expect("create dump file");
        // Little-endian F32 per entry, via a single bytemuck-free
        // write_all over the raw bytes.
        let byte_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(tri.as_ptr() as *const u8, tri.len() * 4)
        };
        f.write_all(byte_slice).expect("write triangle");
        total_bytes += byte_slice.len() as u64;

        manifest_entries.push(serde_json::json!({
            "site": site_tag(*site),
            "layer": layer,
            "n": n,
            "file": fname,
            "tensor_names": tensor_names_for(*site, *layer),
            "triangle_len": tri.len(),
        }));
    }

    let manifest = serde_json::json!({
        "cover": Path::new(model_path).file_name().unwrap().to_string_lossy(),
        "cover_tag": cover_tag,
        "corpus": Path::new(CORPUS_PATH).file_name().unwrap().to_string_lossy(),
        "token_count": ctx_len,
        "hidden_dim": model.config.hidden_dim,
        "ffn_dim": model.config.ffn_dim,
        "n_layers": model.config.n_layers,
        "n_heads": model.config.n_heads,
        "n_kv_heads": model.config.n_kv_heads,
        "head_dim": model.config.head_dim,
        "dtype": "f32",
        "layout": "upper_triangle_row_major",
        "entries": manifest_entries,
    });
    fs::write(
        dump_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).expect("serialize manifest"),
    )
    .expect("write manifest");

    eprintln!(
        "dumped {} files, {:.1} MB, manifest at {}/manifest.json",
        manifest["entries"].as_array().unwrap().len(),
        total_bytes as f64 / (1024.0 * 1024.0),
        dump_dir.display(),
    );
}

#[test]
#[ignore = "D0 measurement: loads smollm2-135m-q8_0 + wiki.test.raw, runs a \
            2048-token forward pass with full-H accumulator, dumps ~190 MB \
            of F32 upper triangles to target/hessian-dump/smollm2-135m-q8_0/. \
            ~7 min on CPU. Run explicitly via --ignored --nocapture."]
fn dump_smollm2_q8_0_hessians() {
    run_dump(SMOLLM2_Q8_0, "smollm2-135m-q8_0", TARGET_TOKENS);
}

#[test]
#[ignore = "D0 measurement: loads smollm2-135m-f16 + wiki.test.raw, runs the \
            same 2048-token forward pass on the F16 cover. Used as a \
            quantization-stability check — H structure should match the \
            Q8_0 dump within dequant-noise tolerance. ~7 min on CPU."]
fn dump_smollm2_f16_hessians() {
    run_dump(SMOLLM2_F16, "smollm2-135m-f16", TARGET_TOKENS);
}
