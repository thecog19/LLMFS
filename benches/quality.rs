//! Coarse "did we notice" quality regression detector. Not a
//! perplexity benchmark (that is V2's job). Given a real GGUF path
//! in `LLMDB_E2E_GGUF`, runs a fixed prompt through `llama-cli` on
//! both the baseline and a stego'd copy, then hashes the outputs so
//! a diff is visible in CI output.
//!
//! Without `LLMDB_E2E_GGUF` set this bench prints a skip message and
//! exits — `cargo bench` stays green on machines without real models.

use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Command;

use crc32fast::Hasher;
use criterion::{Criterion, criterion_group, criterion_main};
use llmdb::stego::device::StegoDevice;
use llmdb::stego::planner::AllocationMode;

const PROMPT: &str = "Once upon a time";
const MAX_TOKENS: &str = "80";

fn llama_cli() -> Option<String> {
    std::env::var("LLMDB_LLAMA_CLI")
        .ok()
        .or_else(|| which("llama-cli"))
}

fn which(binary: &str) -> Option<String> {
    Command::new("which")
        .arg(binary)
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_owned())
}

fn hash(bytes: &[u8]) -> u32 {
    let mut h = Hasher::new();
    h.update(bytes);
    h.finalize()
}

fn run_inference(cli: &str, model: &Path) -> Result<Vec<u8>, String> {
    let output = Command::new(cli)
        .arg("-m")
        .arg(model)
        .arg("-p")
        .arg(PROMPT)
        .arg("-n")
        .arg(MAX_TOKENS)
        .arg("--temp")
        .arg("0")
        .arg("--no-display-prompt")
        .output()
        .map_err(|e| format!("spawn {cli}: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "{cli} exit {:?}: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(output.stdout)
}

fn bench_quality(c: &mut Criterion) {
    let Ok(gguf) = std::env::var("LLMDB_E2E_GGUF") else {
        writeln!(
            std::io::stderr(),
            "LLMDB_E2E_GGUF not set; skipping quality bench"
        )
        .ok();
        return;
    };
    let Some(cli) = llama_cli() else {
        writeln!(
            std::io::stderr(),
            "llama-cli not on PATH (set LLMDB_LLAMA_CLI to override); skipping"
        )
        .ok();
        return;
    };
    let model = Path::new(&gguf);
    if !model.exists() {
        writeln!(
            std::io::stderr(),
            "LLMDB_E2E_GGUF points at {}, which does not exist",
            model.display()
        )
        .ok();
        return;
    }

    // Stage: baseline copy and a stego'd copy side by side.
    let tmp = tempfile::tempdir().expect("tempdir");
    let baseline = tmp.path().join("baseline.gguf");
    let stego = tmp.path().join("stego.gguf");
    fs::copy(model, &baseline).expect("copy baseline");
    fs::copy(model, &stego).expect("copy stego");

    let mut device = StegoDevice::initialize(&stego, AllocationMode::Standard).expect("init");
    device.flush().expect("flush");
    drop(device);

    let baseline_out = run_inference(&cli, &baseline).expect("baseline inference");
    let stego_out = run_inference(&cli, &stego).expect("stego inference");

    let baseline_hash = hash(&baseline_out);
    let stego_hash = hash(&stego_out);
    let diff_bytes = baseline_out
        .iter()
        .zip(stego_out.iter())
        .filter(|(a, b)| a != b)
        .count()
        + baseline_out.len().abs_diff(stego_out.len());

    writeln!(
        std::io::stderr(),
        "quality: baseline_crc={baseline_hash:#010x} stego_crc={stego_hash:#010x} diff_bytes={diff_bytes}"
    )
    .ok();

    // Register a single one-shot measurement so `cargo bench` shows
    // the run in its output. The actual inference already happened
    // above — this is just so Criterion has something to report.
    c.bench_function("quality/init_only", |b| {
        b.iter(|| {
            std::hint::black_box(baseline_hash);
            std::hint::black_box(stego_hash);
        });
    });
}

criterion_group!(benches, bench_quality);
criterion_main!(benches);
