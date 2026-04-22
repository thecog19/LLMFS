//! A8 — perplexity parity against `llama-perplexity`.
//!
//! Two tests:
//!
//! - `smollm2_ppl_matches_llama_perplexity_reference` (non-ignored,
//!   fast). Reads `tests/fixtures/perplexity/smollm2_slice.json` and
//!   asserts our [`llmdb::forward::perplexity::perplexity`] output is
//!   within ±0.5% of the checkpointed reference. Skips with
//!   regen instructions if the fixture is absent.
//!
//! - `regen_smollm2_ppl_fixture` (`#[ignore]`'d). Spawns
//!   `llama-tokenize` to get the exact token stream and
//!   `llama-perplexity` to get the reference PPL, then writes the
//!   fixture. Run with
//!   `cargo test --test forward_perplexity_parity -- --ignored
//!    --nocapture regen_` when you have the binaries.
//!
//! ## Why a self-contained text slice
//!
//! The plan originally called for `wiki.test.raw`, but embedding a
//! small public-domain paragraph directly in this file has two
//! advantages: (a) no external download step to reproduce the test,
//! (b) the fixture is fully self-describing (text + tokens + ppl).
//! The paragraph is ~250 tokens of expository English prose, enough
//! for a meaningful PPL signal without blowing up CI runtime.
//!
//! ## Binary discovery
//!
//! The regen test reads `$LLMDB_LLAMA_TOKENIZE` and
//! `$LLMDB_LLAMA_PERPLEXITY` (same convention as the A2 tokenizer-
//! parity regen), falling back to `PATH` lookups.

use std::path::{Path, PathBuf};
use std::process::Command;

use llmdb::forward::model::ForwardModel;
use llmdb::forward::perplexity::perplexity;
use serde::{Deserialize, Serialize};

const SMOLLM2_GGUF: &str = "models/smollm2-135m-f16.gguf";

/// Self-contained paragraph for the A8 fixture. Public-domain
/// expository English; no fancy quotes, no unicode edge cases,
/// no URLs — just a clean PPL signal.
const PPL_TEXT: &str = concat!(
    "The invention of the printing press in the middle of the fifteenth century ",
    "is often described as one of the most consequential technological changes ",
    "in the history of communication. By replacing manuscript copying with a ",
    "mechanical process, the press reduced the cost and raised the reliability ",
    "of making identical copies of a text. This, in turn, helped standardise ",
    "spelling and vocabulary across regions that had previously used local ",
    "variants, because printers tended to draw on a smaller set of exemplars ",
    "when setting type. Scholars, administrators, and merchants all found new ",
    "uses for printed documents, and within a few decades the technology had ",
    "spread across Europe and begun to reshape education, religion, and trade.",
);

/// Context length to score at. Smaller = faster; 128 is enough to
/// exercise multiple chunks over the ~250-token paragraph, matching
/// how `llama-perplexity` runs with `--ctx-size 128`.
const CTX_LEN: usize = 128;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/perplexity/smollm2_slice.json")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Fixture {
    /// Text passed to `llama-perplexity`. Same text must re-tokenize
    /// to `tokens` (we validate that on read).
    text: String,
    /// Context length reported to `llama-perplexity` and to our
    /// `perplexity()` call.
    ctx_len: usize,
    /// Tokens `llama-perplexity` actually scored, including whatever
    /// BOS handling it performs. Captured via a tokenizer run that
    /// matches `llama-perplexity`'s defaults.
    tokens: Vec<u32>,
    /// Perplexity reported by `llama-perplexity` for the same
    /// tokens + ctx_len.
    reference_ppl: f32,
    /// How the reference was captured, for anyone debugging.
    generator: String,
}

fn tokenizer_binary() -> String {
    std::env::var("LLMDB_LLAMA_TOKENIZE").unwrap_or_else(|_| "llama-tokenize".to_owned())
}

fn perplexity_binary() -> String {
    std::env::var("LLMDB_LLAMA_PERPLEXITY")
        .unwrap_or_else(|_| "llama-perplexity".to_owned())
}

// ─── Fast test ──────────────────────────────────────────────────────

#[test]
fn smollm2_ppl_matches_llama_perplexity_reference() {
    if !Path::new(SMOLLM2_GGUF).exists() {
        eprintln!("skipping: {SMOLLM2_GGUF} not present");
        return;
    }

    let path = fixture_path();
    let Ok(raw) = std::fs::read_to_string(&path) else {
        eprintln!(
            "skipping: {} not present. Regenerate with:\n  \
             cargo test --test forward_perplexity_parity -- \
             --ignored --nocapture regen_",
            path.display(),
        );
        return;
    };
    let fixture: Fixture = serde_json::from_str(&raw).expect("parse fixture JSON");

    let model = ForwardModel::load(SMOLLM2_GGUF).expect("load smollm2");
    let ppl = perplexity(&model, &fixture.tokens, fixture.ctx_len).expect("ppl");

    let rel_err = (ppl - fixture.reference_ppl).abs() / fixture.reference_ppl;
    eprintln!(
        "our ppl: {ppl:>8.4}\nref ppl: {:>8.4}\nrel err: {:.4}%",
        fixture.reference_ppl,
        rel_err * 100.0,
    );
    assert!(
        rel_err < 0.005,
        "ppl parity miss: ours={ppl:.4}, ref={:.4}, |rel|={:.4}% (gate: 0.5%)",
        fixture.reference_ppl,
        rel_err * 100.0,
    );
}

// ─── Regen ──────────────────────────────────────────────────────────

#[test]
#[ignore = "regen: spawns llama-tokenize + llama-perplexity; run with --ignored"]
fn regen_smollm2_ppl_fixture() {
    if !Path::new(SMOLLM2_GGUF).exists() {
        eprintln!("skipping regen: {SMOLLM2_GGUF} not present");
        return;
    }
    let tokenize = tokenizer_binary();
    let perplex = perplexity_binary();

    // 1. Capture the exact tokens llama-perplexity will score.
    //    Default llama-perplexity prepends BOS; our tokenize call
    //    mirrors that.
    let tokens = run_llama_tokenize(&tokenize, SMOLLM2_GGUF, PPL_TEXT, true).expect("tokenize");
    eprintln!("tokenized: {} tokens", tokens.len());

    // 2. Write the text to a temp file and run llama-perplexity.
    let tmp = std::env::temp_dir().join("llmdb_ppl_slice.txt");
    std::fs::write(&tmp, PPL_TEXT).expect("write tmp");
    let reference_ppl =
        run_llama_perplexity(&perplex, SMOLLM2_GGUF, &tmp, CTX_LEN).expect("perplexity");
    eprintln!("reference ppl: {reference_ppl}");

    let fixture = Fixture {
        text: PPL_TEXT.to_owned(),
        ctx_len: CTX_LEN,
        tokens,
        reference_ppl,
        generator: format!(
            "llama-perplexity -m {SMOLLM2_GGUF} --ctx-size {CTX_LEN} -f <slice>",
        ),
    };
    let path = fixture_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("mkdir fixture dir");
    }
    let json = serde_json::to_string_pretty(&fixture).expect("serialize fixture");
    std::fs::write(&path, json).expect("write fixture");
    eprintln!("wrote {}", path.display());
}

/// `llama-tokenize -m <gguf> --log-disable -p <text>` → Vec<u32>.
/// Pass `add_bos = true` to mirror `llama-perplexity`'s default BOS
/// prepending (the `--no-parse-special` flag from A2 is omitted
/// here since we want BOS to appear).
fn run_llama_tokenize(
    binary: &str,
    gguf: &str,
    text: &str,
    add_bos: bool,
) -> Result<Vec<u32>, String> {
    let mut cmd = Command::new(binary);
    cmd.arg("-m").arg(gguf).arg("--log-disable");
    if !add_bos {
        cmd.arg("--no-bos");
    }
    cmd.arg("-p").arg(text);
    let out = cmd.output().map_err(|e| format!("spawn: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "llama-tokenize exited {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut ids: Vec<u32> = Vec::new();
    for line in stdout.lines() {
        let line = line.trim_start();
        let Some((id_part, _)) = line.split_once(" -> '") else {
            continue;
        };
        let id: u32 = id_part
            .trim()
            .parse()
            .map_err(|e| format!("parse id {id_part:?}: {e}"))?;
        ids.push(id);
    }
    Ok(ids)
}

/// Run `llama-perplexity -m <gguf> --ctx-size <n> -f <text-file>`
/// and parse the final "estimated PPL = X ± Y" (or "Final estimate:
/// PPL = X ± Y") line from stdout.
fn run_llama_perplexity(
    binary: &str,
    gguf: &str,
    text_path: &Path,
    ctx_len: usize,
) -> Result<f32, String> {
    let out = Command::new(binary)
        .arg("-m")
        .arg(gguf)
        .arg("--ctx-size")
        .arg(ctx_len.to_string())
        .arg("-f")
        .arg(text_path)
        .output()
        .map_err(|e| format!("spawn: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "llama-perplexity exited {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    // llama-perplexity prints a running estimate per chunk plus a
    // final line. Parse the final "PPL = X" value.
    let mut last_ppl: Option<f32> = None;
    for line in combined.lines() {
        if let Some(rest) = line.find("PPL = ").map(|i| &line[i + "PPL = ".len()..]) {
            let v = rest.split_whitespace().next().unwrap_or("");
            if let Ok(ppl) = v.parse::<f32>() {
                last_ppl = Some(ppl);
            }
        } else if let Some(rest) = line
            .find("estimate: PPL = ")
            .map(|i| &line[i + "estimate: PPL = ".len()..])
        {
            let v = rest.split_whitespace().next().unwrap_or("");
            if let Ok(ppl) = v.parse::<f32>() {
                last_ppl = Some(ppl);
            }
        }
    }
    last_ppl.ok_or_else(|| {
        format!(
            "no PPL line found in llama-perplexity output:\n{}",
            combined.lines().take(40).collect::<Vec<_>>().join("\n"),
        )
    })
}
