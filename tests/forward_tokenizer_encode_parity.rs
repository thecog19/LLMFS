//! A2d — encode parity with llama-tokenize across a diverse input set.
//!
//! Two tests:
//!
//! - `smollm2_encode_matches_checkpointed_fixtures` (fast, always
//!   runs when the model is present). Reads
//!   `tests/fixtures/tokenizer/smollm2_encode.json` and diffs our
//!   encode output against the checkpointed token ids. This is the
//!   default correctness gate for A2. No external binary required.
//! - `regen_smollm2_encode_fixtures` (`#[ignore]`'d). Spawns
//!   `llama-tokenize` for each input in the fixture list, parses
//!   its output, and rewrites the JSON. Run with
//!   `cargo test --test forward_tokenizer_encode_parity -- --ignored
//!    --nocapture` when the input list changes or the GGUF is
//!   refreshed. Requires `llama-tokenize` at `$LLMDB_LLAMA_TOKENIZE`
//!   or on `PATH`.
//!
//! Fixture format: a JSON array of `{"input": "<s>", "ids":
//! [<u32>...]}` records. See the regen test for the full input
//! catalogue (30 diverse strings).

use std::path::{Path, PathBuf};
use std::process::Command;

use llmdb::forward::tokenizer::Tokenizer;
use llmdb::gguf::parser::parse_path;
use serde::{Deserialize, Serialize};

const SMOLLM2_GGUF: &str = "models/smollm2-135m-f16.gguf";

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tokenizer/smollm2_encode.json")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Case {
    /// Human-readable label so the fixture file is scannable and
    /// test failures name *which* case failed, not just the
    /// (possibly non-printable) input bytes.
    label: String,
    input: String,
    ids: Vec<u32>,
}

fn tokenizer_or_skip() -> Option<Tokenizer> {
    if !Path::new(SMOLLM2_GGUF).exists() {
        eprintln!("skipping: {SMOLLM2_GGUF} not present");
        return None;
    }
    let gguf = parse_path(SMOLLM2_GGUF).expect("parse gguf");
    Some(Tokenizer::from_gguf(&gguf).expect("build tokenizer"))
}

#[test]
fn smollm2_encode_matches_checkpointed_fixtures() {
    let Some(t) = tokenizer_or_skip() else {
        return;
    };

    let path = fixture_path();
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "read fixture {}: {e}; regen with \
             `cargo test --test forward_tokenizer_encode_parity \
             -- --ignored --nocapture regen_`",
            path.display(),
        );
    });
    let cases: Vec<Case> = serde_json::from_str(&raw).expect("parse fixture JSON");
    assert!(!cases.is_empty(), "fixture file is empty");

    let mut failures: Vec<String> = Vec::new();
    for case in &cases {
        match t.encode(&case.input) {
            Ok(got) if got == case.ids => {}
            Ok(got) => failures.push(format!(
                "[{}] input={:?}\n      got:  {:?}\n     want:  {:?}",
                case.label, case.input, got, case.ids,
            )),
            Err(e) => failures.push(format!(
                "[{}] input={:?}: encode errored: {e}",
                case.label, case.input,
            )),
        }
    }
    if !failures.is_empty() {
        panic!(
            "{}/{} cases failed parity:\n{}",
            failures.len(),
            cases.len(),
            failures.join("\n"),
        );
    }
}

// ─── Regen ────────────────────────────────────────────────────────────────

/// Inputs captured for the encode-parity fixture. Diverse by
/// construction: single chars, punctuation, digit runs, unicode,
/// whitespace edge cases, contractions, code-shaped text, URLs.
/// Keep this list authoritative — the JSON in
/// `tests/fixtures/tokenizer/smollm2_encode.json` is *derived
/// from* this list via the regen test.
const REGEN_INPUTS: &[(&str, &str)] = &[
    ("empty", ""),
    ("single_letter", "a"),
    ("single_digit", "5"),
    ("single_space", " "),
    ("single_period", "."),
    ("hello_world_punct", "Hello, world!"),
    ("digit_isolation", "abc123def"),
    ("contraction_dont", "don't worry"),
    ("mixed_stress", "It's 3:14pm"),
    ("multi_space", "hello   world"),
    ("long_digit_run", "1234567890"),
    ("all_caps", "HELLO WORLD"),
    ("camel_case", "camelCaseIsCool"),
    ("snake_case", "snake_case_variable"),
    ("kebab_case", "kebab-case-too"),
    ("accented_latin", "Résumé"),
    ("accented_small", "café"),
    ("tab_char", "a\tb"),
    ("newline", "a\nb"),
    ("multi_line", "first line\nsecond line"),
    ("long_sentence", "The quick brown fox jumps over the lazy dog"),
    ("double_quotes", "She said, \"hello\"."),
    ("code_let", "let x = 42;"),
    ("url_http", "https://example.com"),
    ("decimal_pi", "3.14159"),
    ("negative_int", "-42"),
    ("scientific", "1.2e-10"),
    ("brackets", "(a) [b] {c}"),
    ("markdown_heading", "# heading"),
    ("a1b2", "a1 b2"),
];

#[test]
#[ignore = "regen: spawns llama-tokenize per input; run with --ignored"]
fn regen_smollm2_encode_fixtures() {
    if !Path::new(SMOLLM2_GGUF).exists() {
        eprintln!("skipping regen: {SMOLLM2_GGUF} not present");
        return;
    }
    let binary = std::env::var("LLMDB_LLAMA_TOKENIZE")
        .unwrap_or_else(|_| "llama-tokenize".to_owned());

    let mut cases: Vec<Case> = Vec::with_capacity(REGEN_INPUTS.len());
    for (label, input) in REGEN_INPUTS {
        let ids = run_llama_tokenize(&binary, SMOLLM2_GGUF, input).unwrap_or_else(|e| {
            panic!("llama-tokenize {label:?} ({input:?}): {e}");
        });
        eprintln!("  [{label:>20}]  ids.len={:>3}  {input:?}", ids.len());
        cases.push(Case {
            label: label.to_string(),
            input: input.to_string(),
            ids,
        });
    }

    let path = fixture_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("mkdir fixture dir");
    }
    let json = serde_json::to_string_pretty(&cases).expect("serialize fixtures");
    std::fs::write(&path, json).expect("write fixture");
    eprintln!("wrote {} cases to {}", cases.len(), path.display());
}

/// Invoke `llama-tokenize -m <gguf> --log-disable -p <input>` and
/// parse the `id -> 'text'` output format into a `Vec<u32>`.
fn run_llama_tokenize(binary: &str, gguf: &str, input: &str) -> Result<Vec<u32>, String> {
    let out = Command::new(binary)
        .arg("-m")
        .arg(gguf)
        .arg("--log-disable")
        .arg("--no-parse-special")
        .arg("-p")
        .arg(input)
        .output()
        .map_err(|e| format!("spawn: {e}"))?;
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
        // Output format (per `llama-tokenize --log-disable`):
        //   "  <id> -> '<text>'"
        // with leading whitespace. Skip any line that doesn't
        // match this shape.
        let line = line.trim_start();
        let Some((id_part, _rest)) = line.split_once(" -> '") else {
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
