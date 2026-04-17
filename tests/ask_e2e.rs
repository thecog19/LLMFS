//! End-to-end test for the `ask` bridge against a real llama-server.
//!
//! Gated by `LLMDB_E2E_ASK=1` so the default `cargo test` run doesn't
//! depend on a compiled llama.cpp. When enabled, the test:
//!
//! 1. locates `llama-server` (env `LLMDB_LLAMA_SERVER` or PATH),
//! 2. locates an F16 GGUF (env `LLMDB_E2E_GGUF` or
//!    `models/smollm2-135m-f16.gguf`) — must be F16/F32 so the model
//!    still inferences after `init` (see DESIGN-NEW §2),
//! 3. copies the GGUF to a tempdir, runs `llmdb init`, stores two
//!    text files with known contents,
//! 4. spawns `LlamaServer`, waits for `/health`,
//! 5. drives `AskSession` with a prompt that the model should answer
//!    by calling `list_files` + `read_file`,
//! 6. asserts that at least one tool call happened and that the
//!    stored files' names appear in the transcript.
//!
//! Runtime is dominated by model load + a handful of tokens, so the
//! test takes ~10-30 s with smollm2-135m-f16 on CPU.

use std::net::TcpListener;
use std::path::{Path, PathBuf};

use llmdb::ask::bridge::{AskSession, HttpChatClient};
use llmdb::ask::server::LlamaServer;
use llmdb::stego::device::{DeviceOptions, StegoDevice};
use llmdb::stego::planner::AllocationMode;

fn gated() -> bool {
    std::env::var("LLMDB_E2E_ASK").ok().as_deref() == Some("1")
}

fn pick_free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    l.local_addr().unwrap().port()
}

fn locate_model() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("LLMDB_E2E_GGUF") {
        let path = PathBuf::from(p);
        return path.exists().then_some(path);
    }
    let fallback = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("smollm2-135m-f16.gguf");
    fallback.exists().then_some(fallback)
}

fn prepare_device(src: &Path) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let copy = dir.path().join("ask_e2e.gguf");
    std::fs::copy(src, &copy).expect("copy model");

    let mut device = StegoDevice::initialize_with_options(
        &copy,
        AllocationMode::Standard,
        DeviceOptions { verbose: false },
    )
    .expect("init device");

    device
        .store_bytes(
            b"The secret passphrase is PURPLE ELEPHANT.\n",
            "passphrase.txt",
            0o644,
        )
        .expect("store passphrase.txt");
    device
        .store_bytes(
            b"Ingredients:\n- 200g flour\n- 1 banana\n- 2 eggs\n",
            "recipe.txt",
            0o644,
        )
        .expect("store recipe.txt");

    device.close().expect("close");
    (dir, copy)
}

#[test]
fn ask_session_drives_tool_calls_against_real_llama_server() {
    if !gated() {
        eprintln!("LLMDB_E2E_ASK!=1; skipping ask_e2e (set LLMDB_E2E_ASK=1 to run)");
        return;
    }
    let Some(model) = locate_model() else {
        panic!(
            "no F16 GGUF found: set LLMDB_E2E_GGUF or place one at \
             models/smollm2-135m-f16.gguf"
        );
    };

    let (_dir, stego_path) = prepare_device(&model);
    let port = pick_free_port();

    let server = LlamaServer::spawn(&stego_path, port).expect("spawn llama-server");
    assert_eq!(server.port(), port);

    let mut device = StegoDevice::open_with_options(
        &stego_path,
        AllocationMode::Standard,
        DeviceOptions { verbose: false },
    )
    .expect("reopen stego device for session");

    let client = HttpChatClient::new(server.base_url());
    let mut session = AskSession::new(client, &mut device, "ask-e2e");

    let answer = session
        .ask(
            "Call the read_file tool on the file named \"passphrase.txt\" \
             and return its contents verbatim in your reply. Do not \
             paraphrase.",
        )
        .expect("session.ask");

    let tool_invocations: Vec<(&str, &str)> = session
        .messages()
        .iter()
        .filter_map(|m| m.tool_calls.as_ref())
        .flat_map(|calls| calls.iter())
        .map(|c| (c.function.name.as_str(), c.function.arguments.as_str()))
        .collect();
    assert!(
        !tool_invocations.is_empty(),
        "expected at least one tool call in the transcript; got none. \
         final answer was: {answer:?}"
    );
    assert!(
        tool_invocations
            .iter()
            .any(|(name, _)| *name == "read_file"),
        "expected a read_file call; got {tool_invocations:?}. \
         final answer: {answer:?}"
    );

    // Tool result containing the passphrase must have reached the
    // model (evidenced by it appearing in the transcript — either in
    // the tool-response message we injected, or in the final answer).
    let transcript: String = session
        .messages()
        .iter()
        .filter_map(|m| m.content.clone())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        transcript.contains("PURPLE ELEPHANT"),
        "expected passphrase content to appear in the transcript:\n{transcript}"
    );

    let called_names: Vec<&str> = tool_invocations.iter().map(|(n, _)| *n).collect();
    eprintln!(
        "ask_e2e: tools called: {called_names:?}; \
         final answer length = {} bytes",
        answer.len()
    );
}
