//! End-to-end test for the V2 `ask` bridge against a real llama-server.
//!
//! Gated by `LLMDB_E2E_ASK=1` so the default `cargo test` run doesn't
//! depend on a compiled llama.cpp. When enabled, the test:
//!
//! 1. locates `llama-server` (env `LLMDB_LLAMA_SERVER` or PATH),
//! 2. locates an F16 GGUF (env `LLMDB_E2E_GGUF` or
//!    `models/smollm2-135m-f16.gguf`) — must be F16/F32 so the model
//!    still inferences after V2 init,
//! 3. copies the GGUF to a tempdir, builds a V2 filesystem on it via
//!    the public V2 API, writes two files with known contents,
//!    persists the cover bytes back to disk,
//! 4. spawns `LlamaServer` against the modified GGUF, waits for `/health`,
//! 5. drives `AskSession` with a prompt that the model should answer
//!    by calling `list_all_files` + `read`,
//! 6. asserts that at least one tool call happened and that the
//!    stored passphrase appears somewhere in the transcript.
//!
//! Runtime is dominated by model load + a handful of tokens, so the
//! test takes ~10-30 s with smollm2-135m-f16 on CPU.

use std::net::TcpListener;
use std::path::{Path, PathBuf};

use llmdb::ask::bridge::{AskSession, HttpChatClient};
use llmdb::ask::server::LlamaServer;
use llmdb::gguf::parser::parse_path as parse_gguf;
use llmdb::stego::planner::{AllocationMode, build_allocation_plan};
use llmdb::stego::tensor_map::TensorMap;
use llmdb::v2::cover::CoverStorage;
use llmdb::v2::fs::Filesystem as V2Filesystem;

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

/// Open a cover, build the V2 tensor map, return the cover bytes
/// alongside it. Used both before and after writing files.
fn load_cover(model: &Path) -> (Vec<u8>, TensorMap) {
    let parsed = parse_gguf(model).expect("parse gguf");
    let plan = build_allocation_plan(&parsed.tensors, AllocationMode::Standard);
    let map = TensorMap::from_allocation_plan_with_base(&plan, parsed.tensor_data_offset as u64);
    let cover = std::fs::read(model).expect("read cover");
    (cover, map)
}

fn prepare_v2_fs(src: &Path) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let copy = dir.path().join("ask_e2e.gguf");
    std::fs::copy(src, &copy).expect("copy model");

    // Init the V2 filesystem on the copy, write two files, persist.
    let (cover, map) = load_cover(&copy);
    let mut fs = V2Filesystem::init(cover, map).expect("v2 init");
    fs.create_file(
        "/passphrase.txt",
        b"The secret passphrase is PURPLE ELEPHANT.\n",
    )
    .expect("create passphrase.txt");
    fs.create_file(
        "/recipe.txt",
        b"Ingredients:\n- 200g flour\n- 1 banana\n- 2 eggs\n",
    )
    .expect("create recipe.txt");
    let cover_after = fs.unmount().expect("unmount");
    std::fs::write(&copy, cover_after.bytes()).expect("write cover back");

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

    let (_dir, stego_path) = prepare_v2_fs(&model);
    let port = pick_free_port();

    let server = LlamaServer::spawn(&stego_path, port).expect("spawn llama-server");
    assert_eq!(server.port(), port);

    // Re-mount the V2 filesystem for the session — same cover bytes
    // we just wrote, but loaded fresh so the test exercises the
    // mount path.
    let (cover, map) = load_cover(&stego_path);
    let mut fs = V2Filesystem::mount(cover, map).expect("v2 mount for session");

    let client = HttpChatClient::new(server.base_url());
    let mut session = AskSession::new(client, &mut fs, "ask-e2e");

    let answer = session
        .ask(
            "Call the `read` tool on the file at path \"/passphrase.txt\" \
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

    // Either `read` (preferred) or `list_all_files` would suffice for
    // the model to surface the passphrase content. Accept either.
    let called_names: Vec<&str> = tool_invocations.iter().map(|(n, _)| *n).collect();
    assert!(
        called_names.iter().any(|n| matches!(*n, "read" | "list_all_files" | "ls" | "stat")),
        "expected a fs-tool call (read/ls/stat/list_all_files); \
         got {called_names:?}. final answer: {answer:?}"
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

    eprintln!(
        "ask_e2e: tools called: {called_names:?}; \
         final answer length = {} bytes",
        answer.len()
    );
}
