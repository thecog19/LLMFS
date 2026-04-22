//! Unit tests for the V2 ask-bridge tool-call dispatch loop.
//!
//! These do not spawn `llama-server`. We plug a `MockChatClient` into
//! `AskSession` that plays back a canned sequence of chat responses.
//! On each response, the session either dispatches tool calls against
//! a real `v2::Filesystem` or returns a final assistant message. We
//! verify that (1) tool calls land on the right fs method, (2) the
//! iteration cap is respected, and (3) tool payloads carry the right
//! shape (kinds, sizes, CRC32).

use std::cell::RefCell;
use std::rc::Rc;

use llmdb::ask::AskError;
use llmdb::ask::bridge::{
    AskSession, ChatChoice, ChatClient, ChatMessage, ChatRequest, ChatResponse,
    MAX_TOOL_ITERATIONS, ToolCall, ToolCallFunction,
};
use llmdb::gguf::quant::GgufQuantType;
use llmdb::stego::planner::TensorTier;
use llmdb::stego::tensor_map::{TensorMap, TensorSlot};
use llmdb::v2::cdc::FastCdcParams;
use llmdb::v2::fs::Filesystem;

// ─── F16 cover fixture (mirrors v2_fs_directory.rs) ────────────────────

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp32 = ((bits >> 23) & 0xFF) as i32;
    let mantissa32 = bits & 0x7FFFFF;
    if exp32 == 0 {
        return sign << 15;
    }
    let exp16 = exp32 - 127 + 15;
    if exp16 <= 0 {
        return sign << 15;
    }
    if exp16 >= 31 {
        return (sign << 15) | (0x1F << 10);
    }
    let mantissa16 = (mantissa32 >> 13) as u16;
    (sign << 15) | ((exp16 as u16) << 10) | mantissa16
}

fn make_cover() -> (Vec<u8>, TensorMap) {
    let weight_count = 30_000_u64;
    let values: Vec<f32> = (0..weight_count)
        .map(|i| {
            let sign = if i % 3 == 0 { -1.0 } else { 1.0 };
            sign * ((i + 1) as f32) * 0.00002
        })
        .collect();
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for v in &values {
        bytes.extend_from_slice(&f32_to_f16_bits(*v).to_le_bytes());
    }
    let bits = GgufQuantType::F16.stealable_bits_hint() as u64;
    let slot = TensorSlot {
        name: "ask.f16".to_owned(),
        quant_type: GgufQuantType::F16,
        tier: TensorTier::Tier1,
        data_offset: 0,
        weight_count,
        stealable_bits_per_weight: bits as usize,
        capacity_bits: weight_count * bits,
        bit_start: 0,
        bit_end: weight_count * bits,
    };
    let map = TensorMap {
        slots: vec![slot.clone()],
        total_capacity_bits: slot.capacity_bits,
        total_capacity_bytes: slot.capacity_bits / 8,
    };
    (bytes, map)
}

fn small_cdc() -> FastCdcParams {
    FastCdcParams {
        min_size: 32,
        avg_size: 64,
        max_size: 128,
    }
}

/// Build a V2 fs with a small canned tree:
///
/// ```text
/// /alpha.txt          "first file"
/// /beta.txt           "second file"
/// /docs/notes.md      "important notes"
/// ```
fn make_fs_with_files() -> Filesystem {
    let (cover, map) = make_cover();
    let mut fs = Filesystem::init_with_cdc_params(cover, map, small_cdc()).expect("init");
    fs.create_file("/alpha.txt", b"first file").expect("alpha");
    fs.create_file("/beta.txt", b"second file").expect("beta");
    fs.mkdir("/docs").expect("mkdir docs");
    fs.create_file("/docs/notes.md", b"important notes")
        .expect("notes");
    fs
}

// ─── Mock chat client ──────────────────────────────────────────────────

struct MockChatClient {
    replies: RefCell<Vec<ChatResponse>>,
    seen_messages: Rc<RefCell<Vec<Vec<ChatMessage>>>>,
}

impl MockChatClient {
    fn new(replies: Vec<ChatResponse>) -> Self {
        Self {
            replies: RefCell::new(replies),
            seen_messages: Rc::new(RefCell::new(Vec::new())),
        }
    }

    fn seen_handle(&self) -> Rc<RefCell<Vec<Vec<ChatMessage>>>> {
        Rc::clone(&self.seen_messages)
    }
}

impl ChatClient for MockChatClient {
    fn complete(&self, request: &ChatRequest<'_>) -> Result<ChatResponse, AskError> {
        self.seen_messages
            .borrow_mut()
            .push(request.messages.to_vec());
        let mut replies = self.replies.borrow_mut();
        if replies.is_empty() {
            return Err(AskError::MalformedResponse("mock out of replies".into()));
        }
        Ok(replies.remove(0))
    }
}

fn tool_call(id: &str, name: &str, arguments: &str) -> ToolCall {
    ToolCall {
        id: id.into(),
        kind: "function".into(),
        function: ToolCallFunction {
            name: name.into(),
            arguments: arguments.into(),
        },
    }
}

fn tool_reply(calls: Vec<ToolCall>) -> ChatResponse {
    ChatResponse {
        choices: vec![ChatChoice {
            message: ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_calls: Some(calls),
                tool_call_id: None,
            },
            finish_reason: Some("tool_calls".into()),
        }],
    }
}

fn final_reply(content: &str) -> ChatResponse {
    ChatResponse {
        choices: vec![ChatChoice {
            message: ChatMessage {
                role: "assistant".into(),
                content: Some(content.into()),
                tool_calls: None,
                tool_call_id: None,
            },
            finish_reason: Some("stop".into()),
        }],
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[test]
fn ls_at_root_returns_files_and_directories() {
    let mut fs = make_fs_with_files();

    let client = MockChatClient::new(vec![
        tool_reply(vec![tool_call("c1", "ls", r#"{"path":"/"}"#)]),
        final_reply("Top-level: alpha.txt, beta.txt, docs/."),
    ]);
    let seen = client.seen_handle();
    let mut session = AskSession::new(client, &mut fs, "test-model");

    let _ = session.ask("what's at the root?").expect("ask");

    let seen = seen.borrow();
    assert!(seen.len() >= 2);
    let tool_msg = seen[1].iter().find(|m| m.role == "tool").expect("tool msg");
    let content = tool_msg.content.as_deref().unwrap_or_default();
    assert!(
        content.contains("\"name\":\"alpha.txt\"") && content.contains("\"kind\":\"file\""),
        "missing alpha.txt file entry: {content}"
    );
    assert!(
        content.contains("\"name\":\"beta.txt\""),
        "missing beta.txt: {content}"
    );
    assert!(
        content.contains("\"name\":\"docs\"") && content.contains("\"kind\":\"directory\""),
        "missing docs directory entry: {content}"
    );
}

#[test]
fn ls_at_subdirectory_returns_nested_entries() {
    let mut fs = make_fs_with_files();

    let client = MockChatClient::new(vec![
        tool_reply(vec![tool_call("c1", "ls", r#"{"path":"/docs"}"#)]),
        final_reply("docs has notes.md."),
    ]);
    let seen = client.seen_handle();
    let mut session = AskSession::new(client, &mut fs, "test-model");

    let _ = session.ask("what's in /docs?").expect("ask");

    let seen = seen.borrow();
    let tool_msg = seen[1].iter().find(|m| m.role == "tool").expect("tool msg");
    let content = tool_msg.content.as_deref().unwrap_or_default();
    assert!(
        content.contains("\"name\":\"notes.md\""),
        "missing notes.md: {content}"
    );
    // notes.md is "important notes" = 15 bytes
    assert!(
        content.contains("\"size_bytes\":15"),
        "wrong size for notes.md: {content}"
    );
}

#[test]
fn read_returns_file_content_for_nested_path() {
    let mut fs = make_fs_with_files();

    let client = MockChatClient::new(vec![
        tool_reply(vec![tool_call(
            "c1",
            "read",
            r#"{"path":"/docs/notes.md"}"#,
        )]),
        final_reply("The notes say: important notes."),
    ]);
    let seen = client.seen_handle();
    let mut session = AskSession::new(client, &mut fs, "test-model");

    let _ = session.ask("read notes.md").expect("ask");

    let seen = seen.borrow();
    let tool_msg = seen[1].iter().find(|m| m.role == "tool").expect("tool msg");
    let content = tool_msg.content.as_deref().unwrap_or_default();
    assert!(content.contains("important notes"), "missing payload: {content}");
    assert!(content.contains("\"truncated\":false"), "should not truncate small file: {content}");
}

#[test]
fn stat_on_file_returns_size_and_crc32() {
    let mut fs = make_fs_with_files();

    let client = MockChatClient::new(vec![
        tool_reply(vec![tool_call("c1", "stat", r#"{"path":"/alpha.txt"}"#)]),
        final_reply("alpha is 10 bytes."),
    ]);
    let seen = client.seen_handle();
    let mut session = AskSession::new(client, &mut fs, "test-model");

    let _ = session.ask("stat alpha").expect("ask");

    let seen = seen.borrow();
    let tool_msg = seen[1].iter().find(|m| m.role == "tool").expect("tool msg");
    let content = tool_msg.content.as_deref().unwrap_or_default();

    let expected_crc = format!("{:08x}", crc32fast::hash(b"first file"));
    assert!(
        content.contains("\"kind\":\"file\""),
        "wrong kind: {content}"
    );
    assert!(
        content.contains("\"size_bytes\":10"),
        "wrong size: {content}"
    );
    assert!(
        content.contains(&format!("\"crc32\":\"{expected_crc}\"")),
        "wrong crc32 (expected {expected_crc}): {content}"
    );
    // stat must not leak content bytes.
    assert!(
        !content.contains("first file"),
        "stat leaked file contents: {content}"
    );
}

#[test]
fn stat_on_directory_omits_crc32() {
    let mut fs = make_fs_with_files();

    let client = MockChatClient::new(vec![
        tool_reply(vec![tool_call("c1", "stat", r#"{"path":"/docs"}"#)]),
        final_reply("docs is a directory."),
    ]);
    let seen = client.seen_handle();
    let mut session = AskSession::new(client, &mut fs, "test-model");

    let _ = session.ask("stat docs").expect("ask");

    let seen = seen.borrow();
    let tool_msg = seen[1].iter().find(|m| m.role == "tool").expect("tool msg");
    let content = tool_msg.content.as_deref().unwrap_or_default();
    assert!(content.contains("\"kind\":\"directory\""), "wrong kind: {content}");
    assert!(!content.contains("crc32"), "crc32 should be absent for dirs: {content}");
}

#[test]
fn list_all_files_returns_every_file_recursively() {
    let mut fs = make_fs_with_files();

    let client = MockChatClient::new(vec![
        tool_reply(vec![tool_call("c1", "list_all_files", "{}")]),
        final_reply("There are 3 files."),
    ]);
    let seen = client.seen_handle();
    let mut session = AskSession::new(client, &mut fs, "test-model");

    let _ = session.ask("list everything").expect("ask");

    let seen = seen.borrow();
    let tool_msg = seen[1].iter().find(|m| m.role == "tool").expect("tool msg");
    let content = tool_msg.content.as_deref().unwrap_or_default();
    for path in ["/alpha.txt", "/beta.txt", "/docs/notes.md"] {
        assert!(
            content.contains(&format!("\"path\":\"{path}\"")),
            "missing path {path} in: {content}"
        );
    }
    // Directories are not themselves listed.
    assert!(
        !content.contains("\"path\":\"/docs\""),
        "list_all_files should omit directories: {content}"
    );
}

#[test]
fn tool_call_loop_caps_iterations() {
    let mut fs = make_fs_with_files();

    // The mock returns a tool call forever. Session should bail after
    // MAX_TOOL_ITERATIONS.
    let mut replies = Vec::new();
    for i in 0..(MAX_TOOL_ITERATIONS + 4) {
        replies.push(tool_reply(vec![tool_call(
            &format!("c{i}"),
            "ls",
            r#"{"path":"/"}"#,
        )]));
    }
    let client = MockChatClient::new(replies);
    let mut session = AskSession::new(client, &mut fs, "test-model");

    let result = session.ask("spin forever");
    assert!(
        matches!(result, Err(AskError::ToolCallLimitExceeded { .. })),
        "expected ToolCallLimitExceeded, got {result:?}"
    );
}

#[test]
fn invalid_tool_name_surfaces_as_error() {
    let mut fs = make_fs_with_files();

    let client = MockChatClient::new(vec![tool_reply(vec![tool_call("c1", "not_a_tool", "{}")])]);
    let mut session = AskSession::new(client, &mut fs, "test-model");
    let result = session.ask("hi");
    assert!(
        matches!(result, Err(AskError::InvalidToolCall(_))),
        "expected InvalidToolCall, got {result:?}"
    );
}

#[test]
fn ls_with_missing_path_argument_errors() {
    let mut fs = make_fs_with_files();

    let client = MockChatClient::new(vec![tool_reply(vec![tool_call("c1", "ls", "{}")])]);
    let mut session = AskSession::new(client, &mut fs, "test-model");
    let result = session.ask("ls");
    assert!(
        matches!(result, Err(AskError::InvalidToolCall(_))),
        "expected InvalidToolCall, got {result:?}"
    );
}
