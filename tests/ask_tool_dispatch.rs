//! Unit tests for the ask-bridge tool-call dispatch loop.
//!
//! These do not spawn `llama-server`. We plug a `MockChatClient` into
//! `AskSession` that plays back a canned sequence of chat responses. On
//! each response, the session either dispatches tool calls against the
//! real `StegoDevice` or returns a final assistant message. We verify
//! that (1) tool calls land on the right file-layer method, and (2) the
//! iteration cap is respected.

mod common;

use std::cell::RefCell;
use std::rc::Rc;

use common::{SyntheticGgufVersion, SyntheticTensorSpec, write_custom_gguf_fixture};
use llmdb::ask::AskError;
use llmdb::ask::bridge::{
    AskSession, ChatChoice, ChatClient, ChatMessage, ChatRequest, ChatResponse, MAX_TOOL_ITERATIONS,
    ToolCall, ToolCallFunction,
};
use llmdb::gguf::quant::GGML_TYPE_Q8_0_ID;
use llmdb::stego::device::{DeviceOptions, StegoDevice};
use llmdb::stego::planner::AllocationMode;

// -- Fixture + mock --

fn q8_tensors(count: usize) -> Vec<SyntheticTensorSpec> {
    (0..count)
        .map(|i| SyntheticTensorSpec {
            name: format!("blk.{}.ffn_down.weight", count - 1 - i),
            dimensions: vec![8192],
            raw_type_id: GGML_TYPE_Q8_0_ID,
            data: vec![0_u8; (8192 / 32) * 34],
        })
        .collect()
}

fn open_device(name: &str) -> (common::FixtureHandle, StegoDevice) {
    let fx = write_custom_gguf_fixture(SyntheticGgufVersion::V3, name, &q8_tensors(16));
    let device = StegoDevice::initialize_with_options(
        &fx.path,
        AllocationMode::Standard,
        DeviceOptions { verbose: false },
    )
    .expect("init");
    (fx, device)
}

/// Serves canned responses in order. Seen messages are stored behind an
/// `Rc<RefCell<...>>` so the test and the session can both hold handles
/// — `AskSession` takes the client by value.
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

// -- Tests --

#[test]
fn list_files_tool_call_reads_real_device_state() {
    let (_fx, mut device) = open_device("ask_list.gguf");
    device
        .store_bytes(b"first file", "alpha.txt", 0o644)
        .expect("store");
    device
        .store_bytes(b"second file", "beta.txt", 0o644)
        .expect("store");

    let client = MockChatClient::new(vec![
        tool_reply(vec![tool_call("c1", "list_files", "{}")]),
        final_reply("There are two files: alpha.txt and beta.txt."),
    ]);
    let seen = client.seen_handle();
    let mut session = AskSession::new(client, &mut device, "test-model");

    let answer = session.ask("what files are there?").expect("ask");
    assert!(answer.contains("alpha.txt"));
    assert!(answer.contains("beta.txt"));

    // The tool result was threaded back in as a `tool` message. We can
    // introspect by peeking at the mock's recorded requests.
    // Safety: we kept a raw pointer because session took the client by
    // value. This is a test-only, single-threaded borrow.
    let seen = seen.borrow();
    // The second round of input to the model must include a `tool` msg
    // carrying the JSON payload.
    assert!(seen.len() >= 2);
    let tool_msgs: Vec<&ChatMessage> = seen[1]
        .iter()
        .filter(|m| m.role == "tool")
        .collect();
    assert_eq!(tool_msgs.len(), 1);
    let content = tool_msgs[0].content.as_deref().unwrap_or_default();
    assert!(content.contains("alpha.txt"));
    assert!(content.contains("beta.txt"));
}

#[test]
fn read_file_tool_call_returns_stored_content() {
    let (_fx, mut device) = open_device("ask_read.gguf");
    device
        .store_bytes(b"the secret is 42", "secret.txt", 0o644)
        .expect("store");

    let client = MockChatClient::new(vec![
        tool_reply(vec![tool_call("c1", "read_file", r#"{"name":"secret.txt"}"#)]),
        final_reply("The secret is 42."),
    ]);
    let seen = client.seen_handle();
    let mut session = AskSession::new(client, &mut device, "test-model");
    let answer = session.ask("what's in secret.txt?").expect("ask");
    assert!(answer.contains("42"));

    let seen = seen.borrow();
    assert!(seen.len() >= 2);
    let tool_msg = seen[1]
        .iter()
        .find(|m| m.role == "tool")
        .expect("tool response msg");
    let content = tool_msg.content.as_deref().unwrap_or_default();
    assert!(content.contains("the secret is 42"));
}

#[test]
fn file_info_tool_call_returns_metadata_not_content() {
    let (_fx, mut device) = open_device("ask_info.gguf");
    device
        .store_bytes(b"payload contents", "report.bin", 0o600)
        .expect("store");

    let client = MockChatClient::new(vec![
        tool_reply(vec![tool_call("c1", "file_info", r#"{"name":"report.bin"}"#)]),
        final_reply("report.bin is 16 bytes."),
    ]);
    let seen = client.seen_handle();
    let mut session = AskSession::new(client, &mut device, "test-model");
    session.ask("tell me about report.bin").expect("ask");

    let seen = seen.borrow();
    let tool_msg = seen[1]
        .iter()
        .find(|m| m.role == "tool")
        .expect("tool msg");
    let content = tool_msg.content.as_deref().unwrap_or_default();
    assert!(content.contains("\"size_bytes\":16"));
    assert!(content.contains("\"mode\":\"600\""));
    // Must NOT leak the content bytes through file_info.
    assert!(!content.contains("payload contents"));
}

#[test]
fn tool_call_loop_caps_iterations() {
    let (_fx, mut device) = open_device("ask_cap.gguf");

    // The mock returns a tool call forever. Session should bail after
    // MAX_TOOL_ITERATIONS.
    let mut replies = Vec::new();
    for i in 0..(MAX_TOOL_ITERATIONS + 4) {
        replies.push(tool_reply(vec![tool_call(
            &format!("c{i}"),
            "list_files",
            "{}",
        )]));
    }
    let client = MockChatClient::new(replies);
    let mut session = AskSession::new(client, &mut device, "test-model");

    let result = session.ask("spin forever");
    assert!(
        matches!(result, Err(AskError::ToolCallLimitExceeded { .. })),
        "expected ToolCallLimitExceeded, got {result:?}"
    );
}

#[test]
fn invalid_tool_name_surfaces_as_error() {
    let (_fx, mut device) = open_device("ask_bad.gguf");

    let client = MockChatClient::new(vec![tool_reply(vec![tool_call(
        "c1",
        "not_a_tool",
        "{}",
    )])]);
    let mut session = AskSession::new(client, &mut device, "test-model");
    let result = session.ask("hi");
    assert!(
        matches!(result, Err(AskError::InvalidToolCall(_))),
        "expected InvalidToolCall, got {result:?}"
    );
}
