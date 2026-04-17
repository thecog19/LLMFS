//! `ask` bridge — OpenAI-compatible chat client + tool-call dispatch.
//!
//! The model queries its own stored files via three tools defined in
//! DESIGN-NEW §8: `list_files`, `read_file`, `file_info`. Each tool call
//! maps directly to a `StegoDevice` / file-table method. The bridge runs
//! until the model produces an assistant message with no further tool
//! calls (or hits the iteration cap).

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::ask::AskError;
use crate::stego::device::StegoDevice;

/// Kept for backwards-compat with `bootstrap_smoke`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AskBridgeBootstrap {
    pub tool_count: usize,
}

impl Default for AskBridgeBootstrap {
    fn default() -> Self {
        Self { tool_count: 3 }
    }
}

pub const DEFAULT_SYSTEM_PROMPT: &str = "\
You are a helpful assistant with access to the files stored inside an \
LLMDB steganographic device. Use the provided tools (list_files, \
read_file, file_info) to answer questions about those files. Prefer \
tool calls over speculation; if a file is relevant, call read_file \
on it. Keep answers concise.";

pub const MAX_TOOL_ITERATIONS: usize = 8;

/// Upper bound on bytes returned from a single `read_file` tool call.
/// Prevents a 5 MB blob from blowing out the model's context; the model
/// can always follow up with another call. 16 KiB is enough for common
/// text files and still leaves context for reasoning.
pub const READ_FILE_MAX_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// JSON-encoded string (per the OpenAI API), not a parsed object.
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub struct ChatRequest<'a> {
    pub model: &'a str,
    pub messages: &'a [ChatMessage],
    pub tools: &'a [Value],
    #[serde(rename = "tool_choice")]
    pub tool_choice: &'a str,
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    pub message: ChatMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// Abstraction over the chat endpoint so `AskSession` is testable without
/// `llama-server` running. Production uses `HttpChatClient`; unit tests
/// plug in a mock that returns canned responses.
pub trait ChatClient {
    fn complete(&self, request: &ChatRequest<'_>) -> Result<ChatResponse, AskError>;
}

pub struct HttpChatClient {
    base_url: String,
}

impl HttpChatClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
        }
    }
}

impl ChatClient for HttpChatClient {
    fn complete(&self, request: &ChatRequest<'_>) -> Result<ChatResponse, AskError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = serde_json::to_string(request)?;
        let response = ureq::post(&url)
            .set("content-type", "application/json")
            .send_string(&body)
            .map_err(|e| AskError::HttpError(e.to_string()))?;
        let text = response
            .into_string()
            .map_err(|e| AskError::HttpError(e.to_string()))?;
        let parsed: ChatResponse = serde_json::from_str(&text).map_err(|_e| {
            AskError::MalformedResponse(format!(
                "could not parse chat response body: {}",
                truncate_for_log(&text, 400)
            ))
        })?;
        Ok(parsed)
    }
}

/// One interactive "question" ties to an `AskSession`. Messages persist
/// across calls so multi-turn conversations work naturally.
pub struct AskSession<'a, C: ChatClient> {
    client: C,
    device: &'a mut StegoDevice,
    model_name: String,
    messages: Vec<ChatMessage>,
    tools: Vec<Value>,
}

impl<'a, C: ChatClient> AskSession<'a, C> {
    pub fn new(client: C, device: &'a mut StegoDevice, model_name: impl Into<String>) -> Self {
        Self {
            client,
            device,
            model_name: model_name.into(),
            messages: vec![ChatMessage::system(DEFAULT_SYSTEM_PROMPT)],
            tools: tool_definitions(),
        }
    }

    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    pub fn ask(&mut self, question: &str) -> Result<String, AskError> {
        self.messages.push(ChatMessage::user(question));

        for _ in 0..MAX_TOOL_ITERATIONS {
            let request = ChatRequest {
                model: &self.model_name,
                messages: &self.messages,
                tools: &self.tools,
                tool_choice: "auto",
                stream: false,
            };
            let mut resp = self.client.complete(&request)?;
            let choice = resp.choices.pop().ok_or_else(|| {
                AskError::MalformedResponse("chat response had no choices".into())
            })?;
            let message = choice.message;

            // Record the assistant's move regardless.
            self.messages.push(message.clone());

            match message.tool_calls.as_deref() {
                Some(calls) if !calls.is_empty() => {
                    for call in calls {
                        let result = self.dispatch_tool_call(call)?;
                        self.messages
                            .push(ChatMessage::tool(call.id.clone(), result));
                    }
                    // Loop back: feed tool results to the model.
                }
                _ => {
                    return Ok(message.content.unwrap_or_default());
                }
            }
        }

        Err(AskError::ToolCallLimitExceeded {
            limit: MAX_TOOL_ITERATIONS,
        })
    }

    fn dispatch_tool_call(&mut self, call: &ToolCall) -> Result<String, AskError> {
        let args: Value = serde_json::from_str(&call.function.arguments)
            .unwrap_or(Value::Object(Default::default()));
        match call.function.name.as_str() {
            "list_files" => {
                let entries = self.device.list_files()?;
                let payload: Vec<_> = entries
                    .iter()
                    .map(|e| {
                        json!({
                            "name": e.filename,
                            "size_bytes": e.size_bytes,
                            "mode": format!("{:o}", e.mode),
                        })
                    })
                    .collect();
                Ok(serde_json::to_string(&payload)?)
            }
            "read_file" => {
                let name = args.get("name").and_then(Value::as_str).ok_or_else(|| {
                    AskError::InvalidToolCall("read_file requires a `name` string argument".into())
                })?;
                let mut bytes = self.device.read_file_bytes(name)?;
                let truncated = bytes.len() > READ_FILE_MAX_BYTES;
                if truncated {
                    bytes.truncate(READ_FILE_MAX_BYTES);
                }
                let text = String::from_utf8_lossy(&bytes).into_owned();
                let payload = json!({
                    "name": name,
                    "truncated": truncated,
                    "content": text,
                });
                Ok(serde_json::to_string(&payload)?)
            }
            "file_info" => {
                let name = args.get("name").and_then(Value::as_str).ok_or_else(|| {
                    AskError::InvalidToolCall("file_info requires a `name` string argument".into())
                })?;
                let entry = self
                    .device
                    .list_files()?
                    .into_iter()
                    .find(|e| e.filename == name)
                    .ok_or_else(|| {
                        AskError::Fs(crate::fs::file_ops::FsError::FileNotFound(name.into()))
                    })?;
                let payload = json!({
                    "name": entry.filename,
                    "size_bytes": entry.size_bytes,
                    "block_count": entry.block_count,
                    "mode": format!("{:o}", entry.mode),
                    "crc32": format!("{:08x}", entry.crc32),
                    "created": entry.created,
                    "modified": entry.modified,
                });
                Ok(serde_json::to_string(&payload)?)
            }
            other => Err(AskError::InvalidToolCall(format!("unknown tool: {other}"))),
        }
    }
}

pub fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "list_files",
                "description":
                    "List every regular file currently stored in the LLMDB \
                     device, with its size in bytes and POSIX mode.",
                "parameters": { "type": "object", "properties": {}, "required": [] },
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description":
                    "Return up to 16 KiB of the contents of a stored file as \
                     UTF-8 text. The result is truncated for files larger \
                     than the cap; use `file_info` first to check size.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "description": "file name exactly as listed" }
                    },
                    "required": ["name"],
                },
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "file_info",
                "description":
                    "Return metadata for a stored file: size, block count, \
                     mode, CRC32, created/modified timestamps.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "description": "file name exactly as listed" }
                    },
                    "required": ["name"],
                },
            }
        }),
    ]
}

fn truncate_for_log(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_owned()
    } else {
        format!("{}... ({}B total)", &s[..max], s.len())
    }
}
