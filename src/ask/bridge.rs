//! `ask` bridge — OpenAI-compatible chat client + tool-call dispatch.
//!
//! The model queries its own stored files via four tools:
//!
//! - `ls(path)` — directory listing: entries with kind + size.
//! - `read(path)` — file contents, truncated to [`READ_FILE_MAX_BYTES`].
//! - `stat(path)` — metadata for a single path: kind, size, CRC32.
//!   The CRC is computed on the fly from the file bytes (V2 stores
//!   only `length + pointers` per inode).
//! - `list_all_files()` — recursive walk of every file in the tree,
//!   returned as `[{path, size_bytes}]`. Convenient for small models
//!   that struggle to plan multi-step `ls` navigation.
//!
//! Each tool call maps directly to a [`V2Filesystem`] method. The
//! bridge runs until the model produces an assistant message with no
//! further tool calls, or hits [`MAX_TOOL_ITERATIONS`].

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::ask::AskError;
use crate::v2::directory::EntryKind;
use crate::v2::fs::Filesystem as V2Filesystem;

pub const DEFAULT_SYSTEM_PROMPT: &str = "\
You are a helpful assistant with access to the files stored inside an \
LLMDB steganographic V2 filesystem. The filesystem is hierarchical — \
paths are absolute and start with '/'. Use the provided tools \
(ls, read, stat, list_all_files) to answer questions about those \
files. Prefer tool calls over speculation; if a file is relevant, \
call `read` on it. For a broad survey, `list_all_files` returns \
every file in one shot. Keep answers concise.";

pub const MAX_TOOL_ITERATIONS: usize = 8;

/// Upper bound on bytes returned from a single `read` tool call.
/// Prevents a 5 MB blob from blowing out the model's context; the
/// model can always follow up with another call. 16 KiB is enough for
/// common text files and still leaves context for reasoning.
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

/// Abstraction over the chat endpoint so `AskSession` is testable
/// without `llama-server` running. Production uses `HttpChatClient`;
/// unit tests plug in a mock that returns canned responses.
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
    fs: &'a mut V2Filesystem,
    model_name: String,
    messages: Vec<ChatMessage>,
    tools: Vec<Value>,
}

impl<'a, C: ChatClient> AskSession<'a, C> {
    pub fn new(client: C, fs: &'a mut V2Filesystem, model_name: impl Into<String>) -> Self {
        Self {
            client,
            fs,
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
            "ls" => {
                let path = required_path(&args, "ls")?;
                let entries = self.fs.readdir(path)?;
                let mut payload = Vec::with_capacity(entries.len());
                for entry in &entries {
                    let inode = self.fs.inode_at(&join_path(path, &entry.name))?;
                    payload.push(json!({
                        "name": entry.name,
                        "kind": entry_kind_str(entry.kind),
                        "size_bytes": inode.length,
                    }));
                }
                Ok(serde_json::to_string(&payload)?)
            }
            "read" => {
                let path = required_path(&args, "read")?;
                let mut bytes = self.fs.read_file(path)?;
                let truncated = bytes.len() > READ_FILE_MAX_BYTES;
                if truncated {
                    bytes.truncate(READ_FILE_MAX_BYTES);
                }
                let text = String::from_utf8_lossy(&bytes).into_owned();
                let payload = json!({
                    "path": path,
                    "truncated": truncated,
                    "content": text,
                });
                Ok(serde_json::to_string(&payload)?)
            }
            "stat" => {
                let path = required_path(&args, "stat")?;
                // Classify: directory (listable) vs file (readable). `/`
                // is always a directory; for any other path we look at
                // its parent's entry for the kind.
                let kind = path_kind(self.fs, path)?;
                let inode = self.fs.inode_at(path)?;
                let payload = match kind {
                    EntryKind::Directory => json!({
                        "path": path,
                        "kind": "directory",
                        "size_bytes": inode.length,
                    }),
                    EntryKind::File => {
                        let bytes = self.fs.read_file(path)?;
                        let crc = crc32fast::hash(&bytes);
                        json!({
                            "path": path,
                            "kind": "file",
                            "size_bytes": inode.length,
                            "crc32": format!("{crc:08x}"),
                        })
                    }
                };
                Ok(serde_json::to_string(&payload)?)
            }
            "list_all_files" => {
                let mut out = Vec::new();
                walk_files(self.fs, "/", &mut out)?;
                let payload: Vec<_> = out
                    .into_iter()
                    .map(|(path, size)| {
                        json!({
                            "path": path,
                            "size_bytes": size,
                        })
                    })
                    .collect();
                Ok(serde_json::to_string(&payload)?)
            }
            other => Err(AskError::InvalidToolCall(format!("unknown tool: {other}"))),
        }
    }
}

fn required_path<'a>(args: &'a Value, tool: &str) -> Result<&'a str, AskError> {
    args.get("path").and_then(Value::as_str).ok_or_else(|| {
        AskError::InvalidToolCall(format!("{tool} requires a `path` string argument"))
    })
}

fn entry_kind_str(kind: EntryKind) -> &'static str {
    match kind {
        EntryKind::File => "file",
        EntryKind::Directory => "directory",
    }
}

/// Classify `path` as file or directory. `/` is always a directory.
/// Otherwise inspect the parent directory's entry for the leaf.
fn path_kind(fs: &V2Filesystem, path: &str) -> Result<EntryKind, AskError> {
    if path == "/" {
        return Ok(EntryKind::Directory);
    }
    let (parent, leaf) = split_parent_leaf(path);
    for entry in fs.readdir(parent)? {
        if entry.name == leaf {
            return Ok(entry.kind);
        }
    }
    Err(AskError::V2Fs(crate::v2::fs::FsError::PathNotFound(
        path.to_owned(),
    )))
}

/// Split `/a/b/c` → (`/a/b`, `c`); split `/x` → (`/`, `x`).
/// Assumes `path` is absolute and not `/` (caller handles the root).
fn split_parent_leaf(path: &str) -> (&str, &str) {
    let trimmed = path.trim_end_matches('/');
    match trimmed.rfind('/') {
        Some(0) => ("/", &trimmed[1..]),
        Some(i) => (&trimmed[..i], &trimmed[i + 1..]),
        // No leading slash — treat as leaf-only; parent is root.
        None => ("/", trimmed),
    }
}

fn join_path(dir: &str, name: &str) -> String {
    if dir == "/" {
        format!("/{name}")
    } else {
        format!("{}/{}", dir.trim_end_matches('/'), name)
    }
}

/// Walk the tree depth-first, collecting every file's path + size.
/// Directories are recursed into but not themselves reported.
fn walk_files(fs: &V2Filesystem, dir: &str, out: &mut Vec<(String, u64)>) -> Result<(), AskError> {
    for entry in fs.readdir(dir)? {
        let child_path = join_path(dir, &entry.name);
        match entry.kind {
            EntryKind::File => {
                let inode = fs.inode_at(&child_path)?;
                out.push((child_path, inode.length));
            }
            EntryKind::Directory => {
                walk_files(fs, &child_path, out)?;
            }
        }
    }
    Ok(())
}

pub fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "ls",
                "description":
                    "List the entries in an absolute directory path. \
                     Returns an array of {name, kind, size_bytes}. \
                     `kind` is 'file' or 'directory'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "absolute path; '/' for the root" }
                    },
                    "required": ["path"],
                },
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "read",
                "description":
                    "Return up to 16 KiB of the contents of a file at an \
                     absolute path, as UTF-8 text. The result is \
                     truncated for larger files; call `stat` first to \
                     check size.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "absolute file path" }
                    },
                    "required": ["path"],
                },
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "stat",
                "description":
                    "Return metadata for a file or directory: kind, \
                     size_bytes, and (for files) a hex CRC32 of the \
                     contents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "absolute path" }
                    },
                    "required": ["path"],
                },
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "list_all_files",
                "description":
                    "Recursively list every file in the filesystem. \
                     Returns [{path, size_bytes}]. Directories are \
                     traversed but not themselves listed.",
                "parameters": { "type": "object", "properties": {}, "required": [] },
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
