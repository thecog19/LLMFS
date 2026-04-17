# Task 13: Ask Bridge (llama-server + Tool Use)

Status: done (unit path; E2E with real llama-server gated on `LLMDB_E2E_ASK`, not verified locally because the binary isn't on PATH here)
Depends on: 09-file-table-and-file-ops.md, 10-cli-file-commands.md
Spec refs: DESIGN-NEW.MD section "8. `ask` Command"

Objective:
Implement `llmdb ask`: spawn `llama-server` as a subprocess, connect to its
OpenAI-compatible chat API, register tool definitions for file access, and
let the model query its own stored files. This is the "the model reads data
from its own weights" milestone.

Scope:

- Create `src/ask/server.rs`:
  - `struct LlamaServer { process: Child, port: u16, base_url: String }`.
  - `fn spawn(model_path: &Path) -> Result<Self, AskError>`:
    1. Pick a random free port (ephemeral range 49152–65535).
    2. Spawn `llama-server --model <path> --port <port> --jinja` as a child process.
    3. Poll `GET http://localhost:<port>/health` every 250 ms up to a 30 s timeout; succeed when status is `ok`.
  - `impl Drop for LlamaServer { fn drop(&mut self) { self.process.kill(); self.process.wait(); } }`.
- Create `src/ask/bridge.rs`:
  - Tool definitions per §8 (`list_files`, `read_file`, `file_info`) serialized as OpenAI-compatible JSON.
  - System prompt per §8.
  - `struct AskSession { server: LlamaServer, device: StegoDevice, messages: Vec<ChatMessage> }`.
  - `fn ask(&mut self, question: &str) -> Result<String, AskError>`:
    1. Append a user message.
    2. POST to `/v1/chat/completions` with messages and tool definitions.
    3. If the response is a tool call: dispatch it (`list_files`, `read_file` → `StegoDevice` via `src/fs/file_ops.rs`, `file_info` → file table lookup), append the tool result as a `tool` message, loop.
    4. If the response is a final assistant message: return the content.
    5. Hard cap on tool-call iterations (V1: 8) to avoid infinite loops.
- CLI wiring in `src/main.rs`:
  - `ask <model.gguf>` enters an interactive REPL: print prompt, read line from stdin, call `session.ask(line)`, print the answer. Ctrl-D or `exit` ends the session. On exit, `LlamaServer::drop` kills the subprocess.
- `AskError` variants: SpawnFailed, HealthTimeout, HttpError, ToolCallLimitExceeded, InvalidToolCall, `#[from] DeviceError`, `#[from] FsError`.
- Tests:
  - `tests/ask_tool_dispatch.rs`: unit-level — construct an `AskSession` with a mock HTTP client that returns a canned `list_files` tool call; verify the session calls `device.list_files()` and feeds the result back.
  - `tests/ask_e2e.rs` (`#[ignore]` unless `LLMDB_E2E_ASK=1` and `llama-server` is on PATH): store a file named `notes.txt` with known content; ask "What's in notes.txt?"; verify the answer mentions something from the stored content.

Existing code to reuse / rework / delete:
- Reuse: `StegoDevice::list_files` / `get_file` from Task 09, `ureq` dependency added in Task 01
- Rework: `src/main.rs` `ask` subcommand body
- Delete: nothing

Acceptance criteria:
- `cargo test --offline tests::ask_tool_dispatch` passes (mocked HTTP, no subprocess).
- With `LLMDB_E2E_ASK=1` and `llama-server` on PATH: `ask` answers a question about a stored file by issuing at least one `read_file` tool call dispatched through `StegoDevice`.
- `LlamaServer::drop` reliably kills the subprocess (verifiable via `pgrep llama-server` returning empty after the handle is dropped).
- Tool-call iteration limit prevents runaway loops: a malformed tool-call response that never yields a final answer returns `AskError::ToolCallLimitExceeded` after 8 iterations.
- §14 open item: verify `llama-server --jinja` handles SmolLM3-3B's tool-call chat template correctly before declaring this task done. If the template is broken, document the workaround (manual chat-template formatting) inline.
