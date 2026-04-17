//! Subprocess manager for `llama-server` (from llama.cpp).
//!
//! Spawns the binary, waits for its health endpoint, and kills it on Drop
//! so a crashing `ask` CLI never leaves a zombie server behind. The V1
//! health check is a plain `GET /health` on the chosen port — that
//! endpoint returns `{"status": "ok"}` once the model is loaded and
//! ready to accept completions.

use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use crate::ask::AskError;

/// Kept for backwards-compat with `bootstrap_smoke`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AskServerBootstrap {
    pub subprocess_backend: bool,
}

impl Default for AskServerBootstrap {
    fn default() -> Self {
        Self {
            subprocess_backend: true,
        }
    }
}

pub struct LlamaServer {
    process: Child,
    port: u16,
    base_url: String,
}

impl LlamaServer {
    /// Spawn `llama-server --model <path> --port <port> --jinja` and wait
    /// until its `/health` endpoint returns ok. `port` is picked by the
    /// caller (we leave discovery to the caller to keep retries on
    /// port collision out of this function). A 30-second health timeout
    /// covers "the model is loading" without hanging on a broken binary.
    pub fn spawn(model_path: &Path, port: u16) -> Result<Self, AskError> {
        let model = model_path
            .to_str()
            .ok_or_else(|| AskError::SpawnFailed("non-utf8 model path".into()))?;
        let mut cmd = Command::new("llama-server");
        cmd.args([
            "--model",
            model,
            "--port",
            &port.to_string(),
            "--jinja",
            // Keep V1 context small; ask-bridge is Q/A over short files.
            "--ctx-size",
            "4096",
        ]);
        cmd.stdout(Stdio::null()).stderr(Stdio::null());
        let process = cmd
            .spawn()
            .map_err(|e| AskError::SpawnFailed(format!("{e} (is `llama-server` on PATH?)")))?;
        let base_url = format!("http://127.0.0.1:{port}");
        let srv = Self {
            process,
            port,
            base_url,
        };
        srv.wait_for_health(Duration::from_secs(30))?;
        Ok(srv)
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn wait_for_health(&self, timeout: Duration) -> Result<(), AskError> {
        let health = format!("{}/health", self.base_url);
        let deadline = Instant::now() + timeout;
        let poll = Duration::from_millis(250);
        while Instant::now() < deadline {
            match ureq::get(&health).call() {
                Ok(resp) if resp.status() == 200 => return Ok(()),
                _ => {}
            }
            std::thread::sleep(poll);
        }
        Err(AskError::HealthTimeout {
            url: health,
            seconds: timeout.as_secs(),
        })
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        // Kill + reap. Best-effort — we never want a zombie.
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}
