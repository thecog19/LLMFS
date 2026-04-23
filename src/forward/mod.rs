//! Hand-rolled CPU transformer forward pass for V2 progressive
//! calibration.
//!
//! DESIGN-NEW.MD §15.4 calls for activation-aware salience scores
//! (Tier 1 — AWQ — and Tier 2 — Hessian). Both need a forward pass
//! over a calibration corpus; §15.4 picks a hand-rolled Rust
//! implementation over a patched llama.cpp fork or a heavy
//! third-party crate. This module is that implementation.
//!
//! Milestone A builds the forward pass to perplexity parity with
//! `llama-perplexity` on SmolLM2-135M-F16. Milestone B wires AWQ
//! collection + the `llmdb calibrate` CLI + the V2 salience inode
//! into it. Milestones C+ (quantized inference, Hessian / GPTQ, V3
//! GPU) are separate plans.

pub mod awq;
pub mod block;
pub mod compensation;
pub mod config;
pub mod dequant;
pub mod hessian;
pub mod hessian_cache;
pub mod kv_cache;
pub mod linalg;
pub mod model;
pub mod ops;
pub mod perplexity;
pub mod pre_tokenize;
pub mod tokenizer;
pub mod weights;

// Re-exports for the common entry points. Submodules stay `pub`
// for now — the low-level types (`BlockScratch`, `LayerKvCache`,
// `LlamaConfig`, …) are still used directly by A7's test harness
// and whatever wires into V2 calibration next. Narrowing those to
// `pub(crate)` is reasonable once the B-milestones settle the
// external-API shape.
pub use awq::{ActivationSite, AwqCollector};
pub use block::{BlockObserver, NoopObserver};
pub use hessian::HessianAccumulator;
pub use hessian_cache::{CholeskyFactor, HessianFactorCache, LowRankFactor};
pub use config::{ConfigError, LlamaConfig};
pub use kv_cache::{KvCache, LayerKvCache};
pub use model::{ForwardModel, ModelLoadError, ModelScratch};
pub use perplexity::PerplexityError;
pub use tokenizer::{
    DecodeError, EncodeError, SpecialTokens, Tokenizer, TokenizerConfig, TokenizerError,
    TokenizerModel,
};
