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
//!
//! See `/home/suero/.claude/plans/lets-do-4-tool-harmonic-glade.md`
//! for the phased build.

pub mod dequant;
pub mod pre_tokenize;
pub mod tokenizer;
