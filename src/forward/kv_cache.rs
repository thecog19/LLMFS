//! Preallocated KV cache — one buffer pair per layer.
//!
//! `LayerKvCache` stores `[max_ctx, kv_width]` row-major `K` and `V`
//! slabs. Each `forward_block` call appends its new K/V rows to the
//! tail and advances `current_len`, which is also the RoPE starting
//! position for the next call's query rows.
//!
//! This is the minimum API needed to decode beyond prefill:
//!
//! 1. Allocate `KvCache::new(cfg, max_ctx)` once.
//! 2. `forward_block` writes into `cache.layers[n]` and grows
//!    `current_len` by the batch size.
//! 3. Subsequent calls re-use those stored K/V rows and attend
//!    against the full accumulated window.

use crate::forward::config::LlamaConfig;

/// One layer's K and V slabs.
pub struct LayerKvCache {
    /// `[max_ctx, kv_width]` row-major; rows 0..current_len are valid.
    pub k: Vec<f32>,
    /// `[max_ctx, kv_width]` row-major; rows 0..current_len are valid.
    pub v: Vec<f32>,
    /// Number of rows populated so far (= number of tokens already
    /// attended).
    pub current_len: usize,
    /// Preallocated row capacity. Every append must keep
    /// `current_len` ≤ `max_ctx`.
    pub max_ctx: usize,
    /// Width of one row: `n_kv_heads * head_dim`.
    pub kv_width: usize,
}

impl LayerKvCache {
    pub fn new(max_ctx: usize, kv_width: usize) -> Self {
        Self {
            k: vec![0.0; max_ctx * kv_width],
            v: vec![0.0; max_ctx * kv_width],
            current_len: 0,
            max_ctx,
            kv_width,
        }
    }

    /// Reset to empty without deallocating. Useful when replaying
    /// a different sequence without dropping the backing buffers.
    pub fn clear(&mut self) {
        self.current_len = 0;
    }

    /// Append `rows` K/V rows at the tail, copying from `k_src` /
    /// `v_src` (both `[rows, kv_width]` row-major). Panics if the
    /// append would exceed `max_ctx`.
    pub fn append(&mut self, k_src: &[f32], v_src: &[f32], rows: usize) {
        assert_eq!(k_src.len(), rows * self.kv_width);
        assert_eq!(v_src.len(), rows * self.kv_width);
        assert!(
            self.current_len + rows <= self.max_ctx,
            "KV cache overflow: current_len {} + rows {} > max_ctx {}",
            self.current_len,
            rows,
            self.max_ctx,
        );
        let start = self.current_len * self.kv_width;
        let end = start + rows * self.kv_width;
        self.k[start..end].copy_from_slice(k_src);
        self.v[start..end].copy_from_slice(v_src);
        self.current_len += rows;
    }
}

/// Full-model KV cache: one entry per transformer block.
pub struct KvCache {
    pub layers: Vec<LayerKvCache>,
    pub max_ctx: usize,
}

impl KvCache {
    /// Allocate one cache sized for `max_ctx` tokens across every
    /// layer of `cfg`. Each block's cache is zero-initialized
    /// (doesn't matter — rows 0..current_len are always written
    /// before read).
    pub fn new(cfg: &LlamaConfig, max_ctx: usize) -> Self {
        let kv_width = cfg.n_kv_heads * cfg.head_dim;
        let layers = (0..cfg.n_layers)
            .map(|_| LayerKvCache::new(max_ctx, kv_width))
            .collect();
        Self { layers, max_ctx }
    }

    /// Reset every layer to empty.
    pub fn clear(&mut self) {
        for l in self.layers.iter_mut() {
            l.clear();
        }
    }

    /// Current sequence length. All layers share it because every
    /// block processes the same number of tokens per forward call.
    pub fn current_len(&self) -> usize {
        self.layers.first().map(|l| l.current_len).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_grows_current_len_and_copies_rows() {
        let mut c = LayerKvCache::new(4, 3);
        assert_eq!(c.current_len, 0);
        c.append(
            &[1., 2., 3., 4., 5., 6.],
            &[10., 20., 30., 40., 50., 60.],
            2,
        );
        assert_eq!(c.current_len, 2);
        assert_eq!(&c.k[..6], &[1., 2., 3., 4., 5., 6.]);
        assert_eq!(&c.v[..6], &[10., 20., 30., 40., 50., 60.]);
        // Remaining rows untouched.
        assert_eq!(&c.k[6..], &[0.0; 6]);
    }

    #[test]
    fn clear_resets_current_len_without_deallocating() {
        let mut c = LayerKvCache::new(2, 2);
        c.append(&[1., 2.], &[3., 4.], 1);
        c.clear();
        assert_eq!(c.current_len, 0);
        assert_eq!(c.k.len(), 4); // buffer kept
    }

    #[test]
    #[should_panic(expected = "KV cache overflow")]
    fn append_past_max_ctx_panics() {
        let mut c = LayerKvCache::new(1, 2);
        c.append(&[0., 0., 0., 0.], &[0., 0., 0., 0.], 2);
    }
}
