//! Transformer architecture components.
//!
//! Implements transformer blocks with flash-style attention.

use crate::attention::{flash_attention, standard_attention, AttentionConfig};
use ndarray::{s, Array1, Array2, Axis};
use rand::Rng;
use rand_distr::Normal;

/// Configuration for transformer model
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub dropout: f32,
    pub use_flash: bool,
    pub block_size: usize,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            n_heads: 8,
            n_layers: 6,
            d_ff: 1024,
            max_seq_len: 4096,
            dropout: 0.1,
            use_flash: true,
            block_size: 64,
        }
    }
}

/// Layer normalization
#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub gamma: Array1<f32>,
    pub beta: Array1<f32>,
    eps: f32,
}

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
        Self {
            gamma: Array1::ones(dim),
            beta: Array1::zeros(dim),
            eps: 1e-5,
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mean = x.mean_axis(Axis(1)).unwrap();
        let var = x.var_axis(Axis(1), 0.0);

        let mut normalized = x.clone();
        for (i, mut row) in normalized.rows_mut().into_iter().enumerate() {
            let std = (var[i] + self.eps).sqrt();
            for (j, val) in row.iter_mut().enumerate() {
                *val = (*val - mean[i]) / std * self.gamma[j] + self.beta[j];
            }
        }

        normalized
    }
}

/// Positional encoding (learnable)
#[derive(Debug, Clone)]
pub struct PositionalEncoding {
    pub encoding: Array2<f32>,
}

impl PositionalEncoding {
    pub fn new(d_model: usize, max_len: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        let encoding = Array2::from_shape_fn((max_len, d_model), |_| rng.sample(normal));

        Self { encoding }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let seq_len = x.nrows();
        let pos_slice = self.encoding.slice(s![..seq_len, ..]);
        x + &pos_slice
    }
}

/// Feed-forward network
#[derive(Debug, Clone)]
pub struct FeedForward {
    pub w1: Array2<f32>,
    pub b1: Array1<f32>,
    pub w2: Array2<f32>,
    pub b2: Array1<f32>,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale1 = (2.0 / (d_model + d_ff) as f32).sqrt();
        let scale2 = (2.0 / (d_ff + d_model) as f32).sqrt();

        let w1 = Array2::from_shape_fn((d_model, d_ff), |_| {
            rng.gen::<f32>() * 2.0 * scale1 - scale1
        });
        let b1 = Array1::zeros(d_ff);
        let w2 = Array2::from_shape_fn((d_ff, d_model), |_| {
            rng.gen::<f32>() * 2.0 * scale2 - scale2
        });
        let b2 = Array1::zeros(d_model);

        Self { w1, b1, w2, b2 }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // First linear + GELU
        let h = x.dot(&self.w1);
        let h = h + &self.b1;
        let h = gelu(&h);

        // Second linear
        let out = h.dot(&self.w2);
        out + &self.b2
    }
}

/// GELU activation function
fn gelu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| {
        0.5 * v * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (v + 0.044715 * v.powi(3))).tanh())
    })
}

/// Multi-head attention layer
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    pub wq: Array2<f32>,
    pub wk: Array2<f32>,
    pub wv: Array2<f32>,
    pub wo: Array2<f32>,
    pub config: AttentionConfig,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_heads: usize, _use_flash: bool, block_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (2 * d_model) as f32).sqrt();

        let wq =
            Array2::from_shape_fn((d_model, d_model), |_| rng.gen::<f32>() * 2.0 * scale - scale);
        let wk =
            Array2::from_shape_fn((d_model, d_model), |_| rng.gen::<f32>() * 2.0 * scale - scale);
        let wv =
            Array2::from_shape_fn((d_model, d_model), |_| rng.gen::<f32>() * 2.0 * scale - scale);
        let wo =
            Array2::from_shape_fn((d_model, d_model), |_| rng.gen::<f32>() * 2.0 * scale - scale);

        let config = AttentionConfig {
            d_model,
            n_heads,
            block_size,
            causal: false,
            ..Default::default()
        };

        Self {
            wq,
            wk,
            wv,
            wo,
            config,
        }
    }

    pub fn forward(&self, x: &Array2<f32>, use_flash: bool) -> Array2<f32> {
        // Project to Q, K, V
        let q = x.dot(&self.wq);
        let k = x.dot(&self.wk);
        let v = x.dot(&self.wv);

        // Apply attention
        let attn_out = if use_flash {
            flash_attention(&q, &k, &v, &self.config)
        } else {
            standard_attention(&q, &k, &v, &self.config)
        };

        // Output projection
        attn_out.dot(&self.wo)
    }
}

/// Transformer block (pre-norm architecture)
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    pub attention: MultiHeadAttention,
    pub ff: FeedForward,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub use_flash: bool,
}

impl TransformerBlock {
    pub fn new(config: &TransformerConfig) -> Self {
        Self {
            attention: MultiHeadAttention::new(
                config.d_model,
                config.n_heads,
                config.use_flash,
                config.block_size,
            ),
            ff: FeedForward::new(config.d_model, config.d_ff),
            norm1: LayerNorm::new(config.d_model),
            norm2: LayerNorm::new(config.d_model),
            use_flash: config.use_flash,
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // Pre-norm attention
        let normalized = self.norm1.forward(x);
        let attn_out = self.attention.forward(&normalized, self.use_flash);
        let x = x + &attn_out;

        // Pre-norm feed-forward
        let normalized = self.norm2.forward(&x);
        let ff_out = self.ff.forward(&normalized);
        x + &ff_out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(64);
        let x = Array2::from_shape_fn((10, 64), |(i, j)| (i + j) as f32 * 0.1);
        let out = ln.forward(&x);
        assert_eq!(out.dim(), (10, 64));
    }

    #[test]
    fn test_positional_encoding() {
        let pe = PositionalEncoding::new(64, 100);
        let x = Array2::zeros((50, 64));
        let out = pe.forward(&x);
        assert_eq!(out.dim(), (50, 64));
    }

    #[test]
    fn test_feed_forward() {
        let ff = FeedForward::new(64, 256);
        let x = Array2::from_shape_fn((10, 64), |(i, j)| (i + j) as f32 * 0.01);
        let out = ff.forward(&x);
        assert_eq!(out.dim(), (10, 64));
    }

    #[test]
    fn test_transformer_block() {
        let config = TransformerConfig {
            d_model: 64,
            n_heads: 4,
            d_ff: 256,
            use_flash: true,
            block_size: 16,
            ..Default::default()
        };

        let block = TransformerBlock::new(&config);
        let x = Array2::from_shape_fn((32, 64), |(i, j)| ((i + j) as f32 * 0.01).sin());
        let out = block.forward(&x);
        assert_eq!(out.dim(), (32, 64));
    }
}
