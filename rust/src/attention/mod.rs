//! Attention mechanisms for the trading model.
//!
//! This module provides:
//! - Standard scaled dot-product attention
//! - Flash-style tiled attention for memory efficiency

mod flash;
mod standard;

pub use flash::{flash_attention, flash_attention_parallel};
pub use standard::standard_attention;

use ndarray::Array2;

/// Configuration for attention computation
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Dropout probability
    pub dropout: f32,
    /// Block size for tiled attention
    pub block_size: usize,
    /// Whether to use causal masking
    pub causal: bool,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            n_heads: 8,
            dropout: 0.1,
            block_size: 64,
            causal: false,
        }
    }
}

/// Compute attention scores and output
pub fn compute_attention(
    query: &Array2<f32>,
    key: &Array2<f32>,
    value: &Array2<f32>,
    config: &AttentionConfig,
    use_flash: bool,
) -> Array2<f32> {
    if use_flash && query.nrows() > config.block_size * 2 {
        flash_attention(query, key, value, config)
    } else {
        standard_attention(query, key, value, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use ndarray_rand::RandomExt;
    use rand::distributions::Uniform;

    #[test]
    fn test_attention_equivalence() {
        let config = AttentionConfig::default();
        let seq_len = 128;
        let d_model = config.d_model;

        // Create random inputs
        let query = Array2::random((seq_len, d_model), Uniform::new(-1.0, 1.0));
        let key = Array2::random((seq_len, d_model), Uniform::new(-1.0, 1.0));
        let value = Array2::random((seq_len, d_model), Uniform::new(-1.0, 1.0));

        // Compute both versions
        let standard_out = standard_attention(&query, &key, &value, &config);
        let flash_out = flash_attention(&query, &key, &value, &config);

        // Check they produce similar results
        let diff = (&standard_out - &flash_out).mapv(|x| x.abs());
        let max_diff = diff.iter().cloned().fold(0.0f32, f32::max);

        assert!(
            max_diff < 0.01,
            "Max difference between standard and flash attention: {}",
            max_diff
        );
    }
}
