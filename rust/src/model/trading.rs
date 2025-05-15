//! Trading model with Flash Attention.
//!
//! Implements the main trading model for prediction.

use super::transformer::{LayerNorm, PositionalEncoding, TransformerBlock, TransformerConfig};
use ndarray::{s, Array1, Array2, Array3};
use rand::Rng;

/// Configuration for the trading model
#[derive(Debug, Clone)]
pub struct TraderConfig {
    /// Number of input features per timestep
    pub input_dim: usize,
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of transformer layers
    pub n_layers: usize,
    /// Feed-forward hidden dimension
    pub d_ff: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Number of output predictions
    pub n_outputs: usize,
    /// Output type: "regression", "direction", or "allocation"
    pub output_type: OutputType,
    /// Dropout rate
    pub dropout: f32,
    /// Whether to use flash attention
    pub use_flash: bool,
    /// Block size for tiled attention
    pub block_size: usize,
}

/// Type of prediction output
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputType {
    Regression,
    Direction,
    Allocation,
}

impl Default for TraderConfig {
    fn default() -> Self {
        Self {
            input_dim: 10,
            d_model: 256,
            n_heads: 8,
            n_layers: 6,
            d_ff: 1024,
            max_seq_len: 4096,
            n_outputs: 1,
            output_type: OutputType::Regression,
            dropout: 0.1,
            use_flash: true,
            block_size: 64,
        }
    }
}

/// Flash Attention Trading Model
#[derive(Debug)]
pub struct FlashAttentionTrader {
    /// Model configuration
    pub config: TraderConfig,
    /// Input projection weights
    input_proj: Array2<f32>,
    /// Input projection bias
    input_bias: Array1<f32>,
    /// Positional encoding
    pos_encoding: PositionalEncoding,
    /// Transformer layers
    layers: Vec<TransformerBlock>,
    /// Final layer norm
    final_norm: LayerNorm,
    /// Output projection weights
    output_proj: Array2<f32>,
    /// Output projection bias
    output_bias: Array1<f32>,
}

impl FlashAttentionTrader {
    /// Create a new trading model
    pub fn new(config: TraderConfig) -> Self {
        let mut rng = rand::thread_rng();

        // Input projection
        let scale = (2.0 / (config.input_dim + config.d_model) as f32).sqrt();
        let input_proj = Array2::from_shape_fn((config.input_dim, config.d_model), |_| {
            rng.gen::<f32>() * 2.0 * scale - scale
        });
        let input_bias = Array1::zeros(config.d_model);

        // Positional encoding
        let pos_encoding = PositionalEncoding::new(config.d_model, config.max_seq_len);

        // Transformer config
        let transformer_config = TransformerConfig {
            d_model: config.d_model,
            n_heads: config.n_heads,
            n_layers: config.n_layers,
            d_ff: config.d_ff,
            max_seq_len: config.max_seq_len,
            dropout: config.dropout,
            use_flash: config.use_flash,
            block_size: config.block_size,
        };

        // Create transformer layers
        let layers: Vec<TransformerBlock> = (0..config.n_layers)
            .map(|_| TransformerBlock::new(&transformer_config))
            .collect();

        // Final norm
        let final_norm = LayerNorm::new(config.d_model);

        // Output projection
        let out_scale = (2.0 / (config.d_model + config.n_outputs) as f32).sqrt();
        let output_proj = Array2::from_shape_fn((config.d_model, config.n_outputs), |_| {
            rng.gen::<f32>() * 2.0 * out_scale - out_scale
        });
        let output_bias = Array1::zeros(config.n_outputs);

        Self {
            config,
            input_proj,
            input_bias,
            pos_encoding,
            layers,
            final_norm,
            output_proj,
            output_bias,
        }
    }

    /// Forward pass for a single sequence
    ///
    /// # Arguments
    /// * `x` - Input tensor [seq_len, input_dim]
    ///
    /// # Returns
    /// Predictions [n_outputs]
    pub fn forward(&self, x: &Array2<f32>) -> Array1<f32> {
        // Input projection
        let mut h = x.dot(&self.input_proj);
        for mut row in h.rows_mut() {
            row += &self.input_bias;
        }

        // Add positional encoding
        h = self.pos_encoding.forward(&h);

        // Pass through transformer layers
        for layer in &self.layers {
            h = layer.forward(&h);
        }

        // Final normalization
        h = self.final_norm.forward(&h);

        // Use last token for prediction
        let last_hidden = h.row(h.nrows() - 1).to_owned();

        // Output projection
        let mut output = last_hidden.dot(&self.output_proj);
        output += &self.output_bias;

        // Apply output activation based on type
        match self.config.output_type {
            OutputType::Regression => output,
            OutputType::Direction => output.mapv(|v| 1.0 / (1.0 + (-v).exp())), // Sigmoid
            OutputType::Allocation => output.mapv(|v| v.tanh()), // Tanh for [-1, 1]
        }
    }

    /// Forward pass for a batch of sequences
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, input_dim]
    ///
    /// # Returns
    /// Predictions [batch, n_outputs]
    pub fn forward_batch(&self, x: &Array3<f32>) -> Array2<f32> {
        let batch_size = x.dim().0;
        let mut outputs = Array2::<f32>::zeros((batch_size, self.config.n_outputs));

        for b in 0..batch_size {
            let seq = x.slice(s![b, .., ..]).to_owned();
            let pred = self.forward(&seq);
            outputs.row_mut(b).assign(&pred);
        }

        outputs
    }

    /// Predict returns for new data
    ///
    /// # Arguments
    /// * `x` - Input features [seq_len, input_dim]
    ///
    /// # Returns
    /// Predicted returns [n_outputs]
    pub fn predict(&self, x: &Array2<f32>) -> Array1<f32> {
        self.forward(x)
    }

    /// Get model summary
    pub fn summary(&self) -> String {
        let n_params = self.count_parameters();
        format!(
            "FlashAttentionTrader:\n\
             - Input dim: {}\n\
             - Model dim: {}\n\
             - Heads: {}\n\
             - Layers: {}\n\
             - Outputs: {}\n\
             - Output type: {:?}\n\
             - Flash attention: {}\n\
             - Parameters: {}",
            self.config.input_dim,
            self.config.d_model,
            self.config.n_heads,
            self.config.n_layers,
            self.config.n_outputs,
            self.config.output_type,
            self.config.use_flash,
            n_params
        )
    }

    /// Count total parameters
    pub fn count_parameters(&self) -> usize {
        let mut count = 0;

        // Input projection
        count += self.input_proj.len() + self.input_bias.len();

        // Positional encoding
        count += self.pos_encoding.encoding.len();

        // Transformer layers
        for layer in &self.layers {
            // Attention
            count += layer.attention.wq.len();
            count += layer.attention.wk.len();
            count += layer.attention.wv.len();
            count += layer.attention.wo.len();

            // Feed forward
            count += layer.ff.w1.len() + layer.ff.b1.len();
            count += layer.ff.w2.len() + layer.ff.b2.len();

            // Layer norms
            count += layer.norm1.gamma.len() + layer.norm1.beta.len();
            count += layer.norm2.gamma.len() + layer.norm2.beta.len();
        }

        // Final norm
        count += self.final_norm.gamma.len() + self.final_norm.beta.len();

        // Output projection
        count += self.output_proj.len() + self.output_bias.len();

        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trader_creation() {
        let config = TraderConfig {
            input_dim: 10,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            d_ff: 128,
            n_outputs: 3,
            use_flash: true,
            block_size: 16,
            ..Default::default()
        };

        let model = FlashAttentionTrader::new(config);
        println!("{}", model.summary());
    }

    #[test]
    fn test_forward_pass() {
        let config = TraderConfig {
            input_dim: 10,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            d_ff: 128,
            n_outputs: 3,
            use_flash: true,
            block_size: 16,
            ..Default::default()
        };

        let model = FlashAttentionTrader::new(config);

        let x = Array2::from_shape_fn((100, 10), |(i, j)| ((i + j) as f32 * 0.01).sin());
        let output = model.forward(&x);

        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_batch_forward() {
        let config = TraderConfig {
            input_dim: 10,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            d_ff: 128,
            n_outputs: 3,
            use_flash: true,
            block_size: 16,
            ..Default::default()
        };

        let model = FlashAttentionTrader::new(config);

        let x = Array3::from_shape_fn((4, 50, 10), |(b, i, j)| ((b + i + j) as f32 * 0.01).sin());
        let output = model.forward_batch(&x);

        assert_eq!(output.dim(), (4, 3));
    }

    #[test]
    fn test_output_types() {
        for output_type in [OutputType::Regression, OutputType::Direction, OutputType::Allocation] {
            let config = TraderConfig {
                input_dim: 10,
                d_model: 32,
                n_heads: 2,
                n_layers: 1,
                d_ff: 64,
                n_outputs: 1,
                output_type,
                use_flash: false,
                block_size: 8,
                ..Default::default()
            };

            let model = FlashAttentionTrader::new(config);
            let x = Array2::from_shape_fn((20, 10), |(i, j)| (i + j) as f32 * 0.1);
            let output = model.forward(&x);

            match output_type {
                OutputType::Direction => {
                    // Should be in [0, 1]
                    assert!(output[0] >= 0.0 && output[0] <= 1.0);
                }
                OutputType::Allocation => {
                    // Should be in [-1, 1]
                    assert!(output[0] >= -1.0 && output[0] <= 1.0);
                }
                OutputType::Regression => {
                    // No constraint
                }
            }
        }
    }
}
