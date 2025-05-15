//! Standard scaled dot-product attention implementation.
//!
//! This is the fallback attention mechanism when:
//! - Sequences are short enough that tiling overhead isn't worth it
//! - We need attention weights for interpretability

use super::AttentionConfig;
use ndarray::{Array2, Axis};

/// Compute standard scaled dot-product attention.
///
/// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
///
/// # Arguments
/// * `query` - Query matrix [seq_len, d_model]
/// * `key` - Key matrix [seq_len, d_model]
/// * `value` - Value matrix [seq_len, d_model]
/// * `config` - Attention configuration
///
/// # Returns
/// Output matrix [seq_len, d_model]
pub fn standard_attention(
    query: &Array2<f32>,
    key: &Array2<f32>,
    value: &Array2<f32>,
    config: &AttentionConfig,
) -> Array2<f32> {
    let d_k = (config.d_model / config.n_heads) as f32;
    let scale = 1.0 / d_k.sqrt();

    // Compute attention scores: QK^T / sqrt(d_k)
    let scores = query.dot(&key.t()) * scale;

    // Apply causal mask if needed
    let scores = if config.causal {
        apply_causal_mask(scores)
    } else {
        scores
    };

    // Softmax
    let attention_weights = softmax(&scores);

    // Compute output: attention_weights * V
    attention_weights.dot(value)
}

/// Compute attention with weights returned (for interpretability)
pub fn standard_attention_with_weights(
    query: &Array2<f32>,
    key: &Array2<f32>,
    value: &Array2<f32>,
    config: &AttentionConfig,
) -> (Array2<f32>, Array2<f32>) {
    let d_k = (config.d_model / config.n_heads) as f32;
    let scale = 1.0 / d_k.sqrt();

    let scores = query.dot(&key.t()) * scale;

    let scores = if config.causal {
        apply_causal_mask(scores)
    } else {
        scores
    };

    let attention_weights = softmax(&scores);
    let output = attention_weights.dot(value);

    (output, attention_weights)
}

/// Apply causal mask (upper triangular = -inf)
fn apply_causal_mask(mut scores: Array2<f32>) -> Array2<f32> {
    let n = scores.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            scores[[i, j]] = f32::NEG_INFINITY;
        }
    }
    scores
}

/// Numerically stable softmax along rows
fn softmax(x: &Array2<f32>) -> Array2<f32> {
    // Subtract max for numerical stability
    let max_vals = x.map_axis(Axis(1), |row| {
        row.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    });

    let shifted = x - &max_vals.insert_axis(Axis(1));
    let exp_x = shifted.mapv(f32::exp);

    let sum_exp = exp_x.sum_axis(Axis(1));
    let result = exp_x / &sum_exp.insert_axis(Axis(1));

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::arr2;

    #[test]
    fn test_softmax() {
        let x = arr2(&[[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]);
        let result = softmax(&x);

        // Check rows sum to 1
        for row in result.rows() {
            let sum: f32 = row.iter().sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
        }

        // Check second row is uniform (equal inputs)
        assert_abs_diff_eq!(result[[1, 0]], result[[1, 1]], epsilon = 1e-5);
        assert_abs_diff_eq!(result[[1, 1]], result[[1, 2]], epsilon = 1e-5);
    }

    #[test]
    fn test_causal_mask() {
        let x = arr2(&[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
        let masked = apply_causal_mask(x);

        // Check upper triangle is -inf
        assert!(masked[[0, 1]].is_infinite());
        assert!(masked[[0, 2]].is_infinite());
        assert!(masked[[1, 2]].is_infinite());

        // Check diagonal and lower are unchanged
        assert_abs_diff_eq!(masked[[0, 0]], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(masked[[1, 0]], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(masked[[2, 2]], 1.0, epsilon = 1e-5);
    }
}
