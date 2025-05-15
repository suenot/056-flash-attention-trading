//! Flash-style tiled attention implementation.
//!
//! This implements the key ideas from FlashAttention:
//! - Process attention in tiles to reduce memory bandwidth
//! - Use online softmax to avoid materializing the full attention matrix
//! - Recompute attention in backward pass (not implemented here, forward only)
//!
//! Note: This is a CPU implementation for demonstration. For GPU acceleration,
//! use a CUDA-based library like candle with flash-attention support.

use super::AttentionConfig;
use ndarray::{s, Array1, Array2};
use rayon::prelude::*;

/// Compute flash-style tiled attention.
///
/// Instead of computing the full N×N attention matrix at once,
/// this processes attention in blocks, maintaining running statistics
/// for the softmax normalization.
///
/// # Algorithm
///
/// For each block of queries (Q_block):
///   For each block of keys/values (K_block, V_block):
///     1. Compute S_block = Q_block @ K_block^T / sqrt(d_k)
///     2. Update running max for numerical stability
///     3. Compute local softmax with correction factor
///     4. Accumulate output with proper scaling
///
/// # Arguments
/// * `query` - Query matrix [seq_len, d_model]
/// * `key` - Key matrix [seq_len, d_model]
/// * `value` - Value matrix [seq_len, d_model]
/// * `config` - Attention configuration
///
/// # Returns
/// Output matrix [seq_len, d_model]
pub fn flash_attention(
    query: &Array2<f32>,
    key: &Array2<f32>,
    value: &Array2<f32>,
    config: &AttentionConfig,
) -> Array2<f32> {
    let seq_len = query.nrows();
    let d_model = query.ncols();
    let block_size = config.block_size;
    let d_k = (d_model / config.n_heads) as f32;
    let scale = 1.0 / d_k.sqrt();

    // Output accumulator
    let mut output = Array2::<f32>::zeros((seq_len, d_model));

    // Running statistics for online softmax
    let mut max_scores = Array1::<f32>::from_elem(seq_len, f32::NEG_INFINITY);
    let mut sum_exp = Array1::<f32>::zeros(seq_len);

    // Process key-value blocks
    let n_kv_blocks = (seq_len + block_size - 1) / block_size;

    for kv_block_idx in 0..n_kv_blocks {
        let kv_start = kv_block_idx * block_size;
        let kv_end = (kv_start + block_size).min(seq_len);

        let k_block = key.slice(s![kv_start..kv_end, ..]);
        let v_block = value.slice(s![kv_start..kv_end, ..]);

        // Process query blocks
        let n_q_blocks = (seq_len + block_size - 1) / block_size;

        for q_block_idx in 0..n_q_blocks {
            let q_start = q_block_idx * block_size;
            let q_end = (q_start + block_size).min(seq_len);

            let q_block = query.slice(s![q_start..q_end, ..]);

            // Compute attention scores for this block
            let scores = q_block.dot(&k_block.t()) * scale;

            // Apply causal mask if needed
            let scores = if config.causal {
                apply_causal_mask_block(&scores, q_start, kv_start)
            } else {
                scores.to_owned()
            };

            // Update running max and compute local softmax
            for (i, q_idx) in (q_start..q_end).enumerate() {
                let row_scores = scores.row(i);

                // New max for this row
                let local_max = row_scores
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let new_max = max_scores[q_idx].max(local_max);

                // Correction factor for previous accumulations
                let correction = (max_scores[q_idx] - new_max).exp();

                // Compute exp(scores - new_max)
                let exp_scores: Vec<f32> =
                    row_scores.iter().map(|&s| (s - new_max).exp()).collect();
                let local_sum: f32 = exp_scores.iter().sum();

                // Update output with correction
                for d in 0..d_model {
                    output[[q_idx, d]] *= correction;
                }

                // Add contribution from this block
                for (j, &exp_s) in exp_scores.iter().enumerate() {
                    let kv_idx = kv_start + j;
                    if kv_idx < kv_end {
                        for d in 0..d_model {
                            output[[q_idx, d]] += exp_s * v_block[[j, d]];
                        }
                    }
                }

                // Update running statistics
                sum_exp[q_idx] = sum_exp[q_idx] * correction + local_sum;
                max_scores[q_idx] = new_max;
            }
        }
    }

    // Final normalization
    for q_idx in 0..seq_len {
        let norm = sum_exp[q_idx];
        if norm > 0.0 {
            for d in 0..d_model {
                output[[q_idx, d]] /= norm;
            }
        }
    }

    output
}

/// Parallel version of flash attention using rayon
pub fn flash_attention_parallel(
    query: &Array2<f32>,
    key: &Array2<f32>,
    value: &Array2<f32>,
    config: &AttentionConfig,
) -> Array2<f32> {
    let seq_len = query.nrows();
    let d_model = query.ncols();
    let block_size = config.block_size;
    let d_k = (d_model / config.n_heads) as f32;
    let scale = 1.0 / d_k.sqrt();

    // Process each query position in parallel
    let results: Vec<Array1<f32>> = (0..seq_len)
        .into_par_iter()
        .map(|q_idx| {
            let q = query.row(q_idx);

            let mut output = Array1::<f32>::zeros(d_model);
            let mut max_score = f32::NEG_INFINITY;
            let mut sum_exp = 0.0f32;

            // Process key-value blocks
            let n_blocks = (seq_len + block_size - 1) / block_size;

            for block_idx in 0..n_blocks {
                let kv_start = block_idx * block_size;
                let kv_end = (kv_start + block_size).min(seq_len);

                for kv_idx in kv_start..kv_end {
                    // Causal masking
                    if config.causal && kv_idx > q_idx {
                        continue;
                    }

                    let k = key.row(kv_idx);
                    let v = value.row(kv_idx);

                    // Compute score
                    let score: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() * scale;

                    // Update with online softmax
                    let new_max = max_score.max(score);
                    let correction = (max_score - new_max).exp();

                    // Scale previous accumulations
                    output *= correction;
                    sum_exp = sum_exp * correction;

                    // Add this position's contribution
                    let exp_score = (score - new_max).exp();
                    for (d, v_d) in v.iter().enumerate() {
                        output[d] += exp_score * v_d;
                    }

                    sum_exp += exp_score;
                    max_score = new_max;
                }
            }

            // Normalize
            if sum_exp > 0.0 {
                output /= sum_exp;
            }

            output
        })
        .collect();

    // Combine results
    let mut output = Array2::<f32>::zeros((seq_len, d_model));
    for (i, row) in results.into_iter().enumerate() {
        output.row_mut(i).assign(&row);
    }

    output
}

/// Apply causal mask to a block of attention scores
fn apply_causal_mask_block(
    scores: &Array2<f32>,
    q_start: usize,
    kv_start: usize,
) -> Array2<f32> {
    let mut masked = scores.to_owned();
    let (n_q, n_kv) = masked.dim();

    for i in 0..n_q {
        let q_idx = q_start + i;
        for j in 0..n_kv {
            let kv_idx = kv_start + j;
            if kv_idx > q_idx {
                masked[[i, j]] = f32::NEG_INFINITY;
            }
        }
    }

    masked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::standard::standard_attention;
    use ndarray_rand::RandomExt;
    use rand::distributions::Uniform;

    #[test]
    fn test_flash_vs_standard() {
        let config = AttentionConfig {
            d_model: 64,
            n_heads: 4,
            block_size: 16,
            causal: false,
            ..Default::default()
        };

        let seq_len = 64;
        let query = Array2::random((seq_len, config.d_model), Uniform::new(-1.0, 1.0));
        let key = Array2::random((seq_len, config.d_model), Uniform::new(-1.0, 1.0));
        let value = Array2::random((seq_len, config.d_model), Uniform::new(-1.0, 1.0));

        let standard = standard_attention(&query, &key, &value, &config);
        let flash = flash_attention(&query, &key, &value, &config);

        let diff = (&standard - &flash).mapv(|x| x.abs());
        let max_diff = diff.iter().cloned().fold(0.0f32, f32::max);

        assert!(max_diff < 0.01, "Max diff: {}", max_diff);
    }

    #[test]
    fn test_flash_parallel() {
        let config = AttentionConfig {
            d_model: 64,
            n_heads: 4,
            block_size: 16,
            causal: false,
            ..Default::default()
        };

        let seq_len = 64;
        let query = Array2::random((seq_len, config.d_model), Uniform::new(-1.0, 1.0));
        let key = Array2::random((seq_len, config.d_model), Uniform::new(-1.0, 1.0));
        let value = Array2::random((seq_len, config.d_model), Uniform::new(-1.0, 1.0));

        let sequential = flash_attention(&query, &key, &value, &config);
        let parallel = flash_attention_parallel(&query, &key, &value, &config);

        let diff = (&sequential - &parallel).mapv(|x| x.abs());
        let max_diff = diff.iter().cloned().fold(0.0f32, f32::max);

        assert!(max_diff < 0.01, "Max diff: {}", max_diff);
    }
}
