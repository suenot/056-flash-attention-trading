//! Benchmark comparing Flash Attention vs Standard Attention
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use flash_attention_trading::attention::{
    flash_attention, flash_attention_parallel, standard_attention, AttentionConfig,
};
use ndarray::Array2;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

fn benchmark_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("Attention");

    for seq_len in [64, 128, 256, 512] {
        let config = AttentionConfig {
            d_model: 64,
            n_heads: 4,
            block_size: 16,
            causal: false,
            ..Default::default()
        };

        let query = Array2::random((seq_len, config.d_model), Uniform::new(-1.0, 1.0));
        let key = Array2::random((seq_len, config.d_model), Uniform::new(-1.0, 1.0));
        let value = Array2::random((seq_len, config.d_model), Uniform::new(-1.0, 1.0));

        group.bench_with_input(
            BenchmarkId::new("standard", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    standard_attention(
                        black_box(&query),
                        black_box(&key),
                        black_box(&value),
                        black_box(&config),
                    )
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("flash", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    flash_attention(
                        black_box(&query),
                        black_box(&key),
                        black_box(&value),
                        black_box(&config),
                    )
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("flash_parallel", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    flash_attention_parallel(
                        black_box(&query),
                        black_box(&key),
                        black_box(&value),
                        black_box(&config),
                    )
                });
            },
        );
    }

    group.finish();
}

fn benchmark_causal_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("CausalAttention");

    let seq_len = 256;
    let config = AttentionConfig {
        d_model: 64,
        n_heads: 4,
        block_size: 16,
        causal: true,
        ..Default::default()
    };

    let query = Array2::random((seq_len, config.d_model), Uniform::new(-1.0, 1.0));
    let key = Array2::random((seq_len, config.d_model), Uniform::new(-1.0, 1.0));
    let value = Array2::random((seq_len, config.d_model), Uniform::new(-1.0, 1.0));

    group.bench_function("standard_causal", |b| {
        b.iter(|| {
            standard_attention(
                black_box(&query),
                black_box(&key),
                black_box(&value),
                black_box(&config),
            )
        });
    });

    group.bench_function("flash_causal", |b| {
        b.iter(|| {
            flash_attention(
                black_box(&query),
                black_box(&key),
                black_box(&value),
                black_box(&config),
            )
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_attention, benchmark_causal_attention);
criterion_main!(benches);
