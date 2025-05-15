//! Example: Training a Flash Attention Trading Model
//!
//! This example demonstrates how to create and use a Flash Attention
//! enhanced transformer model for trading predictions.
//!
//! Note: This is a simplified example. In production, you would use
//! a proper training loop with backpropagation (e.g., using candle-nn).
//!
//! Run with: cargo run --example train

use anyhow::Result;
use flash_attention_trading::{
    data::{calculate_features, OhlcvData, TradingDataset},
    model::{FlashAttentionTrader, OutputType, TraderConfig},
};
use chrono::Utc;
use ndarray::{Array1, Array2};

fn main() -> Result<()> {
    env_logger::init();

    println!("Flash Attention Trading - Model Training Example");
    println!("=================================================\n");

    // Generate synthetic data for demonstration
    println!("1. Generating synthetic trading data...");
    let data = generate_synthetic_data(1000);
    println!("   ✓ Generated {} data points", data.len());

    // Calculate features
    println!("\n2. Calculating technical features...");
    let features = calculate_features(&data);
    let feature_array = features.to_array();
    println!("   ✓ Feature matrix shape: {:?}", feature_array.dim());

    // Create model
    println!("\n3. Creating Flash Attention Trading Model...");
    let config = TraderConfig {
        input_dim: 10,
        d_model: 128,
        n_heads: 8,
        n_layers: 4,
        d_ff: 512,
        max_seq_len: 2048,
        n_outputs: 1,
        output_type: OutputType::Regression,
        dropout: 0.1,
        use_flash: true,
        block_size: 32,
    };

    let model = FlashAttentionTrader::new(config);
    println!("{}", model.summary());

    // Test forward pass
    println!("\n4. Testing forward pass...");

    let lookback = 256;
    let test_input: Array2<f32> = feature_array
        .slice(ndarray::s![..lookback, ..])
        .to_owned();

    println!("   Input shape: {:?}", test_input.dim());

    let start = std::time::Instant::now();
    let prediction = model.forward(&test_input);
    let elapsed = start.elapsed();

    println!("   ✓ Output shape: {:?}", prediction.dim());
    println!("   ✓ Prediction: {:.6}", prediction[0]);
    println!("   ✓ Inference time: {:?}", elapsed);

    // Test with different sequence lengths
    println!("\n5. Testing with various sequence lengths...");

    for seq_len in [64, 128, 256, 512] {
        if seq_len <= feature_array.nrows() {
            let input: Array2<f32> = feature_array
                .slice(ndarray::s![..seq_len, ..])
                .to_owned();

            let start = std::time::Instant::now();
            let _ = model.forward(&input);
            let elapsed = start.elapsed();

            println!("   Seq len {}: {:?}", seq_len, elapsed);
        }
    }

    // Compare flash vs standard attention
    println!("\n6. Comparing Flash vs Standard Attention...");

    let flash_config = TraderConfig {
        use_flash: true,
        block_size: 32,
        ..config.clone()
    };

    let standard_config = TraderConfig {
        use_flash: false,
        ..config.clone()
    };

    let flash_model = FlashAttentionTrader::new(flash_config);
    let standard_model = FlashAttentionTrader::new(standard_config);

    let test_input: Array2<f32> = feature_array
        .slice(ndarray::s![..256, ..])
        .to_owned();

    // Warm up
    let _ = flash_model.forward(&test_input);
    let _ = standard_model.forward(&test_input);

    // Benchmark
    let iterations = 10;

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = flash_model.forward(&test_input);
    }
    let flash_time = start.elapsed() / iterations;

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = standard_model.forward(&test_input);
    }
    let standard_time = start.elapsed() / iterations;

    println!("   Flash Attention:    {:?} avg", flash_time);
    println!("   Standard Attention: {:?} avg", standard_time);
    println!(
        "   Speedup: {:.2}x",
        standard_time.as_nanos() as f64 / flash_time.as_nanos() as f64
    );

    println!("\n✓ Training example complete!");
    println!("\nNote: This example demonstrates the model architecture.");
    println!("For actual training with gradient descent, use a deep");
    println!("learning framework like candle-nn with CUDA support.");

    Ok(())
}

/// Generate synthetic OHLCV data for demonstration
fn generate_synthetic_data(n: usize) -> Vec<OhlcvData> {
    let mut data = Vec::with_capacity(n);
    let mut price = 100.0;
    let mut rng = rand::thread_rng();

    use rand::Rng;

    for i in 0..n {
        // Add some trends and noise
        let trend = (i as f64 * 0.01).sin() * 0.5;
        let noise: f64 = rng.gen::<f64>() * 2.0 - 1.0;
        let change = trend + noise * 0.5;

        price *= 1.0 + change * 0.02;
        price = price.max(10.0); // Floor price

        let volatility = (rng.gen::<f64>() + 0.5) * 2.0;
        let high = price * (1.0 + volatility * 0.01);
        let low = price * (1.0 - volatility * 0.01);
        let open = price * (1.0 + (rng.gen::<f64>() - 0.5) * 0.01);

        data.push(OhlcvData {
            timestamp: Utc::now(),
            open,
            high,
            low,
            close: price,
            volume: 1000.0 * (1.0 + rng.gen::<f64>()),
        });
    }

    data
}
