//! Example: Backtesting a Flash Attention Trading Strategy
//!
//! This example demonstrates the complete pipeline:
//! 1. Load or generate data
//! 2. Create a Flash Attention model
//! 3. Run backtest simulation
//! 4. Analyze results
//!
//! Run with: cargo run --example backtest

use anyhow::Result;
use chrono::Utc;
use flash_attention_trading::{
    data::{calculate_features, OhlcvData, TradingDataset},
    model::{FlashAttentionTrader, OutputType, TraderConfig},
    strategy::{backtest, BacktestConfig, SignalGenerator},
};
use ndarray::{Array1, Array2};
use rand::Rng;

fn main() -> Result<()> {
    env_logger::init();

    println!("Flash Attention Trading - Backtesting Example");
    println!("==============================================\n");

    // 1. Generate synthetic data (in production, use real data)
    println!("1. Preparing trading data...");
    let ohlcv_data = generate_synthetic_data(500);
    let features = calculate_features(&ohlcv_data);

    let dataset = TradingDataset {
        features: features.to_array(),
        targets: Array1::zeros(features.returns.len()), // Not used in backtest
        timestamps: ohlcv_data.iter().map(|d| d.timestamp).collect(),
        prices: ohlcv_data.iter().map(|d| d.close).collect(),
    };

    println!("   ✓ Loaded {} data points", dataset.features.nrows());
    println!(
        "   Price range: ${:.2} - ${:.2}",
        dataset.prices.iter().cloned().fold(f64::INFINITY, f64::min),
        dataset.prices.iter().cloned().fold(0.0, f64::max)
    );

    // 2. Create the trading model
    println!("\n2. Creating Flash Attention Trading Model...");
    let model_config = TraderConfig {
        input_dim: 10,
        d_model: 64,
        n_heads: 4,
        n_layers: 2,
        d_ff: 256,
        max_seq_len: 1024,
        n_outputs: 1,
        output_type: OutputType::Regression,
        dropout: 0.1,
        use_flash: true,
        block_size: 16,
    };

    let model = FlashAttentionTrader::new(model_config);
    println!("   ✓ Model created with {} parameters", model.count_parameters());

    // 3. Configure backtest
    println!("\n3. Configuring backtest...");
    let backtest_config = BacktestConfig {
        initial_capital: 100_000.0,
        transaction_cost: 0.001, // 0.1%
        slippage: 0.0005,        // 0.05%
        position_size: 0.1,      // 10% of capital per trade
        signal_config: SignalGenerator {
            buy_threshold: 0.002,  // 0.2% expected return
            sell_threshold: -0.002,
            allow_short: true,
            max_position: 1.0,
        },
    };

    println!("   Initial capital: ${:.2}", backtest_config.initial_capital);
    println!("   Transaction cost: {:.2}%", backtest_config.transaction_cost * 100.0);
    println!("   Position size: {:.0}%", backtest_config.position_size * 100.0);

    // 4. Run backtest
    println!("\n4. Running backtest simulation...");
    let lookback = 50;

    let start = std::time::Instant::now();
    let result = backtest(&model, &dataset, lookback, &backtest_config)?;
    let elapsed = start.elapsed();

    println!("   ✓ Backtest completed in {:?}", elapsed);

    // 5. Display results
    println!("\n5. Results:");
    println!("   ─────────────────────────────");
    println!("{}", result.summary());

    // 6. Analyze trades
    println!("\n6. Trade Analysis:");
    println!("   ─────────────────────────────");

    if !result.trades.is_empty() {
        // First few trades
        println!("\n   First 5 trades:");
        for trade in result.trades.iter().take(5) {
            println!(
                "   {:?} @ ${:.2} -> PnL: ${:.2}",
                trade.signal, trade.price, trade.pnl
            );
        }

        // Best and worst trades
        let best_trade = result.trades.iter().max_by(|a, b| {
            a.pnl.partial_cmp(&b.pnl).unwrap()
        });
        let worst_trade = result.trades.iter().min_by(|a, b| {
            a.pnl.partial_cmp(&b.pnl).unwrap()
        });

        if let Some(best) = best_trade {
            println!("\n   Best trade:  ${:.2} profit", best.pnl);
        }
        if let Some(worst) = worst_trade {
            println!("   Worst trade: ${:.2} loss", worst.pnl);
        }
    }

    // 7. Portfolio equity curve summary
    println!("\n7. Equity Curve:");
    println!("   ─────────────────────────────");

    let n_trades = result.trades.len();
    if n_trades > 0 {
        let quarter = n_trades / 4;
        println!("   Start:     ${:.2}", backtest_config.initial_capital);
        if quarter > 0 {
            println!("   Q1:        ${:.2}", result.trades[quarter].portfolio_value);
        }
        if quarter * 2 < n_trades {
            println!("   Q2:        ${:.2}", result.trades[quarter * 2].portfolio_value);
        }
        if quarter * 3 < n_trades {
            println!("   Q3:        ${:.2}", result.trades[quarter * 3].portfolio_value);
        }
        println!("   Final:     ${:.2}", result.final_value);
    }

    println!("\n✓ Backtesting example complete!");

    // Interpretation guide
    println!("\nInterpretation Guide:");
    println!("─────────────────────────────────────────────");
    println!("• Sharpe Ratio > 1.0 is good, > 2.0 is excellent");
    println!("• Max Drawdown < 20% is acceptable for most strategies");
    println!("• Win Rate alone is not enough - consider profit factor");
    println!("• Profit Factor > 1.5 indicates a robust edge");
    println!();
    println!("Note: This uses a randomly initialized model.");
    println!("Real trading requires trained models and rigorous validation.");

    Ok(())
}

/// Generate synthetic OHLCV data with realistic patterns
fn generate_synthetic_data(n: usize) -> Vec<OhlcvData> {
    let mut data = Vec::with_capacity(n);
    let mut price = 100.0;
    let mut rng = rand::thread_rng();

    for i in 0..n {
        // Multiple trend components
        let trend1 = (i as f64 * 0.02).sin() * 0.3;
        let trend2 = (i as f64 * 0.005).cos() * 0.2;
        let noise: f64 = rng.gen::<f64>() * 2.0 - 1.0;

        // Volatility clustering
        let vol_multiplier = 1.0 + (i as f64 * 0.01).sin().abs() * 0.5;
        let change = (trend1 + trend2 + noise * 0.3) * vol_multiplier;

        price *= 1.0 + change * 0.015;
        price = price.max(10.0);

        let volatility = (rng.gen::<f64>() + 0.5) * 1.5 * vol_multiplier;
        let high = price * (1.0 + volatility * 0.01);
        let low = price * (1.0 - volatility * 0.01);
        let open = price * (1.0 + (rng.gen::<f64>() - 0.5) * 0.008);

        data.push(OhlcvData {
            timestamp: Utc::now(),
            open,
            high,
            low,
            close: price,
            volume: 1000.0 * (1.0 + rng.gen::<f64>() * 2.0),
        });
    }

    data
}
