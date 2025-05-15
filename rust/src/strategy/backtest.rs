//! Backtesting framework for trading strategies.
//!
//! Simulates trading with historical data and calculates performance metrics.

use super::signals::{SignalGenerator, TradingSignal};
use crate::data::TradingDataset;
use crate::model::FlashAttentionTrader;
use anyhow::Result;
use chrono::{DateTime, Utc};
use ndarray::Array2;

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Transaction cost (as fraction, e.g., 0.001 = 0.1%)
    pub transaction_cost: f64,
    /// Slippage (as fraction)
    pub slippage: f64,
    /// Position size (as fraction of capital)
    pub position_size: f64,
    /// Signal generator configuration
    pub signal_config: SignalGenerator,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            transaction_cost: 0.001,
            slippage: 0.0005,
            position_size: 0.1,
            signal_config: SignalGenerator::default(),
        }
    }
}

/// Individual trade record
#[derive(Debug, Clone)]
pub struct Trade {
    pub timestamp: DateTime<Utc>,
    pub signal: TradingSignal,
    pub price: f64,
    pub predicted_return: f64,
    pub actual_return: f64,
    pub position_size: f64,
    pub pnl: f64,
    pub cumulative_pnl: f64,
    pub portfolio_value: f64,
}

/// Backtest results
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// All trades
    pub trades: Vec<Trade>,
    /// Final portfolio value
    pub final_value: f64,
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Volatility (annualized)
    pub volatility: f64,
    /// Sharpe ratio (assuming 0% risk-free rate)
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Number of trades
    pub num_trades: usize,
    /// Average trade return
    pub avg_trade_return: f64,
}

impl BacktestResult {
    /// Print summary to console
    pub fn summary(&self) -> String {
        format!(
            "Backtest Results:\n\
             ─────────────────────────────\n\
             Final Value:      ${:.2}\n\
             Total Return:     {:.2}%\n\
             Annualized Return:{:.2}%\n\
             Volatility:       {:.2}%\n\
             Sharpe Ratio:     {:.3}\n\
             Sortino Ratio:    {:.3}\n\
             Max Drawdown:     {:.2}%\n\
             Win Rate:         {:.2}%\n\
             Profit Factor:    {:.2}\n\
             Num Trades:       {}\n\
             Avg Trade Return: {:.4}%",
            self.final_value,
            self.total_return * 100.0,
            self.annualized_return * 100.0,
            self.volatility * 100.0,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown * 100.0,
            self.win_rate * 100.0,
            self.profit_factor,
            self.num_trades,
            self.avg_trade_return * 100.0
        )
    }
}

/// Run backtest on trading model
pub fn backtest(
    model: &FlashAttentionTrader,
    dataset: &TradingDataset,
    lookback: usize,
    config: &BacktestConfig,
) -> Result<BacktestResult> {
    let n = dataset.features.nrows();
    if n < lookback + 10 {
        anyhow::bail!("Not enough data for backtesting");
    }

    let mut trades = Vec::new();
    let mut portfolio_value = config.initial_capital;
    let mut cumulative_pnl = 0.0;
    let mut peak_value = config.initial_capital;
    let mut max_drawdown = 0.0;
    let mut returns = Vec::new();

    // Simulate trading
    for i in lookback..(n - 1) {
        // Get input sequence
        let start = i - lookback;
        let x: Array2<f32> = dataset.features.slice(ndarray::s![start..i, ..]).to_owned();

        // Get prediction
        let prediction = model.forward(&x);
        let pred_return = prediction[0];

        // Generate signal
        let signal = config.signal_config.generate(pred_return);

        // Calculate actual return
        let actual_return = if i + 1 < n {
            (dataset.prices[i + 1] - dataset.prices[i]) / dataset.prices[i]
        } else {
            0.0
        };

        // Calculate PnL
        let position = match signal {
            TradingSignal::Buy => config.position_size,
            TradingSignal::Sell => -config.position_size,
            TradingSignal::Hold => 0.0,
        };

        let gross_pnl = position * actual_return * portfolio_value;
        let costs = position.abs() * config.transaction_cost * portfolio_value;
        let slippage_cost = position.abs() * config.slippage * portfolio_value;
        let net_pnl = gross_pnl - costs - slippage_cost;

        portfolio_value += net_pnl;
        cumulative_pnl += net_pnl;

        // Track returns for metrics
        if position != 0.0 {
            returns.push(net_pnl / config.initial_capital);
        }

        // Update max drawdown
        if portfolio_value > peak_value {
            peak_value = portfolio_value;
        }
        let drawdown = (peak_value - portfolio_value) / peak_value;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }

        // Record trade
        trades.push(Trade {
            timestamp: dataset.timestamps[i],
            signal,
            price: dataset.prices[i],
            predicted_return: pred_return as f64,
            actual_return,
            position_size: position,
            pnl: net_pnl,
            cumulative_pnl,
            portfolio_value,
        });
    }

    // Calculate metrics
    let total_return = (portfolio_value - config.initial_capital) / config.initial_capital;
    let trading_days = trades.len() as f64;
    let annualized_return = (1.0 + total_return).powf(252.0 / trading_days) - 1.0;

    // Volatility
    let mean_return = if returns.is_empty() {
        0.0
    } else {
        returns.iter().sum::<f64>() / returns.len() as f64
    };
    let variance = if returns.len() > 1 {
        returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / (returns.len() - 1) as f64
    } else {
        0.0
    };
    let daily_vol = variance.sqrt();
    let volatility = daily_vol * (252.0_f64).sqrt();

    // Sharpe ratio
    let sharpe_ratio = if volatility > 0.0 {
        annualized_return / volatility
    } else {
        0.0
    };

    // Sortino ratio (downside deviation)
    let negative_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
    let downside_variance = if negative_returns.len() > 1 {
        negative_returns.iter().map(|r| r.powi(2)).sum::<f64>() / (negative_returns.len() - 1) as f64
    } else {
        variance
    };
    let downside_vol = downside_variance.sqrt() * (252.0_f64).sqrt();
    let sortino_ratio = if downside_vol > 0.0 {
        annualized_return / downside_vol
    } else {
        0.0
    };

    // Win rate
    let winning_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl > 0.0).collect();
    let losing_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl < 0.0).collect();
    let num_trades = winning_trades.len() + losing_trades.len();
    let win_rate = if num_trades > 0 {
        winning_trades.len() as f64 / num_trades as f64
    } else {
        0.0
    };

    // Profit factor
    let gross_profit: f64 = winning_trades.iter().map(|t| t.pnl).sum();
    let gross_loss: f64 = losing_trades.iter().map(|t| t.pnl.abs()).sum();
    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else if gross_profit > 0.0 {
        f64::INFINITY
    } else {
        1.0
    };

    // Average trade return
    let avg_trade_return = if num_trades > 0 {
        returns.iter().sum::<f64>() / num_trades as f64
    } else {
        0.0
    };

    Ok(BacktestResult {
        trades,
        final_value: portfolio_value,
        total_return,
        annualized_return,
        volatility,
        sharpe_ratio,
        sortino_ratio,
        max_drawdown,
        win_rate,
        profit_factor,
        num_trades,
        avg_trade_return,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{OutputType, TraderConfig};
    use chrono::Utc;
    use ndarray::{Array1, Array2};

    fn create_test_dataset(n: usize) -> TradingDataset {
        let features = Array2::from_shape_fn((n, 10), |(i, j)| ((i + j) as f32 * 0.01).sin());
        let targets = Array1::from_shape_fn(n, |i| ((i as f32 * 0.1).sin() * 0.02) as f32);
        let timestamps = vec![Utc::now(); n];
        let prices: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0)
            .collect();

        TradingDataset {
            features,
            targets,
            timestamps,
            prices,
        }
    }

    #[test]
    fn test_backtest() {
        let config = TraderConfig {
            input_dim: 10,
            d_model: 32,
            n_heads: 2,
            n_layers: 1,
            d_ff: 64,
            n_outputs: 1,
            output_type: OutputType::Regression,
            use_flash: false,
            block_size: 8,
            ..Default::default()
        };

        let model = FlashAttentionTrader::new(config);
        let dataset = create_test_dataset(200);
        let backtest_config = BacktestConfig::default();

        let result = backtest(&model, &dataset, 50, &backtest_config).unwrap();

        println!("{}", result.summary());
        assert!(result.num_trades > 0);
    }
}
