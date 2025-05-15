//! Trading strategy and backtesting module.
//!
//! Provides backtesting framework and performance metrics.

mod backtest;
mod signals;

pub use backtest::{backtest, BacktestConfig, BacktestResult, Trade};
pub use signals::{generate_signals, SignalGenerator, TradingSignal};
