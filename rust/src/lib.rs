//! # FlashAttention Trading
//!
//! A Rust implementation of FlashAttention-enhanced trading models.
//!
//! This library provides efficient implementations of:
//! - Tiled (Flash-style) attention mechanism
//! - Transformer architecture for financial time series
//! - Data loading from Bybit and Yahoo Finance
//! - Backtesting framework
//!
//! ## Example
//!
//! ```rust,ignore
//! use flash_attention_trading::{
//!     data::BybitClient,
//!     model::FlashAttentionTrader,
//!     strategy::backtest,
//! };
//!
//! // Fetch data
//! let client = BybitClient::new();
//! let data = client.fetch_klines("BTCUSDT", "60", 1000)?;
//!
//! // Create model
//! let model = FlashAttentionTrader::new(config);
//!
//! // Backtest
//! let result = backtest(&model, &data)?;
//! println!("Sharpe: {:.2}", result.sharpe_ratio);
//! ```

pub mod attention;
pub mod data;
pub mod model;
pub mod strategy;

// Re-exports for convenience
pub use attention::{flash_attention, flash_attention_parallel, standard_attention, AttentionConfig};
pub use data::{
    calculate_features, fetch_bybit_klines, fetch_yahoo_data, prepare_features, BybitClient,
    OhlcvData, TradingDataset, TradingFeatures, YahooClient,
};
pub use model::{FlashAttentionTrader, OutputType, TraderConfig};
pub use strategy::{backtest, generate_signals, BacktestConfig, BacktestResult, SignalGenerator, TradingSignal};
