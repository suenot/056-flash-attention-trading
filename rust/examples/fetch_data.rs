//! Example: Fetching data from Bybit and Yahoo Finance
//!
//! This example demonstrates how to fetch cryptocurrency and stock data
//! for use with Flash Attention trading models.
//!
//! Run with: cargo run --example fetch_data

use anyhow::Result;
use flash_attention_trading::data::{
    calculate_features, fetch_bybit_klines, fetch_yahoo_data, BybitClient, YahooClient,
};

fn main() -> Result<()> {
    env_logger::init();

    println!("Flash Attention Trading - Data Fetching Example");
    println!("================================================\n");

    // Example 1: Fetch cryptocurrency data from Bybit
    println!("1. Fetching BTC/USDT data from Bybit...");

    match fetch_bybit_klines("BTCUSDT", "60", 100) {
        Ok(data) => {
            println!("   ✓ Fetched {} hourly candles", data.len());
            if let (Some(first), Some(last)) = (data.first(), data.last()) {
                println!("   First: {} - Close: ${:.2}", first.timestamp, first.close);
                println!("   Last:  {} - Close: ${:.2}", last.timestamp, last.close);
            }

            // Calculate features
            println!("\n   Calculating technical features...");
            let features = calculate_features(&data);
            println!("   ✓ Calculated {} data points with features:", features.returns.len());
            println!("     - Returns, Log Returns");
            println!("     - Volatility (20-period)");
            println!("     - RSI (14-period)");
            println!("     - MACD (12, 26, 9)");
            println!("     - Bollinger Bands (20, 2σ)");
            println!("     - ATR (14-period)");
            println!("     - Volume MA (20-period)");
        }
        Err(e) => {
            println!("   ✗ Error fetching Bybit data: {}", e);
            println!("   (This is expected if running without network access)");
        }
    }

    // Example 2: Fetch stock data from Yahoo Finance
    println!("\n2. Fetching AAPL data from Yahoo Finance...");

    match fetch_yahoo_data("AAPL", "1d", "1mo") {
        Ok(data) => {
            println!("   ✓ Fetched {} daily candles", data.len());
            if let (Some(first), Some(last)) = (data.first(), data.last()) {
                println!("   First: {} - Close: ${:.2}", first.timestamp.format("%Y-%m-%d"), first.close);
                println!("   Last:  {} - Close: ${:.2}", last.timestamp.format("%Y-%m-%d"), last.close);
            }
        }
        Err(e) => {
            println!("   ✗ Error fetching Yahoo data: {}", e);
            println!("   (This is expected if running without network access)");
        }
    }

    // Example 3: Using the client classes directly
    println!("\n3. Using Bybit client for extended data...");

    let bybit_client = BybitClient::new();
    println!("   Bybit client initialized");

    let yahoo_client = YahooClient::new();
    println!("   Yahoo client initialized");

    // Example 4: Multiple symbols
    println!("\n4. Fetching multiple crypto symbols...");
    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    for symbol in symbols {
        match bybit_client.fetch_klines(symbol, "60", 10) {
            Ok(data) => {
                if let Some(last) = data.last() {
                    println!("   {} - ${:.2}", symbol, last.close);
                }
            }
            Err(_) => {
                println!("   {} - (data unavailable)", symbol);
            }
        }
    }

    println!("\n✓ Data fetching example complete!");
    println!("\nNote: In production, you would:");
    println!("  1. Cache data locally for faster access");
    println!("  2. Handle rate limiting appropriately");
    println!("  3. Use async requests for parallel fetching");

    Ok(())
}
