#!/usr/bin/env python3
"""
Example: End-to-End FlashAttention Trading Pipeline

This script demonstrates:
1. Fetching data from Bybit (cryptocurrency)
2. Feature engineering for trading
3. Training a FlashAttention-enhanced Transformer
4. Backtesting the trading strategy
5. Analyzing results

Usage:
    python example_usage.py

Note: For full FlashAttention benefits, run on CUDA-enabled GPU with flash-attn installed.
"""

import torch
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
from model import FlashAttentionTrader, FLASH_AVAILABLE
from data import prepare_flash_attention_data, create_dataloaders
from strategy import (
    backtest_flash_attention_strategy,
    plot_backtest_results,
    print_backtest_summary
)


def main():
    """Run the complete trading pipeline."""

    # =================================================================
    # Configuration
    # =================================================================

    config = {
        # Data settings
        'symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
        'lookback': 512,  # Use 512 for demo; FlashAttention enables 2048+
        'horizon': 24,    # Predict 24 hours ahead
        'data_source': 'bybit',

        # Model settings
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 4,
        'dropout': 0.1,
        'use_flash': True,

        # Training settings
        'batch_size': 8,
        'epochs': 10,  # Use more epochs for better results
        'learning_rate': 1e-4,

        # Strategy settings
        'initial_capital': 100000,
        'transaction_cost': 0.001,
        'position_size': 0.1,
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    logger.info(f"FlashAttention available: {FLASH_AVAILABLE}")

    # =================================================================
    # Step 1: Prepare Data
    # =================================================================

    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Preparing Data")
    logger.info("=" * 60)

    try:
        data = prepare_flash_attention_data(
            symbols=config['symbols'],
            lookback=config['lookback'],
            horizon=config['horizon'],
            data_source=config['data_source']
        )
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        logger.info("Using synthetic data for demonstration...")
        data = generate_synthetic_data(config)

    input_dim = data['X_train'].shape[-1]
    n_outputs = len(config['symbols'])

    logger.info(f"Training samples: {len(data['X_train'])}")
    logger.info(f"Validation samples: {len(data['X_val'])}")
    logger.info(f"Test samples: {len(data['X_test'])}")
    logger.info(f"Input dimension: {input_dim}")
    logger.info(f"Output dimension: {n_outputs}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data, batch_size=config['batch_size']
    )

    # =================================================================
    # Step 2: Create Model
    # =================================================================

    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Creating Model")
    logger.info("=" * 60)

    model = FlashAttentionTrader(
        input_dim=input_dim,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        max_seq_len=config['lookback'],
        n_outputs=n_outputs,
        output_type='regression',
        dropout=config['dropout'],
        use_flash=config['use_flash']
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    logger.info(f"Using FlashAttention: {model.use_flash}")

    model = model.to(device)

    # =================================================================
    # Step 3: Train Model
    # =================================================================

    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Training Model")
    logger.info("=" * 60)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs']
    )

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions, _ = model(batch_x)
            loss = model.compute_loss(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                predictions, _ = model(batch_x)
                loss = model.compute_loss(predictions, batch_y)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        logger.info(
            f"Epoch {epoch+1}/{config['epochs']}: "
            f"Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

        scheduler.step()

    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    logger.info(f"Best validation loss: {best_val_loss:.6f}")

    # =================================================================
    # Step 4: Backtest Strategy
    # =================================================================

    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Backtesting Strategy")
    logger.info("=" * 60)

    test_data = {
        'X': data['X_test'],
        'y': data['y_test']
    }

    result = backtest_flash_attention_strategy(
        model=model,
        test_data=test_data,
        symbols=config['symbols'],
        initial_capital=config['initial_capital'],
        transaction_cost=config['transaction_cost'],
        position_size=config['position_size'],
        device=device
    )

    # =================================================================
    # Step 5: Results Analysis
    # =================================================================

    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Results Analysis")
    logger.info("=" * 60)

    print_backtest_summary(result)

    # Save plot
    try:
        plot_backtest_results(result, save_path='backtest_results.png')
        logger.info("Saved backtest visualization to backtest_results.png")
    except Exception as e:
        logger.warning(f"Could not create plot: {e}")

    # Sample predictions
    logger.info("\nSample predictions vs actual returns:")
    for i, trade in enumerate(result.trades[:5]):
        logger.info(
            f"  {trade['symbol']}: "
            f"Predicted {trade['predicted_return']:.4f}, "
            f"Actual {trade['actual_return']:.4f}, "
            f"Action: {trade['action']}"
        )

    return model, result


def generate_synthetic_data(config):
    """Generate synthetic data for demonstration when API is unavailable."""
    logger.info("Generating synthetic trading data...")

    n_samples = 500
    lookback = config['lookback']
    n_symbols = len(config['symbols'])
    n_features = n_symbols * 8  # 8 features per symbol

    # Generate random walk prices with some patterns
    np.random.seed(42)

    X = np.random.randn(n_samples, lookback, n_features).astype(np.float32) * 0.1
    y = np.random.randn(n_samples, n_symbols).astype(np.float32) * 0.02

    # Add some signal
    for i in range(n_samples):
        for j in range(n_symbols):
            trend = X[i, -24:, j * 8].mean()  # Use recent returns
            y[i, j] += trend * 0.5  # Add trend following signal

    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    return {
        'X_train': X[:train_size],
        'y_train': y[:train_size],
        'X_val': X[train_size:train_size + val_size],
        'y_val': y[train_size:train_size + val_size],
        'X_test': X[train_size + val_size:],
        'y_test': y[train_size + val_size:],
        'symbols': config['symbols'],
        'feature_names': [f"feature_{i}" for i in range(n_features)]
    }


if __name__ == '__main__':
    model, result = main()
