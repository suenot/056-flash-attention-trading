"""
Chapter 58: FlashAttention for Algorithmic Trading

This module provides implementations of FlashAttention-enhanced trading models,
including data loading from Bybit and Yahoo Finance, model training, and backtesting.

Key Components:
- FlashAttentionTrader: Transformer model with FlashAttention for price prediction
- DataLoader: Fetch and prepare data from Bybit and Yahoo Finance
- Strategy: Trading strategy and backtesting framework

Example:
    >>> from flash_attention_trading import FlashAttentionTrader, prepare_data
    >>> data = prepare_data(['BTCUSDT', 'ETHUSDT'], lookback=2048)
    >>> model = FlashAttentionTrader(input_dim=data['X'].shape[-1], n_outputs=2)
    >>> # Train and backtest...
"""

from .model import (
    FlashAttentionTrader,
    FlashMultiHeadAttention,
    FlashTransformerBlock,
    FLASH_AVAILABLE
)
from .data import (
    fetch_bybit_klines,
    fetch_yahoo_data,
    prepare_flash_attention_data
)
from .strategy import (
    BacktestResult,
    backtest_flash_attention_strategy,
    calculate_metrics
)

__version__ = '1.0.0'
__author__ = 'Machine Learning for Trading'

__all__ = [
    # Model
    'FlashAttentionTrader',
    'FlashMultiHeadAttention',
    'FlashTransformerBlock',
    'FLASH_AVAILABLE',
    # Data
    'fetch_bybit_klines',
    'fetch_yahoo_data',
    'prepare_flash_attention_data',
    # Strategy
    'BacktestResult',
    'backtest_flash_attention_strategy',
    'calculate_metrics',
]
