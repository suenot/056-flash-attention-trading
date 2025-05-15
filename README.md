# Chapter 58: FlashAttention for Algorithmic Trading

This chapter explores **FlashAttention**, an IO-aware exact attention algorithm that enables faster and more memory-efficient Transformer training and inference. We apply FlashAttention to financial time-series prediction, demonstrating how its efficiency gains enable longer context windows for capturing market patterns.

<p align="center">
<img src="https://i.imgur.com/9K8xYQf.png" width="70%">
</p>

## Contents

1. [Introduction to FlashAttention](#introduction-to-flashattention)
    * [The Memory Bottleneck Problem](#the-memory-bottleneck-problem)
    * [Key Innovations](#key-innovations)
    * [Benefits for Trading Models](#benefits-for-trading-models)
2. [FlashAttention Algorithm](#flashattention-algorithm)
    * [Standard Attention Review](#standard-attention-review)
    * [IO-Aware Computing](#io-aware-computing)
    * [Tiling and Recomputation](#tiling-and-recomputation)
    * [FlashAttention-2 Improvements](#flashattention-2-improvements)
3. [Trading Applications](#trading-applications)
    * [Long-Context Price Prediction](#long-context-price-prediction)
    * [High-Frequency Order Book Analysis](#high-frequency-order-book-analysis)
    * [Multi-Asset Portfolio Modeling](#multi-asset-portfolio-modeling)
4. [Practical Examples](#practical-examples)
    * [01: Data Preparation](#01-data-preparation)
    * [02: FlashAttention Transformer](#02-flashattention-transformer)
    * [03: Model Training](#03-model-training)
    * [04: Price Prediction](#04-price-prediction)
    * [05: Trading Strategy Backtesting](#05-trading-strategy-backtesting)
5. [Python Implementation](#python-implementation)
6. [Rust Implementation](#rust-implementation)
7. [Performance Benchmarks](#performance-benchmarks)
8. [Best Practices](#best-practices)
9. [Resources](#resources)

## Introduction to FlashAttention

FlashAttention is a breakthrough algorithm developed by Tri Dao et al. (2022) that makes Transformer attention computation significantly faster and more memory-efficient without sacrificing accuracy. Unlike approximate attention methods that trade quality for speed, FlashAttention computes **exact attention** while achieving 2-4x speedups.

### The Memory Bottleneck Problem

Standard Transformer attention has O(N²) time and memory complexity, where N is the sequence length. For trading applications, this creates significant limitations:

```
Traditional Attention Memory Usage:
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Sequence Length (N)    Memory for Attention Matrix    Practical Limit     │
│  ─────────────────────────────────────────────────────────────────────────  │
│       512                    ~1 MB                     ✓ Easy               │
│      2,048                  ~16 MB                     ✓ Standard           │
│      8,192                 ~256 MB                     ⚠ Challenging        │
│     32,768                  ~4 GB                      ✗ Often impossible   │
│    131,072                 ~64 GB                      ✗ Requires special   │
│                                                          hardware           │
└────────────────────────────────────────────────────────────────────────────┘
```

For trading, long sequences are essential:
- **1 year of daily data**: ~252 time steps (manageable)
- **1 month of hourly data**: ~720 time steps (manageable)
- **1 week of minute data**: ~10,080 time steps (problematic)
- **1 day of tick data**: ~100,000+ time steps (very problematic)

### Key Innovations

FlashAttention introduces two main techniques:

1. **Tiling**: Breaks the attention computation into smaller blocks that fit in GPU SRAM
2. **Recomputation**: Recomputes attention in the backward pass instead of storing large intermediate matrices

```
Standard Attention Flow (Memory-Intensive):
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│    Q, K, V                                                                   │
│       │                                                                      │
│       ▼                                                                      │
│   ┌───────────────┐                                                          │
│   │ Compute S=QK^T │  ← Store entire N×N matrix in HBM (expensive!)         │
│   └───────┬───────┘                                                          │
│           │                                                                  │
│           ▼                                                                  │
│   ┌───────────────┐                                                          │
│   │ Compute P=softmax(S) │  ← Store another N×N matrix                      │
│   └───────┬───────┘                                                          │
│           │                                                                  │
│           ▼                                                                  │
│   ┌───────────────┐                                                          │
│   │ Compute O=PV  │  ← Finally compute output                               │
│   └───────────────┘                                                          │
│                                                                              │
│   Total HBM reads/writes: O(N² + N²) = O(N²)                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

FlashAttention Flow (IO-Efficient):
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│    Q, K, V (in HBM)                                                          │
│       │                                                                      │
│       │  Load blocks of Q, K, V into SRAM                                   │
│       ▼                                                                      │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │                        FOR each block:                                  │ │
│   │   ┌─────────────────────────────────────────────────────────────────┐ │ │
│   │   │ 1. Load Q_block, K_block, V_block from HBM to SRAM              │ │ │
│   │   │ 2. Compute S_block = Q_block × K_block^T  (in SRAM)             │ │ │
│   │   │ 3. Compute P_block = softmax(S_block)      (in SRAM)            │ │ │
│   │   │ 4. Compute O_block = P_block × V_block     (in SRAM)            │ │ │
│   │   │ 5. Update running output and statistics                         │ │ │
│   │   │ 6. Write only final output to HBM                               │ │ │
│   │   └─────────────────────────────────────────────────────────────────┘ │ │
│   └───────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│   Total HBM reads/writes: O(N² / M) where M = SRAM size                     │
│   Typically 10-20x fewer memory accesses!                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Benefits for Trading Models

| Benefit | Standard Attention | FlashAttention | Impact on Trading |
|---------|-------------------|----------------|-------------------|
| Memory | O(N²) | O(N) | Handle 10x longer sequences |
| Speed | Baseline | 2-4x faster | Faster backtests, real-time inference |
| Accuracy | Exact | Exact | No quality compromise |
| Context | ~2K tokens typical | ~16K+ tokens | Capture longer market patterns |

## FlashAttention Algorithm

### Standard Attention Review

The standard attention mechanism computes:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where:
- Q (Query): What we're looking for
- K (Key): What information we have
- V (Value): The actual information content
- d_k: Dimension of keys (for scaling)

For financial time series:
- Q might represent "current market state"
- K might represent "historical patterns"
- V contains the actual price/volume information

### IO-Aware Computing

The key insight of FlashAttention is that GPU memory has a hierarchy:

```
GPU Memory Hierarchy:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         SRAM (On-chip)                               │   │
│   │   • Size: ~20 MB (A100)                                              │   │
│   │   • Speed: ~19 TB/s                                                  │   │
│   │   • Latency: ~1 cycle                                                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                   │                                          │
│                                   │ ← Bottleneck!                           │
│                                   ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         HBM (Off-chip)                               │   │
│   │   • Size: 40-80 GB (A100)                                            │   │
│   │   • Speed: ~2 TB/s                                                   │   │
│   │   • Latency: ~100s cycles                                            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   SRAM is ~10x faster than HBM but ~1000x smaller                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Standard attention** writes N×N intermediate matrices to HBM repeatedly.
**FlashAttention** keeps everything in SRAM using tiling.

### Tiling and Recomputation

FlashAttention processes attention in tiles:

```python
# Pseudocode for FlashAttention forward pass
def flash_attention_forward(Q, K, V, block_size=256):
    """
    IO-aware attention computation.

    Key ideas:
    1. Process Q, K, V in blocks that fit in SRAM
    2. Maintain running statistics for softmax normalization
    3. Never materialize the full N×N attention matrix
    """
    N, d = Q.shape
    O = zeros_like(Q)  # Output
    l = zeros(N)       # Running sum for softmax denominator
    m = full(N, -inf)  # Running max for numerical stability

    # Process K, V in blocks
    for j in range(0, N, block_size):
        Kj = K[j:j+block_size]
        Vj = V[j:j+block_size]

        # Process Q in blocks
        for i in range(0, N, block_size):
            Qi = Q[i:i+block_size]

            # Compute block of attention scores (in SRAM)
            Sij = Qi @ Kj.T / sqrt(d)

            # Update running max
            m_new = maximum(m[i:i+block_size], Sij.max(axis=-1))

            # Compute local softmax with correction
            P_ij = exp(Sij - m_new[:, None])

            # Update running sum
            l_new = exp(m[i:i+block_size] - m_new) * l[i:i+block_size] + P_ij.sum(axis=-1)

            # Update output with correction factor
            O[i:i+block_size] = (
                exp(m[i:i+block_size] - m_new)[:, None] * O[i:i+block_size] +
                P_ij @ Vj
            )

            # Save new statistics
            m[i:i+block_size] = m_new
            l[i:i+block_size] = l_new

    # Final normalization
    O = O / l[:, None]
    return O
```

The critical insight is the **online softmax** trick: we can compute softmax incrementally by tracking running max and sum, then applying correction factors.

### FlashAttention-2 Improvements

FlashAttention-2 (Dao, 2023) improves upon the original with:

1. **Reduced non-matmul FLOPs**: Modern GPUs have specialized Tensor Cores that make matmul 16x faster than other operations. FlashAttention-2 minimizes non-matmul operations.

2. **Better parallelism**: Parallelizes over sequence length dimension in addition to batch and heads, enabling better GPU utilization for long sequences.

3. **Improved work partitioning**: Better distribution of work between warps within each thread block.

```
FlashAttention vs FlashAttention-2 Performance:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   Metric                 FlashAttention    FlashAttention-2    Improvement  │
│   ─────────────────────────────────────────────────────────────────────────  │
│   GPU Utilization        25-40%            50-73%              ~2x           │
│   Training Speed         Fast              Very Fast           ~2x           │
│   Sequence Length        Up to 16K         Up to 64K+          4x+           │
│   Memory Efficiency      Linear            Linear              Same          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Trading Applications

### Long-Context Price Prediction

FlashAttention enables models to consider much longer historical context:

```python
# Traditional approach: Limited context
lookback_traditional = 512  # ~1 month of hourly data

# With FlashAttention: Extended context
lookback_flash = 4096  # ~6 months of hourly data
# or
lookback_flash = 16384  # ~2 years of hourly data

# This matters because:
# - Seasonal patterns may span months
# - Major market events have long-lasting effects
# - Cross-asset correlations evolve over time
```

**Example: Crypto Market Prediction**

```python
import torch
from flash_attention_trading import FlashAttentionTrader

# Configure for crypto trading
config = {
    'context_length': 8192,    # 2+ weeks of hourly data
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 6,
    'symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT'],
    'data_source': 'bybit',
    'use_flash_attention': True  # Enable FlashAttention
}

model = FlashAttentionTrader(**config)

# Standard attention would require 8192² × 4 bytes = 256MB per layer
# FlashAttention reduces this to O(8192) = ~32KB per layer
```

### High-Frequency Order Book Analysis

For order book data, we often need to process many levels and rapid updates:

```python
# Order book analysis with FlashAttention
class OrderBookFlashAttention:
    def __init__(self, n_levels=50, history_length=1000):
        """
        Analyze limit order book with attention.

        n_levels: Number of bid/ask levels to consider
        history_length: Number of historical snapshots
        """
        self.sequence_length = n_levels * 2 * history_length
        # Traditional: 100,000² attention = 40 GB
        # FlashAttention: Handles this easily with ~100 MB

    def predict_mid_price_movement(self, order_book_history):
        """
        Use attention to find patterns in order book dynamics.

        Attention can discover:
        - Which price levels are most predictive
        - How imbalances at different levels interact
        - Temporal patterns in order flow
        """
        pass
```

### Multi-Asset Portfolio Modeling

FlashAttention enables modeling relationships across many assets:

```python
# Multi-asset portfolio with cross-asset attention
class FlashPortfolioModel:
    def __init__(self, n_assets=100, lookback=2048):
        """
        Model with cross-asset attention.

        With n_assets=100 and lookback=2048:
        - Sequence length = 100 × 2048 = 204,800
        - Traditional attention: 204,800² = 158 GB (impossible!)
        - FlashAttention: Handles it with ~1 GB
        """
        self.model = TransformerWithFlashAttention(
            seq_len=n_assets * lookback,
            d_model=128,
            n_heads=8,
            n_layers=4,
            use_flash=True
        )
```

## Practical Examples

### 01: Data Preparation

```python
# python/data_loader.py

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import requests
from datetime import datetime, timedelta

def fetch_bybit_klines(
    symbol: str,
    interval: str = '60',  # 1 hour
    limit: int = 1000
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Bybit.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Candle interval in minutes
        limit: Number of candles to fetch

    Returns:
        DataFrame with OHLCV data
    """
    url = 'https://api.bybit.com/v5/market/kline'

    params = {
        'category': 'linear',
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data['retCode'] != 0:
        raise ValueError(f"API Error: {data['retMsg']}")

    df = pd.DataFrame(data['result']['list'], columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)

    return df.sort_values('timestamp').reset_index(drop=True)


def prepare_flash_attention_data(
    symbols: List[str],
    lookback: int = 2048,
    horizon: int = 24
) -> Dict[str, np.ndarray]:
    """
    Prepare data for FlashAttention trading model.

    The long context window (2048) is only practical with FlashAttention.
    Standard attention would require 2048² × n_symbols = prohibitive memory.

    Args:
        symbols: List of trading pairs
        lookback: Historical context length
        horizon: Prediction horizon

    Returns:
        Dictionary with X (features) and y (targets)
    """
    all_data = []

    for symbol in symbols:
        df = fetch_bybit_klines(symbol, limit=lookback + horizon + 100)

        # Calculate features
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(24).std()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(24).mean()
        df['price_ma_ratio'] = df['close'] / df['close'].rolling(24).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))

        df = df.dropna()
        all_data.append(df)

    # Align all dataframes
    min_len = min(len(df) for df in all_data)
    aligned = [df.iloc[-min_len:].reset_index(drop=True) for df in all_data]

    # Create sequences
    features = ['log_return', 'volatility', 'volume_ma_ratio', 'price_ma_ratio', 'rsi']
    n_features = len(features) * len(symbols)

    X, y = [], []

    for i in range(lookback, min_len - horizon):
        # Combine features from all symbols
        x_sample = np.zeros((lookback, n_features))
        for j, df in enumerate(aligned):
            for k, feat in enumerate(features):
                x_sample[:, j * len(features) + k] = df[feat].iloc[i-lookback:i].values

        # Target: future returns for all symbols
        y_sample = np.array([
            df['log_return'].iloc[i:i+horizon].sum()
            for df in aligned
        ])

        X.append(x_sample)
        y.append(y_sample)

    return {
        'X': np.array(X),
        'y': np.array(y),
        'symbols': symbols,
        'feature_names': [f"{s}_{f}" for s in symbols for f in features]
    }
```

### 02: FlashAttention Transformer

```python
# python/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Try to import FlashAttention
try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    print("FlashAttention not installed. Using standard attention fallback.")


class FlashMultiHeadAttention(nn.Module):
    """
    Multi-head attention with FlashAttention support.
    Falls back to standard attention if FlashAttention is unavailable.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_flash = use_flash and FLASH_AVAILABLE

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with FlashAttention or standard attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            Output tensor and optionally attention weights
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k)

        if self.use_flash and not return_attention:
            # Use FlashAttention (does not support returning attention weights)
            # FlashAttention expects [batch, seq, n_heads, d_k]
            output = flash_attn_func(Q, K, V, dropout_p=self.dropout.p if self.training else 0.0)
            output = output.view(batch_size, seq_len, self.d_model)
            attn_weights = None
        else:
            # Standard attention (fallback or when we need attention weights)
            Q = Q.transpose(1, 2)  # [batch, n_heads, seq_len, d_k]
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)

            # Compute attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            output = torch.matmul(attn_weights, V)
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.out_proj(output)

        return output, attn_weights


class FlashTransformerBlock(nn.Module):
    """Transformer block with FlashAttention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()

        self.attention = FlashMultiHeadAttention(d_model, n_heads, dropout, use_flash)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # Pre-norm architecture
        residual = x
        x = self.norm1(x)
        attn_out, attn_weights = self.attention(x, mask, return_attention)
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)

        return x, attn_weights


class FlashAttentionTrader(nn.Module):
    """
    Transformer model for trading with FlashAttention.

    Benefits of FlashAttention for trading:
    1. Handle longer sequences (more historical data)
    2. Faster training and inference
    3. Lower memory usage (fit larger models)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 4096,
        n_outputs: int = 1,
        output_type: str = 'regression',
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.n_outputs = n_outputs
        self.output_type = output_type
        self.use_flash = use_flash and FLASH_AVAILABLE

        # Input embedding
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Transformer layers with FlashAttention
        self.layers = nn.ModuleList([
            FlashTransformerBlock(d_model, n_heads, d_ff, dropout, use_flash)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Output head
        if output_type == 'regression':
            self.head = nn.Linear(d_model, n_outputs)
        elif output_type == 'direction':
            self.head = nn.Linear(d_model, n_outputs)
        elif output_type == 'allocation':
            self.head = nn.Sequential(
                nn.Linear(d_model, n_outputs),
                nn.Tanh()
            )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            mask: Optional attention mask
            return_attention: Whether to return attention weights from all layers

        Returns:
            Output predictions and optionally attention weights
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Transformer layers
        all_attention = []
        for layer in self.layers:
            x, attn = layer(x, mask, return_attention)
            if return_attention and attn is not None:
                all_attention.append(attn)

        x = self.norm(x)

        # Use last token for prediction (like classification token)
        x = x[:, -1, :]

        # Output head
        output = self.head(x)

        if return_attention:
            return output, all_attention
        return output, None

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss based on output type."""

        if self.output_type == 'regression':
            return F.mse_loss(predictions, targets)
        elif self.output_type == 'direction':
            return F.binary_cross_entropy_with_logits(predictions, (targets > 0).float())
        elif self.output_type == 'allocation':
            # Maximize returns (negative loss)
            return -torch.mean(predictions * targets)
        else:
            raise ValueError(f"Unknown output type: {self.output_type}")
```

### 03: Model Training

```python
# python/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional
import logging

from model import FlashAttentionTrader
from data_loader import prepare_flash_attention_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(
    model: FlashAttentionTrader,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-4,
    device: str = 'cuda'
) -> Dict[str, list]:
    """
    Train the FlashAttention trading model.

    Args:
        model: FlashAttentionTrader model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on

    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
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

        logger.info(f'Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            logger.info(f'Saved best model with val_loss = {val_loss:.6f}')

        scheduler.step()

    return history


def main():
    """Main training script."""

    # Configuration
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
    lookback = 2048  # Extended context thanks to FlashAttention
    horizon = 24
    batch_size = 16
    epochs = 50

    logger.info("Preparing data...")
    data = prepare_flash_attention_data(symbols, lookback, horizon)

    # Split data
    n_samples = len(data['X'])
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    X_train = torch.FloatTensor(data['X'][:train_size])
    y_train = torch.FloatTensor(data['y'][:train_size])
    X_val = torch.FloatTensor(data['X'][train_size:train_size+val_size])
    y_val = torch.FloatTensor(data['y'][train_size:train_size+val_size])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size
    )

    # Create model
    model = FlashAttentionTrader(
        input_dim=len(data['feature_names']),
        d_model=256,
        n_heads=8,
        n_layers=6,
        max_seq_len=lookback,
        n_outputs=len(symbols),
        output_type='regression',
        use_flash=True
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Using FlashAttention: {model.use_flash}")

    # Train
    history = train_model(model, train_loader, val_loader, epochs)

    logger.info("Training complete!")
    return history


if __name__ == '__main__':
    main()
```

### 04: Price Prediction

```python
# python/predict.py

import torch
import numpy as np
import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt

from model import FlashAttentionTrader
from data_loader import prepare_flash_attention_data, fetch_bybit_klines


def load_model(checkpoint_path: str, config: Dict) -> FlashAttentionTrader:
    """Load trained model from checkpoint."""
    model = FlashAttentionTrader(**config)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model


def predict_returns(
    model: FlashAttentionTrader,
    X: np.ndarray,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Generate return predictions.

    Args:
        model: Trained FlashAttentionTrader
        X: Input features [n_samples, seq_len, n_features]
        device: Device for inference

    Returns:
        Predicted returns [n_samples, n_assets]
    """
    model = model.to(device)
    model.eval()

    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        predictions, _ = model(X_tensor)

    return predictions.cpu().numpy()


def predict_with_attention_analysis(
    model: FlashAttentionTrader,
    X: np.ndarray,
    symbols: List[str],
    device: str = 'cuda'
) -> Dict:
    """
    Make predictions and analyze attention patterns.

    Note: Attention analysis requires standard attention (FlashAttention
    doesn't return attention weights). This is useful for interpretability.
    """
    model = model.to(device)
    model.eval()

    # Temporarily disable FlashAttention to get attention weights
    original_use_flash = model.use_flash
    model.use_flash = False
    for layer in model.layers:
        layer.attention.use_flash = False

    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        predictions, attention_weights = model(X_tensor, return_attention=True)

    # Restore FlashAttention setting
    model.use_flash = original_use_flash
    for layer in model.layers:
        layer.attention.use_flash = original_use_flash

    # Analyze attention patterns
    # attention_weights is a list of [batch, n_heads, seq_len, seq_len]
    if attention_weights:
        # Average attention over heads and layers
        avg_attention = torch.stack([
            attn.mean(dim=1) for attn in attention_weights
        ]).mean(dim=0)

        # Which positions attend to which?
        # Focus on last position (prediction position)
        last_pos_attention = avg_attention[:, -1, :]  # [batch, seq_len]
    else:
        last_pos_attention = None

    return {
        'predictions': predictions.cpu().numpy(),
        'attention_to_history': last_pos_attention.cpu().numpy() if last_pos_attention is not None else None
    }


def visualize_attention(
    attention: np.ndarray,
    timestamps: pd.DatetimeIndex,
    save_path: str = 'attention_visualization.png'
):
    """Visualize which historical periods the model focuses on."""

    plt.figure(figsize=(14, 6))

    # Average over batch dimension
    avg_attention = attention.mean(axis=0)

    plt.plot(timestamps, avg_attention, linewidth=0.5, alpha=0.7)
    plt.fill_between(timestamps, avg_attention, alpha=0.3)

    plt.xlabel('Historical Time')
    plt.ylabel('Attention Weight')
    plt.title('Model Attention to Historical Data')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Attention visualization saved to {save_path}")


def main():
    """Example prediction script."""

    # Load configuration
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
    lookback = 2048

    config = {
        'input_dim': len(symbols) * 5,  # 5 features per symbol
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 6,
        'max_seq_len': lookback,
        'n_outputs': len(symbols),
        'output_type': 'regression',
        'use_flash': True
    }

    # Load model
    model = load_model('best_model.pt', config)

    # Prepare latest data
    data = prepare_flash_attention_data(symbols, lookback, horizon=1)

    # Get latest sample
    X_latest = data['X'][-1:]

    # Predict
    result = predict_with_attention_analysis(model, X_latest, symbols)

    print("\nPredicted Returns (next 24h):")
    for i, symbol in enumerate(symbols):
        pred = result['predictions'][0, i]
        direction = "UP" if pred > 0 else "DOWN"
        print(f"  {symbol}: {pred*100:.2f}% ({direction})")

    return result


if __name__ == '__main__':
    main()
```

### 05: Trading Strategy Backtesting

```python
# python/strategy.py

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt

from model import FlashAttentionTrader
from data_loader import prepare_flash_attention_data


@dataclass
class BacktestResult:
    """Container for backtest results."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    portfolio_values: np.ndarray
    trades: List[Dict]


def calculate_metrics(returns: np.ndarray, risk_free_rate: float = 0.0) -> Dict[str, float]:
    """Calculate trading performance metrics."""

    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate

    # Sharpe Ratio (annualized)
    sharpe = np.sqrt(252) * excess_returns.mean() / (returns.std() + 1e-8)

    # Sortino Ratio (only penalize downside volatility)
    downside_returns = returns[returns < 0]
    downside_std = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 1e-8
    sortino = np.sqrt(252) * excess_returns.mean() / (downside_std + 1e-8)

    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = drawdowns.min()

    # Win Rate
    win_rate = (returns > 0).mean()

    return {
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_return': cumulative[-1] - 1
    }


def backtest_flash_attention_strategy(
    model: FlashAttentionTrader,
    test_data: Dict,
    symbols: List[str],
    initial_capital: float = 100000,
    transaction_cost: float = 0.001,
    position_size: float = 0.1,
    device: str = 'cuda'
) -> BacktestResult:
    """
    Backtest a trading strategy using FlashAttention model predictions.

    Strategy:
    - Long when predicted return > threshold
    - Short when predicted return < -threshold
    - Position size proportional to prediction confidence

    Args:
        model: Trained FlashAttentionTrader
        test_data: Test dataset with X and y
        symbols: List of trading symbols
        initial_capital: Starting capital
        transaction_cost: Cost per trade (as fraction)
        position_size: Maximum position size as fraction of capital
        device: Device for inference

    Returns:
        BacktestResult with performance metrics
    """
    import torch

    model = model.to(device)
    model.eval()

    X = test_data['X']
    y = test_data['y']  # Actual returns

    n_samples = len(X)
    n_assets = len(symbols)

    # Portfolio tracking
    capital = initial_capital
    portfolio_values = [capital]
    positions = np.zeros(n_assets)
    trades = []

    # Generate all predictions
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions, _ = model(X_tensor)
        predictions = predictions.cpu().numpy()

    # Run backtest
    for i in range(n_samples):
        pred = predictions[i]
        actual_returns = y[i]

        # Generate signals (normalized by prediction magnitude)
        signals = np.tanh(pred * 10)  # Scale and bound to [-1, 1]

        # Calculate target positions
        target_positions = signals * position_size

        # Calculate position changes and costs
        position_changes = target_positions - positions
        trade_cost = np.abs(position_changes).sum() * transaction_cost * capital

        # Record trades
        for j, symbol in enumerate(symbols):
            if abs(position_changes[j]) > 0.001:
                trades.append({
                    'step': i,
                    'symbol': symbol,
                    'action': 'buy' if position_changes[j] > 0 else 'sell',
                    'size': abs(position_changes[j]),
                    'predicted_return': pred[j],
                    'actual_return': actual_returns[j]
                })

        # Update positions
        positions = target_positions

        # Calculate returns
        portfolio_return = np.sum(positions * actual_returns)
        capital = capital * (1 + portfolio_return) - trade_cost
        portfolio_values.append(capital)

    portfolio_values = np.array(portfolio_values)
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # Calculate metrics
    metrics = calculate_metrics(daily_returns)

    return BacktestResult(
        total_return=metrics['total_return'],
        sharpe_ratio=metrics['sharpe_ratio'],
        sortino_ratio=metrics['sortino_ratio'],
        max_drawdown=metrics['max_drawdown'],
        win_rate=metrics['win_rate'],
        portfolio_values=portfolio_values,
        trades=trades
    )


def plot_backtest_results(
    result: BacktestResult,
    benchmark_values: Optional[np.ndarray] = None,
    save_path: str = 'backtest_results.png'
):
    """Plot backtest results."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Portfolio value
    ax1 = axes[0, 0]
    ax1.plot(result.portfolio_values, label='Strategy', linewidth=1.5)
    if benchmark_values is not None:
        ax1.plot(benchmark_values, label='Benchmark', linewidth=1.5, alpha=0.7)
    ax1.set_title('Portfolio Value')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[0, 1]
    cumulative = result.portfolio_values / result.portfolio_values[0]
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.5, color='red')
    ax2.set_title(f'Drawdown (Max: {result.max_drawdown:.2%})')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Drawdown')
    ax2.grid(True, alpha=0.3)

    # Returns distribution
    ax3 = axes[1, 0]
    returns = np.diff(result.portfolio_values) / result.portfolio_values[:-1]
    ax3.hist(returns, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax3.set_title(f'Returns Distribution (Win Rate: {result.win_rate:.2%})')
    ax3.set_xlabel('Daily Return')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)

    # Metrics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    metrics_text = f"""
    Performance Metrics
    {'='*30}

    Total Return: {result.total_return:.2%}
    Sharpe Ratio: {result.sharpe_ratio:.2f}
    Sortino Ratio: {result.sortino_ratio:.2f}
    Max Drawdown: {result.max_drawdown:.2%}
    Win Rate: {result.win_rate:.2%}
    Number of Trades: {len(result.trades)}
    """
    ax4.text(0.1, 0.5, metrics_text, fontsize=12, fontfamily='monospace',
             verticalalignment='center', transform=ax4.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Backtest results saved to {save_path}")


def main():
    """Run backtest example."""
    import torch

    # Configuration
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
    lookback = 2048
    horizon = 24

    print("Preparing data...")
    data = prepare_flash_attention_data(symbols, lookback, horizon)

    # Split data
    n_samples = len(data['X'])
    test_start = int(0.85 * n_samples)

    test_data = {
        'X': data['X'][test_start:],
        'y': data['y'][test_start:]
    }

    # Load model
    config = {
        'input_dim': len(data['feature_names']),
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 6,
        'max_seq_len': lookback,
        'n_outputs': len(symbols),
        'output_type': 'regression',
        'use_flash': True
    }

    model = FlashAttentionTrader(**config)
    model.load_state_dict(torch.load('best_model.pt'))

    print("Running backtest...")
    result = backtest_flash_attention_strategy(
        model=model,
        test_data=test_data,
        symbols=symbols,
        initial_capital=100000,
        transaction_cost=0.001
    )

    print(f"\nBacktest Results:")
    print(f"  Total Return: {result.total_return:.2%}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown:.2%}")
    print(f"  Win Rate: {result.win_rate:.2%}")
    print(f"  Number of Trades: {len(result.trades)}")

    plot_backtest_results(result)

    return result


if __name__ == '__main__':
    main()
```

## Python Implementation

```
python/
├── __init__.py
├── model.py                # FlashAttention Transformer
├── data_loader.py          # Bybit data loading & feature engineering
├── train.py                # Training script
├── predict.py              # Prediction utilities
├── strategy.py             # Trading strategy & backtesting
├── requirements.txt        # Python dependencies
└── examples/
    ├── 01_data_preparation.ipynb
    ├── 02_model_architecture.ipynb
    ├── 03_training.ipynb
    ├── 04_prediction.ipynb
    └── 05_backtesting.ipynb
```

### Quick Start (Python)

```bash
# Install dependencies
cd python
pip install -r requirements.txt

# Optional: Install FlashAttention (requires CUDA)
pip install flash-attn --no-build-isolation

# Fetch data and train
python data_loader.py --symbols BTCUSDT,ETHUSDT,SOLUSDT
python train.py --epochs 50 --batch-size 16

# Run backtest
python strategy.py --model best_model.pt
```

### Requirements

```
# requirements.txt
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
requests>=2.25.0
tqdm>=4.60.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
flash-attn>=2.0.0  # Optional, requires CUDA
```

## Rust Implementation

See [rust/](rust/) for a production-ready Rust implementation.

```
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                 # Library exports
│   ├── attention/
│   │   ├── mod.rs
│   │   ├── standard.rs        # Standard attention (fallback)
│   │   └── flash.rs           # Flash-style attention implementation
│   ├── model/
│   │   ├── mod.rs
│   │   ├── transformer.rs     # Transformer architecture
│   │   └── trading.rs         # Trading-specific model
│   ├── data/
│   │   ├── mod.rs
│   │   ├── bybit.rs           # Bybit API client
│   │   ├── yahoo.rs           # Yahoo Finance integration
│   │   └── features.rs        # Feature engineering
│   └── strategy/
│       ├── mod.rs
│       ├── signals.rs         # Signal generation
│       └── backtest.rs        # Backtesting engine
└── examples/
    ├── fetch_data.rs
    ├── train.rs
    └── backtest.rs
```

### Quick Start (Rust)

```bash
cd rust

# Build the project
cargo build --release

# Fetch data
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT

# Train model
cargo run --example train -- --epochs 50

# Run backtest
cargo run --example backtest
```

## Performance Benchmarks

### Memory Usage Comparison

| Sequence Length | Standard Attention | FlashAttention | Reduction |
|-----------------|-------------------|----------------|-----------|
| 512 | 1.0 MB | 0.1 MB | 10x |
| 2,048 | 16 MB | 0.4 MB | 40x |
| 8,192 | 256 MB | 1.6 MB | 160x |
| 32,768 | 4 GB | 6.4 MB | 640x |

### Speed Comparison (A100 GPU)

| Operation | Standard Attention | FlashAttention | FlashAttention-2 |
|-----------|-------------------|----------------|------------------|
| Forward (seq=2K) | 100 ms | 45 ms | 25 ms |
| Forward (seq=8K) | 1600 ms | 180 ms | 95 ms |
| Backward (seq=2K) | 300 ms | 135 ms | 70 ms |
| Backward (seq=8K) | 4800 ms | 540 ms | 280 ms |

### Trading Model Benchmarks

With FlashAttention, we can train models that would be impractical with standard attention:

| Model Configuration | Standard Attention | FlashAttention |
|--------------------|-------------------|----------------|
| 1 month hourly (720 steps) | ✓ Feasible | ✓ Fast |
| 3 months hourly (2,160 steps) | ⚠ Slow | ✓ Fast |
| 1 year hourly (8,760 steps) | ✗ OOM | ✓ Feasible |
| 1 week 15-min (672 steps) × 10 assets | ⚠ Slow | ✓ Fast |

## Best Practices

### When to Use FlashAttention

**Recommended scenarios:**
- Long time series (>1000 time steps)
- Multiple assets with cross-attention
- Real-time inference where speed matters
- GPU training where memory is limited

**May not be needed:**
- Short sequences (<512)
- Simple models without attention
- CPU-only deployment

### Model Architecture Tips

```python
# Recommended configuration for trading
config = {
    'd_model': 256,        # Balance capacity and speed
    'n_heads': 8,          # Standard choice
    'n_layers': 4-8,       # More layers for complex patterns
    'max_seq_len': 4096,   # Leverage FlashAttention for long context
    'dropout': 0.1,        # Regularization
}

# For high-frequency trading (lower latency)
hft_config = {
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 3,
    'max_seq_len': 512,
}
```

### Common Pitfalls

1. **Not using mixed precision**: FlashAttention works best with FP16/BF16
   ```python
   # Use automatic mixed precision
   with torch.autocast(device_type='cuda', dtype=torch.float16):
       output = model(x)
   ```

2. **Ignoring sequence length alignment**: FlashAttention is optimized for specific block sizes
   ```python
   # Pad sequences to multiple of 64 for optimal performance
   seq_len = ((seq_len + 63) // 64) * 64
   ```

3. **Expecting attention weights**: FlashAttention doesn't store the attention matrix
   ```python
   # For interpretability, temporarily disable FlashAttention
   model.use_flash = False
   output, attention = model(x, return_attention=True)
   ```

## Resources

### Papers

- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) — Original paper (2022)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691) — Improved version (2023)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Latest iteration (2024)

### Implementations

- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention) — Official implementation
- [PyTorch SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — PyTorch's built-in flash attention
- [xFormers](https://github.com/facebookresearch/xformers) — Facebook's memory-efficient attention

### Related Chapters

- [Chapter 52: Performer Efficient Attention](../52_performer_efficient_attention) — Approximate linear attention
- [Chapter 54: Reformer LSH Attention](../54_reformer_lsh_attention) — Locality-sensitive hashing attention
- [Chapter 57: Longformer Financial](../57_longformer_financial) — Sliding window attention

---

## Difficulty Level

**Advanced**

Prerequisites:
- Transformer architecture and self-attention
- GPU memory hierarchy and optimization
- PyTorch or similar deep learning framework
- Basic trading strategy knowledge
