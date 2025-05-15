"""
Data Loading and Feature Engineering for FlashAttention Trading

This module provides utilities for fetching market data from:
- Bybit (cryptocurrency futures)
- Yahoo Finance (stocks and ETFs)

And engineering features suitable for the FlashAttention trading model.

Usage:
    >>> from data import prepare_flash_attention_data
    >>> data = prepare_flash_attention_data(
    ...     symbols=['BTCUSDT', 'ETHUSDT'],
    ...     lookback=2048,
    ...     horizon=24
    ... )
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
import requests
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_bybit_klines(
    symbol: str,
    interval: str = '60',
    limit: int = 1000,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    max_retries: int = 3
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Bybit API.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Candle interval in minutes ('1', '5', '15', '60', '240', 'D')
        limit: Number of candles to fetch (max 200 per request)
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds
        max_retries: Maximum retry attempts on failure

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume, turnover

    Example:
        >>> df = fetch_bybit_klines('BTCUSDT', interval='60', limit=1000)
        >>> print(df.head())
    """
    url = 'https://api.bybit.com/v5/market/kline'

    all_data = []
    current_end = end_time or int(datetime.now().timestamp() * 1000)

    while len(all_data) < limit:
        batch_limit = min(200, limit - len(all_data))

        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': batch_limit
        }

        if current_end:
            params['end'] = current_end

        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                data = response.json()

                if data['retCode'] != 0:
                    raise ValueError(f"API Error: {data['retMsg']}")

                batch = data['result']['list']
                if not batch:
                    break

                all_data.extend(batch)

                # Update end time for next batch
                current_end = int(batch[-1][0]) - 1
                break

            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Retry {attempt + 1}/{max_retries} for {symbol}: {e}")
                time.sleep(1)

        if len(data['result']['list']) < batch_limit:
            break

    if not all_data:
        raise ValueError(f"No data received for {symbol}")

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)

    return df.sort_values('timestamp').reset_index(drop=True)


def fetch_yahoo_data(
    symbol: str,
    period: str = '2y',
    interval: str = '1h'
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.

    Args:
        symbol: Stock ticker (e.g., 'AAPL', 'SPY')
        period: Data period ('1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
        interval: Data interval ('1m', '5m', '15m', '1h', '1d', '1wk')

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume

    Example:
        >>> df = fetch_yahoo_data('AAPL', period='1y', interval='1d')
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Install with: pip install yfinance")

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    df = df.reset_index()
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # Ensure timezone-naive timestamps
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    return df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical features for trading model.

    Features computed:
    - log_return: Log returns
    - volatility: Rolling volatility (24-period)
    - volume_ma_ratio: Volume relative to moving average
    - price_ma_ratio: Price relative to moving average
    - rsi: Relative Strength Index (14-period)
    - macd: MACD line
    - macd_signal: MACD signal line
    - bb_position: Bollinger Band position (-1 to 1)

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with additional feature columns
    """
    df = df.copy()

    # Log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Volatility (24-period rolling std of returns)
    df['volatility'] = df['log_return'].rolling(24).std()

    # Volume relative to moving average
    volume_ma = df['volume'].rolling(24).mean()
    df['volume_ma_ratio'] = df['volume'] / volume_ma

    # Price relative to moving average
    price_ma = df['close'].rolling(24).mean()
    df['price_ma_ratio'] = df['close'] / price_ma

    # RSI (14-period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'] / 100  # Normalize to [0, 1]

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = (ema12 - ema26) / df['close']  # Normalize by price
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # Bollinger Bands position
    bb_ma = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_position'] = (df['close'] - bb_ma) / (2 * bb_std + 1e-10)
    df['bb_position'] = df['bb_position'].clip(-1, 1)

    return df


def prepare_flash_attention_data(
    symbols: List[str],
    lookback: int = 2048,
    horizon: int = 24,
    data_source: str = 'bybit',
    interval: str = '60',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Dict[str, np.ndarray]:
    """
    Prepare data for FlashAttention trading model.

    The long context window (2048+) is only practical with FlashAttention.
    Standard attention would require 2048² × n_symbols memory per layer.

    Args:
        symbols: List of trading pairs/tickers
        lookback: Historical context length (FlashAttention enables 2048+)
        horizon: Prediction horizon (how many periods ahead to predict)
        data_source: 'bybit' for crypto or 'yahoo' for stocks
        interval: Data interval
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation

    Returns:
        Dictionary with:
        - X_train, X_val, X_test: Feature arrays [n_samples, lookback, n_features]
        - y_train, y_val, y_test: Target arrays [n_samples, n_symbols]
        - symbols: List of symbols
        - feature_names: List of feature names

    Example:
        >>> data = prepare_flash_attention_data(
        ...     symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
        ...     lookback=2048,
        ...     horizon=24,
        ...     data_source='bybit'
        ... )
        >>> print(f"Training samples: {len(data['X_train'])}")
    """
    logger.info(f"Fetching data for {len(symbols)} symbols...")

    all_dfs = []
    for symbol in symbols:
        logger.info(f"  Fetching {symbol}...")
        try:
            if data_source == 'bybit':
                df = fetch_bybit_klines(symbol, interval=interval, limit=lookback + horizon + 500)
            elif data_source == 'yahoo':
                df = fetch_yahoo_data(symbol)
            else:
                raise ValueError(f"Unknown data_source: {data_source}")

            df = calculate_features(df)
            df['symbol'] = symbol
            all_dfs.append(df)
            logger.info(f"    Got {len(df)} records")
        except Exception as e:
            logger.error(f"    Failed to fetch {symbol}: {e}")
            raise

    # Align all dataframes on timestamp
    logger.info("Aligning data...")
    min_len = min(len(df) for df in all_dfs)
    aligned_dfs = [df.iloc[-min_len:].reset_index(drop=True) for df in all_dfs]

    # Feature columns (same for all symbols)
    feature_cols = [
        'log_return', 'volatility', 'volume_ma_ratio',
        'price_ma_ratio', 'rsi', 'macd', 'macd_signal', 'bb_position'
    ]

    n_features = len(feature_cols) * len(symbols)
    feature_names = [f"{s}_{f}" for s in symbols for f in feature_cols]

    # Drop NaN rows
    min_valid_idx = 0
    for df in aligned_dfs:
        valid_idx = df[feature_cols].dropna().index[0]
        min_valid_idx = max(min_valid_idx, valid_idx)

    aligned_dfs = [df.iloc[min_valid_idx:].reset_index(drop=True) for df in aligned_dfs]
    min_len = len(aligned_dfs[0])

    # Create sequences
    logger.info("Creating sequences...")
    X_list, y_list = [], []

    for i in range(lookback, min_len - horizon):
        # Combine features from all symbols
        x_sample = np.zeros((lookback, n_features))
        for j, df in enumerate(aligned_dfs):
            for k, feat in enumerate(feature_cols):
                x_sample[:, j * len(feature_cols) + k] = df[feat].iloc[i-lookback:i].values

        # Target: cumulative future returns for all symbols
        y_sample = np.array([
            df['log_return'].iloc[i:i+horizon].sum()
            for df in aligned_dfs
        ])

        X_list.append(x_sample)
        y_list.append(y_sample)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Handle any remaining NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # Split data
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    logger.info(f"Total samples: {n_samples}")
    logger.info(f"Train: {train_end}, Val: {val_end - train_end}, Test: {n_samples - val_end}")

    return {
        'X_train': X[:train_end],
        'y_train': y[:train_end],
        'X_val': X[train_end:val_end],
        'y_val': y[train_end:val_end],
        'X_test': X[val_end:],
        'y_test': y[val_end:],
        'symbols': symbols,
        'feature_names': feature_names
    }


def create_dataloaders(
    data: Dict[str, np.ndarray],
    batch_size: int = 16,
    num_workers: int = 0
) -> Tuple:
    """
    Create PyTorch DataLoaders from prepared data.

    Args:
        data: Dictionary from prepare_flash_attention_data
        batch_size: Batch size for training
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    train_dataset = TensorDataset(
        torch.FloatTensor(data['X_train']),
        torch.FloatTensor(data['y_train'])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(data['X_val']),
        torch.FloatTensor(data['y_val'])
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(data['X_test']),
        torch.FloatTensor(data['y_test'])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test data loading
    print("Testing Bybit data fetch...")
    df = fetch_bybit_klines('BTCUSDT', interval='60', limit=100)
    print(f"Fetched {len(df)} records")
    print(df.head())

    print("\nCalculating features...")
    df = calculate_features(df)
    print(df[['timestamp', 'close', 'log_return', 'rsi', 'macd']].tail())

    print("\nTest passed!")
