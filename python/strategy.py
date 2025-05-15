"""
Trading Strategy and Backtesting for FlashAttention Models

This module provides:
- Backtesting framework for FlashAttention trading strategies
- Performance metrics calculation (Sharpe, Sortino, Max Drawdown)
- Visualization of backtest results

Usage:
    >>> from strategy import backtest_flash_attention_strategy
    >>> result = backtest_flash_attention_strategy(
    ...     model=model,
    ...     test_data={'X': X_test, 'y': y_test},
    ...     symbols=['BTCUSDT', 'ETHUSDT']
    ... )
    >>> print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results and metrics."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    n_trades: int
    portfolio_values: np.ndarray
    returns: np.ndarray
    positions_history: List[np.ndarray] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)


def calculate_metrics(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate comprehensive trading performance metrics.

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year

    Returns:
        Dictionary of performance metrics
    """
    if len(returns) == 0:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'volatility': 0.0
        }

    # Risk-free rate per period
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_per_period

    # Total and annualized return
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(periods_per_year)

    # Sharpe Ratio (annualized)
    if volatility > 0:
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    # Sortino Ratio (only penalize downside volatility)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        sortino = excess_returns.mean() / downside_std * np.sqrt(periods_per_year)
    else:
        sortino = sharpe if sharpe > 0 else 0.0

    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = drawdowns.min()

    # Win Rate
    win_rate = (returns > 0).mean()

    # Profit Factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else float('inf')

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'volatility': volatility
    }


def backtest_flash_attention_strategy(
    model,
    test_data: Dict[str, np.ndarray],
    symbols: List[str],
    initial_capital: float = 100000.0,
    transaction_cost: float = 0.001,
    position_size: float = 0.1,
    rebalance_threshold: float = 0.05,
    device: str = 'cuda'
) -> BacktestResult:
    """
    Backtest a trading strategy using FlashAttention model predictions.

    Strategy logic:
    - Generate predictions for future returns of each asset
    - Convert predictions to position signals via tanh (bounded to [-1, 1])
    - Scale positions by position_size parameter
    - Rebalance only when position change exceeds threshold

    Args:
        model: Trained FlashAttentionTrader model
        test_data: Dictionary with 'X' (features) and 'y' (actual returns)
        symbols: List of trading symbols
        initial_capital: Starting capital
        transaction_cost: Transaction cost as fraction of trade value
        position_size: Maximum position size as fraction of capital per asset
        rebalance_threshold: Minimum position change to trigger rebalance
        device: Device for inference ('cuda' or 'cpu')

    Returns:
        BacktestResult with performance metrics and history
    """
    import torch

    model = model.to(device)
    model.eval()

    X = test_data['X']
    y = test_data['y']

    n_samples = len(X)
    n_assets = len(symbols)

    # Portfolio tracking
    capital = initial_capital
    portfolio_values = [capital]
    positions = np.zeros(n_assets)
    positions_history = [positions.copy()]
    trades = []
    period_returns = []

    # Generate all predictions at once (more efficient)
    logger.info("Generating predictions...")
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions, _ = model(X_tensor)
        predictions = predictions.cpu().numpy()

    logger.info(f"Running backtest on {n_samples} periods...")

    for i in range(n_samples):
        pred = predictions[i]
        actual_returns = y[i]

        # Convert predictions to signals using tanh (bounds to [-1, 1])
        signals = np.tanh(pred * 10)  # Scale factor for sensitivity

        # Calculate target positions
        target_positions = signals * position_size

        # Check if rebalancing is needed
        position_changes = target_positions - positions
        should_rebalance = np.any(np.abs(position_changes) > rebalance_threshold)

        if should_rebalance:
            # Calculate transaction costs
            trade_value = np.abs(position_changes).sum() * capital
            costs = trade_value * transaction_cost

            # Record trades
            for j, symbol in enumerate(symbols):
                if abs(position_changes[j]) > rebalance_threshold:
                    trades.append({
                        'period': i,
                        'symbol': symbol,
                        'action': 'buy' if position_changes[j] > 0 else 'sell',
                        'position_change': position_changes[j],
                        'predicted_return': pred[j],
                        'actual_return': actual_returns[j]
                    })

            # Update positions
            positions = target_positions
        else:
            costs = 0

        # Calculate portfolio return
        portfolio_return = np.sum(positions * actual_returns)

        # Update capital
        capital = capital * (1 + portfolio_return) - costs
        portfolio_values.append(capital)

        # Track period return
        period_return = (capital - portfolio_values[-2]) / portfolio_values[-2]
        period_returns.append(period_return)

        positions_history.append(positions.copy())

    portfolio_values = np.array(portfolio_values)
    period_returns = np.array(period_returns)

    # Calculate metrics
    metrics = calculate_metrics(period_returns)

    return BacktestResult(
        total_return=metrics['total_return'],
        annualized_return=metrics['annualized_return'],
        sharpe_ratio=metrics['sharpe_ratio'],
        sortino_ratio=metrics['sortino_ratio'],
        max_drawdown=metrics['max_drawdown'],
        win_rate=metrics['win_rate'],
        profit_factor=metrics['profit_factor'],
        avg_trade_return=np.mean([t['actual_return'] for t in trades]) if trades else 0.0,
        n_trades=len(trades),
        portfolio_values=portfolio_values,
        returns=period_returns,
        positions_history=positions_history,
        trades=trades
    )


def plot_backtest_results(
    result: BacktestResult,
    benchmark_values: Optional[np.ndarray] = None,
    symbols: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Create visualization of backtest results.

    Args:
        result: BacktestResult from backtest
        benchmark_values: Optional benchmark portfolio values for comparison
        symbols: List of symbol names for labeling
        save_path: Path to save the figure (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed. Skipping visualization.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Portfolio Value
    ax1 = axes[0, 0]
    ax1.plot(result.portfolio_values, label='Strategy', linewidth=1.5, color='blue')
    if benchmark_values is not None:
        ax1.plot(benchmark_values, label='Benchmark', linewidth=1.5, alpha=0.7, color='gray')
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    ax2 = axes[0, 1]
    cumulative = result.portfolio_values / result.portfolio_values[0]
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.5, color='red')
    ax2.set_title(f'Drawdown (Max: {result.max_drawdown:.2%})')
    ax2.set_xlabel('Period')
    ax2.set_ylabel('Drawdown')
    ax2.grid(True, alpha=0.3)

    # 3. Returns Distribution
    ax3 = axes[1, 0]
    ax3.hist(result.returns, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax3.axvline(x=result.returns.mean(), color='green', linestyle='--', linewidth=1, label='Mean')
    ax3.set_title(f'Returns Distribution (Win Rate: {result.win_rate:.2%})')
    ax3.set_xlabel('Period Return')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Performance Summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    metrics_text = f"""
    Performance Summary
    {'=' * 35}

    Total Return:        {result.total_return:>10.2%}
    Annualized Return:   {result.annualized_return:>10.2%}
    Sharpe Ratio:        {result.sharpe_ratio:>10.2f}
    Sortino Ratio:       {result.sortino_ratio:>10.2f}
    Max Drawdown:        {result.max_drawdown:>10.2%}
    Win Rate:            {result.win_rate:>10.2%}
    Profit Factor:       {result.profit_factor:>10.2f}
    Number of Trades:    {result.n_trades:>10d}
    """

    ax4.text(0.1, 0.5, metrics_text, fontsize=11, fontfamily='monospace',
             verticalalignment='center', transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved backtest plot to {save_path}")
    else:
        plt.show()

    plt.close()


def compare_strategies(
    results: Dict[str, BacktestResult],
    save_path: Optional[str] = None
):
    """
    Compare multiple backtest results.

    Args:
        results: Dictionary mapping strategy names to BacktestResult
        save_path: Path to save comparison figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed. Skipping visualization.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Portfolio values comparison
    ax1 = axes[0]
    for name, result in results.items():
        normalized = result.portfolio_values / result.portfolio_values[0]
        ax1.plot(normalized, label=f'{name} ({result.total_return:.2%})', linewidth=1.5)

    ax1.set_title('Normalized Portfolio Value')
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Value (normalized)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Metrics comparison
    ax2 = axes[1]
    metrics = ['Sharpe', 'Sortino', 'Win Rate', 'Max DD']

    x = np.arange(len(metrics))
    width = 0.8 / len(results)

    for i, (name, result) in enumerate(results.items()):
        values = [
            result.sharpe_ratio,
            result.sortino_ratio,
            result.win_rate * 10,  # Scale for visibility
            -result.max_drawdown * 10  # Negative and scaled
        ]
        ax2.bar(x + i * width, values, width, label=name)

    ax2.set_xticks(x + width * (len(results) - 1) / 2)
    ax2.set_xticklabels(metrics)
    ax2.set_title('Strategy Metrics Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {save_path}")
    else:
        plt.show()

    plt.close()


def print_backtest_summary(result: BacktestResult):
    """Print formatted backtest summary to console."""
    print("\n" + "=" * 50)
    print("           BACKTEST SUMMARY")
    print("=" * 50)
    print(f"Total Return:        {result.total_return:>12.2%}")
    print(f"Annualized Return:   {result.annualized_return:>12.2%}")
    print(f"Sharpe Ratio:        {result.sharpe_ratio:>12.2f}")
    print(f"Sortino Ratio:       {result.sortino_ratio:>12.2f}")
    print(f"Max Drawdown:        {result.max_drawdown:>12.2%}")
    print(f"Win Rate:            {result.win_rate:>12.2%}")
    print(f"Profit Factor:       {result.profit_factor:>12.2f}")
    print(f"Number of Trades:    {result.n_trades:>12d}")
    print(f"Avg Trade Return:    {result.avg_trade_return:>12.4f}")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    # Quick test with dummy data
    print("Testing strategy module...")

    # Create dummy returns
    np.random.seed(42)
    n_periods = 252
    returns = np.random.normal(0.001, 0.02, n_periods)

    # Calculate metrics
    metrics = calculate_metrics(returns)
    print("\nSample metrics:")
    for key, value in metrics.items():
        if 'return' in key or 'drawdown' in key or 'rate' in key:
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value:.2f}")

    print("\nTest passed!")
