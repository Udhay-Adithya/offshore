"""
Simple backtesting engine for Offshore.

Implements a directional trading strategy based on model predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Container for backtest results."""

    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float

    # Time series
    equity_curve: np.ndarray
    returns: np.ndarray
    positions: np.ndarray

    # Additional info
    trades: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "num_trades": self.num_trades,
            "avg_trade_return": self.avg_trade_return,
        }

    def print_summary(self) -> None:
        """Print backtest summary."""
        print("\n" + "=" * 60)
        print("Backtest Results Summary")
        print("=" * 60)
        print(f"  Total Return:      {self.total_return:.2%}")
        print(f"  Annualized Return: {self.annualized_return:.2%}")
        print(f"  Max Drawdown:      {self.max_drawdown:.2%}")
        print(f"  Sharpe Ratio:      {self.sharpe_ratio:.2f}")
        print(f"  Win Rate:          {self.win_rate:.2%}")
        print(f"  Profit Factor:     {self.profit_factor:.2f}")
        print(f"  Number of Trades:  {self.num_trades}")
        print(f"  Avg Trade Return:  {self.avg_trade_return:.4%}")
        print("=" * 60)


class BacktestEngine:
    """
    Simple directional backtesting engine.

    Strategy:
    - If model predicts UP with prob > threshold, go long
    - If model predicts DOWN with prob > threshold, go short (if allowed)
    - Otherwise, stay flat (no position)

    Example:
        >>> engine = BacktestEngine(initial_capital=10000, prob_threshold=0.55)
        >>> result = engine.run(prices, predictions, probabilities)
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        prob_threshold: float = 0.55,
        allow_short: bool = True,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        position_size: float = 1.0,
        risk_free_rate: float = 0.04,
    ):
        """
        Initialize the backtest engine.

        Args:
            initial_capital: Starting capital.
            prob_threshold: Minimum probability to take a position.
            allow_short: Whether to allow short positions.
            transaction_cost: Transaction cost as fraction of trade value.
            slippage: Slippage as fraction of price.
            position_size: Fraction of capital to use per trade.
            risk_free_rate: Annual risk-free rate for Sharpe ratio.
        """
        self.initial_capital = initial_capital
        self.prob_threshold = prob_threshold
        self.allow_short = allow_short
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.position_size = position_size
        self.risk_free_rate = risk_free_rate

    def run(
        self,
        prices: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            prices: Array of closing prices.
            predictions: Array of predicted classes (0=down, 1=up for binary).
            probabilities: Array of prediction probabilities (n_samples, n_classes).
            timestamps: Optional array of timestamps.

        Returns:
            BacktestResult with performance metrics and equity curve.
        """
        n = len(prices)

        # Initialize arrays
        positions = np.zeros(n)  # 1=long, -1=short, 0=flat
        equity = np.zeros(n)
        equity[0] = self.initial_capital
        returns = np.zeros(n)

        trades: list[dict[str, Any]] = []
        current_position = 0
        entry_price = 0.0
        entry_idx = 0

        # Determine position for each bar
        for i in range(n):
            # Get prediction and probability
            pred = predictions[i]
            probs = probabilities[i]

            # Determine signal
            signal = 0
            if len(probs) == 2:
                # Binary classification
                if pred == 1 and probs[1] > self.prob_threshold:
                    signal = 1  # Long
                elif pred == 0 and probs[0] > self.prob_threshold and self.allow_short:
                    signal = -1  # Short
            else:
                # Multi-class: use directional signal
                # Classes 0,1 = down, 2 = flat, 3,4 = up
                up_prob = probs[3:].sum() if len(probs) > 3 else probs[-1]
                down_prob = probs[:2].sum() if len(probs) > 2 else probs[0]

                if up_prob > self.prob_threshold:
                    signal = 1
                elif down_prob > self.prob_threshold and self.allow_short:
                    signal = -1

            # Update position
            positions[i] = signal

            # Calculate returns
            if i > 0:
                price_return = (prices[i] - prices[i - 1]) / prices[i - 1]

                # Apply position from previous bar
                position_return = current_position * price_return * self.position_size

                # Apply costs on position changes
                if current_position != signal:
                    cost = self.transaction_cost + self.slippage
                    position_return -= cost

                    # Record trade
                    if current_position != 0:
                        trade_return = (prices[i] / entry_price - 1) * current_position
                        trades.append(
                            {
                                "entry_idx": entry_idx,
                                "exit_idx": i,
                                "direction": "long" if current_position > 0 else "short",
                                "entry_price": entry_price,
                                "exit_price": prices[i],
                                "return": trade_return - 2 * cost,
                                "timestamp": timestamps[i] if timestamps is not None else i,
                            }
                        )

                    if signal != 0:
                        entry_price = prices[i]
                        entry_idx = i

                returns[i] = position_return
                equity[i] = equity[i - 1] * (1 + position_return)

            current_position = signal

        # Calculate metrics
        result = self._calculate_metrics(equity, returns, trades)
        result.positions = positions

        return result

    def _calculate_metrics(
        self, equity: np.ndarray, returns: np.ndarray, trades: list[dict[str, Any]]
    ) -> BacktestResult:
        """Calculate backtest performance metrics."""

        # Total return
        total_return = (equity[-1] / equity[0]) - 1

        # Annualized return (assuming 252 trading days)
        n_days = len(equity)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = abs(drawdown.min())

        # Sharpe ratio (annualized)
        daily_returns = returns[1:]  # Exclude first day
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            excess_return = daily_returns.mean() - self.risk_free_rate / 252
            sharpe_ratio = excess_return / daily_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Win rate and profit factor
        if len(trades) > 0:
            trade_returns = [t["return"] for t in trades]
            winning_trades = [r for r in trade_returns if r > 0]
            losing_trades = [r for r in trade_returns if r < 0]

            win_rate = len(winning_trades) / len(trades)

            gross_profit = sum(winning_trades) if winning_trades else 0
            gross_loss = abs(sum(losing_trades)) if losing_trades else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            avg_trade_return = np.mean(trade_returns)
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_trade_return = 0.0

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor if profit_factor != float("inf") else 99.99,
            num_trades=len(trades),
            avg_trade_return=avg_trade_return,
            equity_curve=equity,
            returns=returns,
            positions=np.zeros(len(equity)),  # Will be set by run()
            trades=trades,
        )


def run_backtest(
    prices: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    config: Optional[dict[str, Any]] = None,
    timestamps: Optional[np.ndarray] = None,
) -> BacktestResult:
    """
    Run backtest with given predictions.

    Convenience function that creates a BacktestEngine and runs backtest.

    Args:
        prices: Array of closing prices.
        predictions: Array of predicted classes.
        probabilities: Array of prediction probabilities.
        config: Backtest configuration dictionary.
        timestamps: Optional timestamps.

    Returns:
        BacktestResult with performance metrics.
    """
    config = config or {}

    engine = BacktestEngine(
        initial_capital=config.get("initial_capital", 10000.0),
        prob_threshold=config.get("prob_threshold", 0.55),
        allow_short=config.get("allow_short", True),
        transaction_cost=config.get("transaction_cost", 0.001),
        slippage=config.get("slippage", 0.0005),
        position_size=config.get("position_size", 1.0),
        risk_free_rate=config.get("risk_free_rate", 0.04),
    )

    return engine.run(prices, predictions, probabilities, timestamps)


def compare_to_buy_and_hold(prices: np.ndarray, strategy_equity: np.ndarray) -> dict[str, float]:
    """
    Compare strategy performance to buy-and-hold.

    Args:
        prices: Price array.
        strategy_equity: Strategy equity curve.

    Returns:
        Comparison metrics.
    """
    # Buy and hold equity
    bh_equity = prices / prices[0] * strategy_equity[0]

    # Returns
    strategy_return = (strategy_equity[-1] / strategy_equity[0]) - 1
    bh_return = (bh_equity[-1] / bh_equity[0]) - 1

    # Max drawdowns
    strategy_peak = np.maximum.accumulate(strategy_equity)
    strategy_dd = abs(((strategy_equity - strategy_peak) / strategy_peak).min())

    bh_peak = np.maximum.accumulate(bh_equity)
    bh_dd = abs(((bh_equity - bh_peak) / bh_peak).min())

    return {
        "strategy_return": strategy_return,
        "buy_hold_return": bh_return,
        "excess_return": strategy_return - bh_return,
        "strategy_max_dd": strategy_dd,
        "buy_hold_max_dd": bh_dd,
        "dd_improvement": bh_dd - strategy_dd,
    }
