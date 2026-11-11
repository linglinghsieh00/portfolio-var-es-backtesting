"""Fully rebalanced constant-mix portfolio return and P&L calculations."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from .config import WEIGHTS, INITIAL_VALUE


def calculate_portfolio(prices: pd.DataFrame, weights=WEIGHTS, initial_value=INITIAL_VALUE):
    """Daily-rebalanced, constant-target-weight *one-day* returns and dollar P&L.

    Dollar P&L is a constant-notional (initial_value) hypothetical daily P&L,
    not the daily dollar P&L of a compounding self-financing strategy.
    """
    w = pd.Series(weights, dtype=float)
    if set(w.index) != set(prices.columns):
        raise ValueError("Portfolio weights must cover precisely the price tickers")
    if not np.isfinite(w.to_numpy()).all() or (w < 0).any() or not math.isclose(float(w.sum()), 1.0, abs_tol=1e-10):
        raise ValueError("Weights must be nonnegative, finite and sum to 1")
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("initial_value must be positive and finite")
    if prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("Prices must be complete and strictly positive")
    returns = prices.loc[:, w.index].pct_change(fill_method=None).iloc[1:]
    if returns.empty:
        raise ValueError("Need at least two price observations")
    portfolio_returns = returns.dot(w)
    result = pd.DataFrame(index=returns.index)
    result.index.name = "Date"
    result["portfolio_return"] = portfolio_returns
    result["realized_pnl_usd"] = initial_value * portfolio_returns
    result["realized_loss_usd"] = -result["realized_pnl_usd"]
    result["wealth_index"] = (1.0 + portfolio_returns).cumprod()
    result["cumulative_return"] = result["wealth_index"] - 1.0
    return returns, result
