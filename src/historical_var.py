
import numpy as np


def _validate_returns(returns, confidence_level):
    """Validate historical returns and confidence level."""

    returns = np.asarray(returns, dtype=float)

    if returns.ndim != 1 or returns.size == 0:
        raise ValueError("Returns must be a non-empty 1D array.")

    if not np.all(np.isfinite(returns)):
        raise ValueError("Returns must contain only finite values.")

    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")

    return returns


def historical_var(returns, confidence_level=0.95):
    """Calculate Historical Simulation VaR."""

    returns = _validate_returns(returns, confidence_level)

    alpha = 1 - confidence_level

    var = -np.quantile(
        returns,
        alpha,
        method="linear"
    )

    return float(var)


def historical_es(returns, confidence_level=0.95):
    """Calculate fractional-tail Historical Expected Shortfall."""

    returns = _validate_returns(returns, confidence_level)

    sorted_returns = np.sort(returns)

    alpha = 1 - confidence_level
    tail_size = len(sorted_returns) * alpha

    k = int(np.floor(tail_size))
    fraction = tail_size - k

    tail_sum = np.sum(sorted_returns[:k])

    if fraction > 0:
        tail_sum += fraction * sorted_returns[k]

    es = -tail_sum / tail_size

    return float(es)
