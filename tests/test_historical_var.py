
import numpy as np
import pytest

from src.historical_var import historical_var, historical_es


def test_historical_var_95():
    returns = np.array([-0.05, -0.03, -0.01, 0.01, 0.02])

    result = historical_var(returns, 0.95)

    assert result == pytest.approx(0.046)


def test_historical_var_99():
    returns = np.array([-0.05, -0.03, -0.01, 0.01, 0.02])

    result = historical_var(returns, 0.99)

    assert result == pytest.approx(0.0492)


def test_historical_es_95():
    returns = np.array([-0.05, -0.03, -0.01, 0.01, 0.02])

    result = historical_es(returns, 0.95)

    assert result == pytest.approx(0.05)


def test_historical_es_99():
    returns = np.array([-0.05, -0.03, -0.01, 0.01, 0.02])

    result = historical_es(returns, 0.99)

    assert result == pytest.approx(0.05)


def test_es_greater_than_var():
    returns = np.array([-0.06, -0.04, -0.02, 0.01, 0.03])

    for confidence in [0.95, 0.99]:
        assert historical_es(returns, confidence) >= historical_var(
            returns, confidence
        )


def test_invalid_confidence_level():
    returns = np.array([-0.02, 0.01, 0.03])

    with pytest.raises(ValueError):
        historical_var(returns, 1.5)

    with pytest.raises(ValueError):
        historical_es(returns, -0.1)


def test_nan_returns():
    returns = np.array([-0.02, np.nan, 0.03])

    with pytest.raises(ValueError):
        historical_var(returns, 0.95)

    with pytest.raises(ValueError):
        historical_es(returns, 0.95)


def test_fractional_tail_es():
    returns = np.array([-0.10, -0.05, 0.00, 0.02, 0.04])

    result = historical_es(returns, 0.70)

    # Tail size = 5 * 0.30 = 1.5 observations
    # ES = -(-0.10 + 0.5*(-0.05)) / 1.5
    expected = 0.125 / 1.5

    assert result == pytest.approx(expected)
