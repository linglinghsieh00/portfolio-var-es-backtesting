
import numpy as np
import pytest
from scipy.stats import norm

from src.parametric_var import parametric_var, parametric_es


def test_parametric_var_95():
    mean = 0.001
    volatility = 0.02

    expected = -mean + norm.ppf(0.95) * volatility
    result = parametric_var(mean, volatility, 0.95)

    assert result == pytest.approx(expected)


def test_parametric_var_99():
    mean = 0.001
    volatility = 0.02

    expected = -mean + norm.ppf(0.99) * volatility
    result = parametric_var(mean, volatility, 0.99)

    assert result == pytest.approx(expected)


def test_parametric_es_95():
    mean = 0.001
    volatility = 0.02

    z = norm.ppf(0.95)
    expected = -mean + volatility * norm.pdf(z) / 0.05

    result = parametric_es(mean, volatility, 0.95)

    assert result == pytest.approx(expected)


def test_parametric_es_99():
    mean = 0.001
    volatility = 0.02

    z = norm.ppf(0.99)
    expected = -mean + volatility * norm.pdf(z) / 0.01

    result = parametric_es(mean, volatility, 0.99)

    assert result == pytest.approx(expected)


@pytest.mark.parametrize("confidence", [0.95, 0.99])
def test_es_greater_than_var(confidence):
    mean = 0.001
    volatility = 0.02

    assert parametric_es(
        mean, volatility, confidence
    ) >= parametric_var(
        mean, volatility, confidence
    )


@pytest.mark.parametrize(
    "confidence",
    [-0.1, 0.0, 1.0, 1.5, np.nan]
)
def test_invalid_confidence_level(confidence):
    with pytest.raises(ValueError):
        parametric_var(0.001, 0.02, confidence)

    with pytest.raises(ValueError):
        parametric_es(0.001, 0.02, confidence)


def test_negative_volatility():
    with pytest.raises(ValueError):
        parametric_var(0.001, -0.02)

    with pytest.raises(ValueError):
        parametric_es(0.001, -0.02)


def test_nonfinite_parameters():
    with pytest.raises(ValueError):
        parametric_var(np.nan, 0.02)

    with pytest.raises(ValueError):
        parametric_es(0.001, np.inf)


def test_zero_volatility():
    mean = 0.001

    assert parametric_var(mean, 0.0) == pytest.approx(-mean)
    assert parametric_es(mean, 0.0) == pytest.approx(-mean)
