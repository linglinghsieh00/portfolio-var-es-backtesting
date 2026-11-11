
import numpy as np
import pytest

from src.monte_carlo_var import monte_carlo_var_es
from src.parametric_var import parametric_var, parametric_es


@pytest.fixture
def model_inputs():
    mean = np.array([0.0005, 0.0003])
    covariance = np.array([
        [0.0004, 0.0001],
        [0.0001, 0.0003]
    ])
    weights = np.array([0.6, 0.4])

    return mean, covariance, weights


def test_reproducibility(model_inputs):
    mean, cov, weights = model_inputs

    result_1 = monte_carlo_var_es(
        mean, cov, weights, 10000, seed=42
    )

    result_2 = monte_carlo_var_es(
        mean, cov, weights, 10000, seed=42
    )

    assert result_1 == result_2


def test_es_greater_than_var(model_inputs):
    mean, cov, weights = model_inputs

    result = monte_carlo_var_es(
        mean, cov, weights, 10000, seed=42
    )

    assert result["es_95"] >= result["var_95"]
    assert result["es_99"] >= result["var_99"]


def test_confidence_level_ordering(model_inputs):
    mean, cov, weights = model_inputs

    result = monte_carlo_var_es(
        mean, cov, weights, 10000, seed=42
    )

    assert result["var_99"] >= result["var_95"]
    assert result["es_99"] >= result["es_95"]


def test_monte_carlo_close_to_parametric(model_inputs):
    mean, cov, weights = model_inputs

    portfolio_mean = weights @ mean
    portfolio_vol = np.sqrt(weights @ cov @ weights)

    mc = monte_carlo_var_es(
        mean, cov, weights, 100_000, seed=42
    )

    # Absolute tolerance: 0.002 in decimal returns
    tolerance = 0.002

    assert mc["var_95"] == pytest.approx(
        parametric_var(portfolio_mean, portfolio_vol, 0.95),
        abs=tolerance
    )

    assert mc["var_99"] == pytest.approx(
        parametric_var(portfolio_mean, portfolio_vol, 0.99),
        abs=tolerance
    )

    assert mc["es_95"] == pytest.approx(
        parametric_es(portfolio_mean, portfolio_vol, 0.95),
        abs=tolerance
    )

    assert mc["es_99"] == pytest.approx(
        parametric_es(portfolio_mean, portfolio_vol, 0.99),
        abs=tolerance
    )


def test_invalid_weights(model_inputs):
    mean, cov, weights = model_inputs
    bad_weights = np.array([0.6, 0.6])

    with pytest.raises(ValueError):
        monte_carlo_var_es(
            mean, cov, bad_weights, 1000
        )


def test_invalid_covariance_dimension(model_inputs):
    mean, cov, weights = model_inputs

    with pytest.raises(ValueError):
        monte_carlo_var_es(
            mean, np.eye(3), weights, 1000
        )


def test_non_psd_covariance(model_inputs):
    mean, cov, weights = model_inputs

    bad_cov = np.array([
        [0.0001, 0.0003],
        [0.0003, 0.0001]
    ])

    with pytest.raises(ValueError):
        monte_carlo_var_es(
            mean, bad_cov, weights, 1000
        )


def test_invalid_simulations(model_inputs):
    mean, cov, weights = model_inputs

    for bad_n in [0, 10, 100.5, True]:
        with pytest.raises(ValueError):
            monte_carlo_var_es(
                mean, cov, weights, bad_n
            )
