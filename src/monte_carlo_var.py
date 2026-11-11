

import numpy as np

from src.historical_var import historical_var, historical_es


def monte_carlo_var_es(
    mean_vector,
    cov_matrix,
    weights,
    n_simulations=100_000,
    seed=42
):
    """
    Calculate one-day Monte Carlo VaR and ES
    under a multivariate normal distribution.
    """

    mean_vector = np.asarray(mean_vector, dtype=float)
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Validate vector dimensions
    if mean_vector.ndim != 1 or mean_vector.size == 0:
        raise ValueError("Mean vector must be a non-empty 1D array.")

    n_assets = len(mean_vector)

    if weights.shape != (n_assets,):
        raise ValueError("Weights dimension mismatch.")

    if cov_matrix.shape != (n_assets, n_assets):
        raise ValueError("Covariance matrix dimension mismatch.")

    # Validate finite values
    if not (
        np.isfinite(mean_vector).all()
        and np.isfinite(cov_matrix).all()
        and np.isfinite(weights).all()
    ):
        raise ValueError("Inputs must contain only finite values.")

    # Validate portfolio weights
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")

    # Validate covariance matrix symmetry
    if not np.allclose(cov_matrix, cov_matrix.T):
        raise ValueError("Covariance matrix must be symmetric.")

    # Validate positive semidefiniteness
    eigenvalues = np.linalg.eigvalsh(cov_matrix)

    if np.min(eigenvalues) < -1e-10:
        raise ValueError(
            "Covariance matrix must be positive semidefinite."
        )

    # Validate number of simulations
    if (
        isinstance(n_simulations, (bool, np.bool_))
        or not isinstance(n_simulations, (int, np.integer))
        or n_simulations < 100
    ):
        raise ValueError(
            "n_simulations must be an integer >= 100."
        )

    # Create reproducible random generator
    rng = np.random.default_rng(seed)

    # Simulate correlated asset returns
    simulated_returns = rng.multivariate_normal(
        mean=mean_vector,
        cov=cov_matrix,
        size=n_simulations
    )

    # Calculate portfolio returns
    portfolio_returns = simulated_returns @ weights

    return {
        "var_95": historical_var(portfolio_returns, 0.95),
        "var_99": historical_var(portfolio_returns, 0.99),
        "es_95": historical_es(portfolio_returns, 0.95),
        "es_99": historical_es(portfolio_returns, 0.99)
    }
