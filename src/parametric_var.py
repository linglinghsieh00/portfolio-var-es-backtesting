
import numpy as np
from scipy.stats import norm


def _validate_parameters(mean, volatility, confidence_level):

    mean = float(mean)
    volatility = float(volatility)

    if not np.isfinite(mean):
        raise ValueError("Mean must be finite.")

    if not np.isfinite(volatility) or volatility < 0:
        raise ValueError(
            "Volatility must be finite and non-negative."
        )

    if (
        not np.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise ValueError(
            "Confidence level must be between 0 and 1."
        )

    return mean, volatility


def parametric_var(mean, volatility, confidence_level=0.95):
    """Calculate one-day Normal Parametric VaR."""

    mean, volatility = _validate_parameters(
        mean, volatility, confidence_level
    )

    z = norm.ppf(confidence_level)

    return float(-mean + z * volatility)


def parametric_es(mean, volatility, confidence_level=0.95):
    """Calculate one-day Normal Parametric ES."""

    mean, volatility = _validate_parameters(
        mean, volatility, confidence_level
    )

    z = norm.ppf(confidence_level)

    es = (
        -mean
        + volatility * norm.pdf(z) / (1 - confidence_level)
    )

    return float(es)
