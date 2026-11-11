
import numpy as np
from scipy.stats import chi2
from scipy.special import xlogy
import statsmodels.api as sm



def var_exceptions(actual_returns, var_forecasts):
    """
    Calculate one-day VaR exception indicators.

    Exception occurs when:
        realized_loss > forecast_VaR

    Parameters
    ----------
    actual_returns : array-like
        Realized portfolio returns.
    var_forecasts : array-like
        Forecast VaR values in return units.

    Returns
    -------
    np.ndarray
        Boolean array of VaR exceptions.
    """

    returns = np.asarray(actual_returns, dtype=float)
    var = np.asarray(var_forecasts, dtype=float)

    if returns.ndim != 1 or var.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if returns.size == 0 or returns.shape != var.shape:
        raise ValueError("Inputs must be nonempty and aligned.")

    if not np.isfinite(returns).all():
        raise ValueError("Returns must be finite.")

    if not np.isfinite(var).all():
        raise ValueError("VaR forecasts must be finite.")

    if np.any(var < 0):
        raise ValueError("VaR forecasts must be nonnegative.")

    realized_loss = -returns

    return realized_loss > var



def kupiec_pof_test(exceptions, confidence_level):
    """
    Kupiec Proportion of Failures test.

    H0: Exception probability = 1 - confidence_level
    """

    exceptions = np.asarray(exceptions)

    if exceptions.ndim != 1 or exceptions.size == 0:
        raise ValueError("Exceptions must be a nonempty 1D array.")

    if not np.isin(exceptions, [0, 1]).all():
        raise ValueError("Exceptions must contain only 0 or 1.")

    if not np.isfinite(confidence_level) or not (
        0 < confidence_level < 1
    ):
        raise ValueError("Invalid confidence level.")

    n = len(exceptions)
    x = int(exceptions.sum())

    p0 = 1 - confidence_level
    p_hat = x / n

    # Null log-likelihood
    log_l0 = (
        xlogy(x, p0)
        + xlogy(n - x, 1 - p0)
    )

    # Unrestricted log-likelihood
    log_l1 = (
        xlogy(x, p_hat)
        + xlogy(n - x, 1 - p_hat)
    )

    lr_pof = max(0.0, 2 * (log_l1 - log_l0))
    p_value = chi2.sf(lr_pof, df=1)

    return {
        "n_observations": n,
        "n_exceptions": x,
        "expected_rate": p0,
        "observed_rate": p_hat,
        "LR_POF": lr_pof,
        "p_value": p_value,
        "reject_5pct": p_value < 0.05
    }



def christoffersen_independence_test(exceptions):
    """
    Christoffersen first-order independence test.

    H0: P(I_t=1 | I_{t-1}=0)
        = P(I_t=1 | I_{t-1}=1)
    """

    exceptions = np.asarray(exceptions)

    if exceptions.ndim != 1 or exceptions.size < 3:
        raise ValueError("At least 3 exception observations required.")

    if not np.isin(exceptions, [0, 1]).all():
        raise ValueError("Exceptions must contain only 0 or 1.")

    previous = exceptions[:-1]
    current = exceptions[1:]

    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))

    n0 = n00 + n01
    n1 = n10 + n11

    if n0 == 0 or n1 == 0:
        raise ValueError(
            "Both previous states must occur for this test."
        )

    pi01 = n01 / n0
    pi11 = n11 / n1

    total = n0 + n1
    pi = (n01 + n11) / total

    # Independence log-likelihood
    log_l0 = (
        xlogy(n00 + n10, 1 - pi)
        + xlogy(n01 + n11, pi)
    )

    # First-order Markov log-likelihood
    log_l1 = (
        xlogy(n00, 1 - pi01)
        + xlogy(n01, pi01)
        + xlogy(n10, 1 - pi11)
        + xlogy(n11, pi11)
    )

    lr_ind = max(0.0, 2 * (log_l1 - log_l0))
    p_value = chi2.sf(lr_ind, df=1)

    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "pi01": pi01,
        "pi11": pi11,
        "LR_IND": lr_ind,
        "p_value": p_value,
        "reject_5pct": bool(p_value < 0.05)
    }



def christoffersen_conditional_coverage_test(
    exceptions, confidence_level
):
    """
    Christoffersen conditional coverage test.

    H0: Correct VaR coverage and first-order independence.

    Uses the same T-1 transitions for both likelihoods.
    """

    if not np.isscalar(confidence_level):
        raise ValueError("Confidence level must be a scalar.")

    if not np.isfinite(confidence_level) or not (
        0 < confidence_level < 1
    ):
        raise ValueError("Invalid confidence level.")

    # Reuse the validated independence function
    ind = christoffersen_independence_test(exceptions)

    n00 = ind["n00"]
    n01 = ind["n01"]
    n10 = ind["n10"]
    n11 = ind["n11"]

    pi01 = ind["pi01"]
    pi11 = ind["pi11"]

    # Expected exception probability
    p0 = 1 - confidence_level

    # Likelihood under H0: correct coverage + independence
    log_l0 = (
        xlogy(n00 + n10, 1 - p0)
        + xlogy(n01 + n11, p0)
    )

    # Likelihood under unrestricted Markov alternative
    log_l1 = (
        xlogy(n00, 1 - pi01)
        + xlogy(n01, pi01)
        + xlogy(n10, 1 - pi11)
        + xlogy(n11, pi11)
    )

    lr_cc = max(0.0, 2 * (log_l1 - log_l0))
    p_value = chi2.sf(lr_cc, df=2)

    return {
        "n_transitions": n00 + n01 + n10 + n11,
        "LR_CC": lr_cc,
        "p_value": p_value,
        "reject_5pct": bool(p_value < 0.05)
    }



def joint_var_es_score(
    actual_loss,
    var_forecast,
    es_forecast,
    confidence_level
):
    """
    FZ0 joint VaR-ES scoring function for positive ES forecasts.

    Lower scores indicate better joint forecast performance.

    All inputs use decimal return units, not percentages.
    """

    loss = np.asarray(actual_loss, dtype=float)
    var = np.asarray(var_forecast, dtype=float)
    es = np.asarray(es_forecast, dtype=float)

    # Validate confidence level
    if not np.isscalar(confidence_level):
        raise ValueError("Confidence level must be a scalar.")

    if not np.isfinite(confidence_level) or not (
        0 < confidence_level < 1
    ):
        raise ValueError("Invalid confidence level.")

    # Validate arrays
    if loss.ndim != 1 or var.ndim != 1 or es.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if loss.size == 0:
        raise ValueError("Inputs must not be empty.")

    if loss.shape != var.shape or loss.shape != es.shape:
        raise ValueError("Input shapes must match.")

    if not (
        np.isfinite(loss).all()
        and np.isfinite(var).all()
        and np.isfinite(es).all()
    ):
        raise ValueError("Inputs must be finite.")

    if np.any(es <= 0):
        raise ValueError("ES forecasts must be positive.")

    if np.any(es < var):
        raise ValueError("ES forecasts must be >= VaR.")

    alpha = 1 - confidence_level

    exceptions = (loss > var).astype(float)

    scores = (
        var / es
        + np.log(es)
        - 1
        + exceptions * (loss - var) / (alpha * es)
    )

    return scores



def hac_score_difference_test(
    score_a, score_b, maxlags=10
):
    """
    Test equal mean predictive scores using HAC standard errors.

    H0: E[Score_B - Score_A] = 0

    A positive mean difference means Model A has
    a lower (better) average score.
    """

    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Scores must be one-dimensional.")

    if a.size == 0 or a.shape != b.shape:
        raise ValueError("Scores must be nonempty and aligned.")

    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Scores must be finite.")

    if (
        isinstance(maxlags, (bool, np.bool_))
        or not isinstance(maxlags, (int, np.integer))
        or maxlags < 0
        or maxlags >= len(a)
    ):
        raise ValueError("Invalid HAC maxlags.")

    difference = b - a

    # Intercept-only regression estimates mean score difference
    X = np.ones((len(difference), 1))

    fitted = sm.OLS(difference, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": int(maxlags)}
    )

    return {
        "n_observations": len(difference),
        "mean_difference": float(fitted.params[0]),
        "hac_se": float(fitted.bse[0]),
        "t_stat": float(fitted.tvalues[0]),
        "p_value": float(fitted.pvalues[0]),
        "reject_5pct": bool(fitted.pvalues[0] < 0.05)
    }
