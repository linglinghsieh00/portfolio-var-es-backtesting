
import numpy as np
import pytest
from src.backtesting import (
    var_exceptions,
    kupiec_pof_test,
    christoffersen_independence_test,
    christoffersen_conditional_coverage_test,
    joint_var_es_score,
    hac_score_difference_test
)



def test_var_exceptions():
    returns = np.array([-0.04, -0.02, 0.01])
    var = np.array([0.03, 0.03, 0.03])

    result = var_exceptions(returns, var)

    assert result.tolist() == [True, False, False]


def test_equal_loss_is_not_exception():
    returns = np.array([-0.03])
    var = np.array([0.03])

    result = var_exceptions(returns, var)

    assert not result[0]


def test_mismatched_shapes():
    with pytest.raises(ValueError):
        var_exceptions(
            [-0.02, -0.03],
            [0.02]
        )


def test_invalid_returns():
    with pytest.raises(ValueError):
        var_exceptions(
            [np.nan, -0.02],
            [0.03, 0.03]
        )


def test_negative_var():
    with pytest.raises(ValueError):
        var_exceptions(
            [-0.02],
            [-0.03]
        )


def test_kupiec_historical_99():
    # Actual project result: 50 exceptions / 3732 days
    exceptions = np.array([1] * 50 + [0] * 3682)

    result = kupiec_pof_test(exceptions, 0.99)

    assert result["n_exceptions"] == 50
    assert result["n_observations"] == 3732
    assert result["LR_POF"] == pytest.approx(3.9329, abs=1e-4)
    assert result["reject_5pct"]


def test_kupiec_historical_95():
    exceptions = np.array([1] * 176 + [0] * 3556)

    result = kupiec_pof_test(exceptions, 0.95)

    assert result["observed_rate"] == pytest.approx(176 / 3732)
    assert not result["reject_5pct"]


def test_kupiec_zero_exceptions():
    exceptions = np.zeros(100, dtype=int)

    result = kupiec_pof_test(exceptions, 0.99)

    assert np.isfinite(result["LR_POF"])
    assert np.isfinite(result["p_value"])


def test_kupiec_invalid_confidence():
    exceptions = np.array([0, 1, 0, 0])

    with pytest.raises(ValueError):
        kupiec_pof_test(exceptions, 1.0)


def test_kupiec_invalid_exceptions():
    exceptions = np.array([0, 1, 2, 0])

    with pytest.raises(ValueError):
        kupiec_pof_test(exceptions, 0.99)


def test_christoffersen_transition_counts():

    exceptions = np.array([0, 0, 1, 1, 0])

    result = christoffersen_independence_test(exceptions)

    assert result["n00"] == 1
    assert result["n01"] == 1
    assert result["n10"] == 1
    assert result["n11"] == 1


def test_christoffersen_conditional_probabilities():

    exceptions = np.array([0, 0, 1, 1, 0])

    result = christoffersen_independence_test(exceptions)

    assert result["pi01"] == pytest.approx(0.5)
    assert result["pi11"] == pytest.approx(0.5)


def test_christoffersen_lr_nonnegative():

    exceptions = np.array([0, 1, 1, 0, 0, 1, 0])

    result = christoffersen_independence_test(exceptions)

    assert result["LR_IND"] >= 0
    assert 0 <= result["p_value"] <= 1


def test_christoffersen_invalid_values():

    with pytest.raises(ValueError):
        christoffersen_independence_test([0, 1, 2, 0])


def test_christoffersen_no_exception_state():

    with pytest.raises(ValueError):
        christoffersen_independence_test([0, 0, 0, 0])



def test_cc_transition_count():
    exceptions = [0, 0, 1, 1, 0]

    result = christoffersen_conditional_coverage_test(
        exceptions, 0.95
    )

    assert result["n_transitions"] == 4


def test_cc_likelihood_decomposition():
    exceptions = [0, 0, 1, 1, 0, 0, 1, 0]

    cc = christoffersen_conditional_coverage_test(
        exceptions, 0.95
    )

    ind = christoffersen_independence_test(exceptions)

    # POF must use the same T-1 observations
    pof = kupiec_pof_test(exceptions[1:], 0.95)

    assert cc["LR_CC"] == pytest.approx(
        ind["LR_IND"] + pof["LR_POF"],
        abs=1e-10
    )


def test_cc_manual_likelihood():
    exceptions = [0, 0, 1, 1, 0]

    result = christoffersen_conditional_coverage_test(
        exceptions, 0.90
    )

    # All four transition counts equal one
    log_alt = 4 * np.log(0.5)
    log_null = 2 * np.log(0.9) + 2 * np.log(0.1)

    expected_lr = 2 * (log_alt - log_null)

    assert result["LR_CC"] == pytest.approx(expected_lr)


def test_cc_detects_strong_clustering():
    exceptions = [0] * 90 + [1] * 10

    result = christoffersen_conditional_coverage_test(
        exceptions, 0.99
    )

    assert result["reject_5pct"]
    assert result["p_value"] < 0.05


def test_cc_invalid_confidence():
    with pytest.raises(ValueError):
        christoffersen_conditional_coverage_test(
            [0, 0, 1, 0], 1.0
        )



def test_fz0_manual_calculation():

    loss = np.array([0.04])
    var = np.array([0.03])
    es = np.array([0.05])

    result = joint_var_es_score(
        loss, var, es, 0.99
    )

    expected = (
        0.03 / 0.05
        + np.log(0.05)
        - 1
        + (0.04 - 0.03) / (0.01 * 0.05)
    )

    assert result[0] == pytest.approx(expected)


def test_fz0_no_exception():

    loss = np.array([0.01])
    var = np.array([0.03])
    es = np.array([0.05])

    result = joint_var_es_score(
        loss, var, es, 0.99
    )

    expected = 0.03 / 0.05 + np.log(0.05) - 1

    assert result[0] == pytest.approx(expected)


def test_fz0_vector_output():

    loss = np.array([0.01, 0.04, 0.02])
    var = np.array([0.03, 0.03, 0.03])
    es = np.array([0.05, 0.05, 0.05])

    result = joint_var_es_score(
        loss, var, es, 0.99
    )

    assert result.shape == (3,)
    assert np.isfinite(result).all()


def test_fz0_invalid_es():

    with pytest.raises(ValueError):
        joint_var_es_score(
            [0.02], [0.03], [0.0], 0.99
        )


def test_fz0_es_less_than_var():

    with pytest.raises(ValueError):
        joint_var_es_score(
            [0.02], [0.04], [0.03], 0.99
        )


def test_fz0_invalid_confidence():

    with pytest.raises(ValueError):
        joint_var_es_score(
            [0.02], [0.03], [0.05], 1.0
        )


def test_hac_mean_difference():

    score_a = np.array([1.0, 2.0, 3.0, 4.0])
    score_b = np.array([1.1, 2.3, 3.2, 4.4])

    result = hac_score_difference_test(
        score_a, score_b, maxlags=1
    )

    expected = np.mean(score_b - score_a)

    assert result["mean_difference"] == pytest.approx(expected)


def test_hac_identical_scores():

    score_a = np.array([1.0, 2.0, 3.0, 4.0])
    score_b = score_a.copy()

    result = hac_score_difference_test(
        score_a, score_b, maxlags=1
    )

    assert result["mean_difference"] == pytest.approx(0.0)
    assert result["hac_se"] == pytest.approx(0.0)


def test_hac_positive_difference():

    score_a = np.array([1.0, 2.0, 3.0, 4.0])
    score_b = np.array([1.1, 2.3, 3.2, 4.4])

    result = hac_score_difference_test(
        score_a, score_b, maxlags=1
    )

    assert result["mean_difference"] > 0


def test_hac_mismatched_lengths():

    with pytest.raises(ValueError):
        hac_score_difference_test(
            [1.0, 2.0],
            [1.0, 2.0, 3.0],
            maxlags=1
        )


def test_hac_invalid_lag():

    with pytest.raises(ValueError):
        hac_score_difference_test(
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            maxlags=5
        )


def test_hac_nonfinite_scores():

    with pytest.raises(ValueError):
        hac_score_difference_test(
            [1.0, np.nan, 3.0],
            [1.1, 2.1, 3.1],
            maxlags=1
        )
