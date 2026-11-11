
# tests/test_pipeline_integration.py

import numpy as np
import pytest

from run_backtesting import (
    load_and_validate_models,
    run_validation,
    run_hac_comparison,
)


@pytest.fixture(scope="module")
def pipeline_results():
    """Run the validation pipeline once for all tests."""

    models = load_and_validate_models()

    backtesting_df, fz_df = run_validation(models)

    hac_df = run_hac_comparison(
        models,
        maxlags=10
    )

    return backtesting_df, fz_df, hac_df


def test_pipeline_output_dimensions(pipeline_results):

    backtesting_df, fz_df, hac_df = pipeline_results

    assert len(backtesting_df) == 6
    assert len(fz_df) == 6
    assert len(hac_df) == 6

    assert (backtesting_df["Observations"] == 3732).all()
    assert (fz_df["Observations"] == 3732).all()
    assert (hac_df["Observations"] == 3732).all()


def test_pipeline_reproduces_exception_counts(pipeline_results):

    backtesting_df, _, _ = pipeline_results

    expected = {
        ("Historical", "95%"): 176,
        ("Historical", "99%"): 50,
        ("Parametric", "95%"): 181,
        ("Parametric", "99%"): 83,
        ("Monte Carlo", "95%"): 181,
        ("Monte Carlo", "99%"): 85,
    }

    for (model, confidence), count in expected.items():

        row = backtesting_df.loc[
            (backtesting_df["Model"] == model)
            & (backtesting_df["Confidence"] == confidence)
        ]

        assert len(row) == 1
        assert int(row["Exceptions"].iloc[0]) == count

    assert np.isfinite(
        backtesting_df[
            ["LR_POF", "LR_IND", "LR_CC",
             "p_POF", "p_IND", "p_CC"]
        ].to_numpy()
    ).all()


def test_pipeline_reproduces_fz0_and_holm(pipeline_results):

    _, fz_df, hac_df = pipeline_results

    expected_fz0 = {
        ("Historical", "95%"): -3.620178,
        ("Historical", "99%"): -3.045914,
        ("Parametric", "95%"): -3.572465,
        ("Parametric", "99%"): -2.756745,
        ("Monte Carlo", "95%"): -3.572331,
        ("Monte Carlo", "99%"): -2.755703,
    }

    for (model, confidence), expected in expected_fz0.items():

        row = fz_df.loc[
            (fz_df["Model"] == model)
            & (fz_df["Confidence"] == confidence)
        ]

        assert len(row) == 1

        assert float(row["Mean_FZ0"].iloc[0]) == pytest.approx(
            expected,
            abs=1e-5
        )

    assert np.isfinite(
        hac_df["Holm_Adjusted_p"].to_numpy()
    ).all()

    assert not hac_df["Reject_After_Holm"].any()
