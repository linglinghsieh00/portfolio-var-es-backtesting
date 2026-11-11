
# run_backtesting.py
# Automated Portfolio VaR / ES Model Validation

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting import (
    hac_score_difference_test,
    var_exceptions,
    kupiec_pof_test,
    christoffersen_independence_test,
    christoffersen_conditional_coverage_test,
    joint_var_es_score,
)

from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

MODEL_FILES = {
    "Historical": "historical_var_es_forecasts.csv",
    "Parametric": "parametric_var_es_forecasts.csv",
    "Monte Carlo": "monte_carlo_var_es_forecasts.csv",
}

REQUIRED_COLUMNS = [
    "Date",
    "actual_return",
    "var_95",
    "var_99",
    "es_95",
    "es_99",
]


def load_and_validate_models():

    models = {}

    for name, filename in MODEL_FILES.items():

        path = RESULTS_DIR / filename

        df = pd.read_csv(path, parse_dates=["Date"])

        # Check required columns
        missing = set(REQUIRED_COLUMNS) - set(df.columns)

        if missing:
            raise ValueError(
                f"{name}: Missing columns: {missing}"
            )

        if df["Date"].isna().any():
            raise ValueError(f"{name}: Invalid dates.")

        if not df["Date"].is_unique:
            raise ValueError(f"{name}: Duplicate dates.")

        if not df["Date"].is_monotonic_increasing:
            raise ValueError(f"{name}: Dates not sorted.")

        numeric_cols = REQUIRED_COLUMNS[1:]

        if not np.isfinite(df[numeric_cols].to_numpy()).all():
            raise ValueError(
                f"{name}: Invalid numeric values."
            )

        for level in [95, 99]:
            if (df[f"es_{level}"] <= 0).any():
                raise ValueError(f"{name}: Nonpositive ES.")

            if (
                df[f"es_{level}"] < df[f"var_{level}"]
            ).any():
                raise ValueError(f"{name}: ES below VaR.")

        if (df["var_99"] < df["var_95"]).any():
            raise ValueError(f"{name}: Invalid VaR ordering.")

        models[name] = df

    # Verify common forecast dates and actual returns
    reference = models["Historical"]

    for name, df in models.items():

        if len(df) != len(reference):
            raise ValueError(f"{name}: Row count mismatch.")

        if not df["Date"].equals(reference["Date"]):
            raise ValueError(f"{name}: Date mismatch.")

        if not np.allclose(
            df["actual_return"].to_numpy(),
            reference["actual_return"].to_numpy(),
            atol=1e-10,
            rtol=0,
        ):
            raise ValueError(
                f"{name}: Actual return mismatch."
            )

    print("All model forecasts validated successfully!")

    return models


def run_validation(models):

    backtesting_results = []
    fz_results = []

    for model_name, df in models.items():

        actual_returns = df["actual_return"].to_numpy()
        actual_loss = -actual_returns

        for confidence in [0.95, 0.99]:

            level = int(round(confidence * 100))

            var = df[f"var_{level}"].to_numpy()
            es = df[f"es_{level}"].to_numpy()

            # VaR exceptions
            exceptions = var_exceptions(
                actual_returns, var
            )

            # Statistical coverage tests
            kupiec = kupiec_pof_test(
                exceptions, confidence
            )

            independence = (
                christoffersen_independence_test(
                    exceptions
                )
            )

            conditional = (
                christoffersen_conditional_coverage_test(
                    exceptions, confidence
                )
            )

            backtesting_results.append({
                "Model": model_name,
                "Confidence": f"{level}%",
                "Observations": len(df),
                "Exceptions": int(exceptions.sum()),
                "Exception_Rate": exceptions.mean(),
                "LR_POF": kupiec["LR_POF"],
                "p_POF": kupiec["p_value"],
                "LR_IND": independence["LR_IND"],
                "p_IND": independence["p_value"],
                "LR_CC": conditional["LR_CC"],
                "p_CC": conditional["p_value"],
            })

            # Joint VaR-ES scoring
            scores = joint_var_es_score(
                actual_loss,
                var,
                es,
                confidence
            )

            fz_results.append({
                "Model": model_name,
                "Confidence": f"{level}%",
                "Observations": len(scores),
                "Mean_FZ0": scores.mean(),
                "Std_FZ0": scores.std(ddof=1),
            })

    return (
        pd.DataFrame(backtesting_results),
        pd.DataFrame(fz_results),
    )



def main():

    # 1. Create results folder if needed
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load and validate three model forecasts
    models = load_and_validate_models()

    # 3. Run VaR backtesting and FZ0 scoring
    backtesting_df, fz_df = run_validation(models)

    # 4. Run HAC comparison and Holm correction (NEW)
    hac_df = run_hac_comparison(
        models,
        maxlags=10
    )

    # 5. Define output file paths
    backtesting_path = (
        RESULTS_DIR / "var_backtesting_summary.csv"
    )

    fz_path = (
        RESULTS_DIR / "fz0_score_summary.csv"
    )

    hac_path = (
        RESULTS_DIR / "hac_model_comparison.csv"
    )

    # 6. Save all three results
    backtesting_df.to_csv(backtesting_path, index=False)
    fz_df.to_csv(fz_path, index=False)
    hac_df.to_csv(hac_path, index=False)

    # 7. Print results
    print("\n=== VaR Backtesting Summary ===")
    print(backtesting_df.to_string(index=False))

    print("\n=== FZ0 Scoring Summary ===")
    print(fz_df.to_string(index=False))

    print("\n=== HAC Model Comparison ===")
    print(hac_df.to_string(index=False))

    print("\nAll results saved successfully!")


def run_hac_comparison(models, maxlags=10):
    """
    Compare daily FZ0 scores across models using HAC.

    Positive difference = Model A has lower mean score.
    Apply Holm correction across all six comparisons.
    """

    comparisons = [
        ("Historical", "Parametric"),
        ("Historical", "Monte Carlo"),
        ("Parametric", "Monte Carlo"),
    ]

    results = []

    for confidence in [0.95, 0.99]:

        level = int(round(confidence * 100))

        # Calculate daily FZ0 scores for each model
        scores_by_model = {}

        for name, df in models.items():

            actual_loss = -df["actual_return"].to_numpy()

            scores_by_model[name] = joint_var_es_score(
                actual_loss=actual_loss,
                var_forecast=df[f"var_{level}"].to_numpy(),
                es_forecast=df[f"es_{level}"].to_numpy(),
                confidence_level=confidence,
            )

        # Compare aligned daily scores
        for model_a, model_b in comparisons:

            test = hac_score_difference_test(
                score_a=scores_by_model[model_a],
                score_b=scores_by_model[model_b],
                maxlags=maxlags,
            )

            results.append({
                "Confidence": f"{level}%",
                "Model_A": model_a,
                "Model_B": model_b,
                "HAC_Lags": maxlags,
                "Observations": test["n_observations"],
                "Mean_Difference": test["mean_difference"],
                "HAC_SE": test["hac_se"],
                "t_stat": test["t_stat"],
                "p_value": test["p_value"],
            })

    hac_df = pd.DataFrame(results)

    # Correct for six simultaneous comparisons
    reject, adjusted_p, _, _ = multipletests(
        hac_df["p_value"].to_numpy(),
        alpha=0.05,
        method="holm",
    )

    hac_df["Holm_Adjusted_p"] = adjusted_p
    hac_df["Reject_After_Holm"] = reject

    return hac_df

if __name__ == "__main__":
    main()
