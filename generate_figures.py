
# generate_figures.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Load rolling forecasts
historical = pd.read_csv(
    RESULTS_DIR / "historical_var_es_forecasts.csv",
    parse_dates=["Date"]
)

parametric = pd.read_csv(
    RESULTS_DIR / "parametric_var_es_forecasts.csv",
    parse_dates=["Date"]
)

monte_carlo = pd.read_csv(
    RESULTS_DIR / "monte_carlo_var_es_forecasts.csv",
    parse_dates=["Date"]
)


# Validate date alignment
assert historical["Date"].equals(parametric["Date"])
assert historical["Date"].equals(monte_carlo["Date"])

# Calculate realized loss
actual_loss = -historical["actual_return"]

# Detect Historical 99% VaR exceptions
exceptions = actual_loss > historical["var_99"]


# Create figure
fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(
    historical["Date"],
    actual_loss * 100,
    label="Realized Daily Loss",
    linewidth=0.7,
    alpha=0.45
)

ax.plot(
    historical["Date"],
    historical["var_99"] * 100,
    label="Historical 99% VaR",
    linewidth=1.4
)

ax.plot(
    parametric["Date"],
    parametric["var_99"] * 100,
    label="Parametric 99% VaR",
    linewidth=1.2
)

ax.plot(
    monte_carlo["Date"],
    monte_carlo["var_99"] * 100,
    label="Monte Carlo 99% VaR",
    linewidth=1.0,
    alpha=0.75
)

# Highlight Historical VaR exceptions
ax.scatter(
    historical.loc[exceptions, "Date"],
    (actual_loss[exceptions] * 100),
    label="Historical VaR Exceptions",
    color="red",
    s=18,
    zorder=5
)

ax.axhline(0, linewidth=0.8, color="black")

ax.set_title(
    "Portfolio Daily Loss vs Rolling 99% VaR Forecasts",
    fontsize=14
)

ax.set_xlabel("Date")
ax.set_ylabel("Daily Loss / VaR (%)")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()

# Export high-resolution figure
output_path = (
    FIGURES_DIR / "rolling_var_99_vs_realized_loss.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved figure to:", output_path)
