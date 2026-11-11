# Project Specification v1.0

## Objective
Compare 95% and 99% one-day-ahead VaR and ES forecasts across Historical Simulation, Gaussian parametric, and Gaussian Monte Carlo models on a fixed-weight ETF portfolio. Evaluate breach frequency, serial dependence, model limitations, and stress sensitivity.

## Portfolio & Data
SPY 40%, QQQ 25%, IWM 15%, EFA 10%, EEM 10%. $1m constant notional, daily rebalanced weights, simple adjusted-price returns; 2010-01-01 through 2026-10-30 inclusive. ETF market prices from yfinance, subject to validation. Primary 500-return rolling window; 250-return robustness window.

## Loss & Risk
Portfolio return r_p,t = sum_i w_i r_i,t; loss L_t = -$1m * r_p,t. VaR_alpha = quantile_alpha(L); ES_alpha = (1/(1-alpha))*integral_{alpha}^1 VaR_u du. For normal portfolio returns (mu_p,sigma_p), VaR = $1m*(z_alpha*sigma_p-mu_p) and ES = $1m*(-mu_p+sigma_p*phi(z_alpha)/(1-alpha)). Gaussian MC should converge to Gaussian analytical results with the same moments.

## Forecast Alignment
At the close of t, estimate using returns through t only and forecast loss for t+1. Record exceedance I_{t+1}=1[L_{t+1}>VaR_{alpha,t+1|t}]. Use all possible non-leaking forecasts; report a separate later-period evaluation as a robustness check.

## Backtesting
Kupiec proportion-of-failures (coverage), Christoffersen independence (first-order Markov transitions), and conditional coverage (LR_CC=LR_POF+LR_IND); 5% hypothesis-test significance. Handle zero or near-zero breach counts and discuss test power. These are VaR tests; do not claim they formally validate ES.

## Stress & Sensitivity
Historical 2018 Q4, COVID selloff, 2022 equity bear market. Hypothetical simultaneous ETF returns; compute shock loss with unchanged weights, plus contributions. Explore 250 vs 500 observations and optional Student-t Monte Carlo.

## Acceptance & Deliverables
Three risk-model forecast tables (95/99 VaR/ES); breach timeline; test-statistic comparison; sensitivity/stress output; data quality documentation; unit tests; model validation report. Implement incrementally; Phase 1 only has been built in initial repository.
