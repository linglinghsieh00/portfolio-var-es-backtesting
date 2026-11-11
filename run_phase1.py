"""Run from project root: python run_phase1.py"""
from pathlib import Path
import json
from src.config import TICKERS, WEIGHTS, INITIAL_VALUE
from src.data_loader import download_prices, validate_and_clean_prices, save_data
from src.portfolio import calculate_portfolio


def main():
    root = Path(__file__).resolve().parent
    print("Downloading adjusted ETF prices:", ", ".join(TICKERS))
    raw = download_prices()
    clean, report = validate_and_clean_prices(raw)
    returns, portfolio = calculate_portfolio(clean, WEIGHTS, INITIAL_VALUE)
    save_data(raw, clean, report, root)
    out = root / "data" / "processed"
    returns.to_csv(out / "asset_returns.csv", float_format="%.12g")
    portfolio.to_csv(out / "portfolio_returns.csv", float_format="%.12g")
    returns.corr().to_csv(out / "correlation_matrix.csv", float_format="%.12g")
    if len(portfolio) <= 500:
        raise ValueError("Insufficient returns for the planned 500-day rolling forecast")
    print(f"Complete price observations: {len(clean):,}; return observations: {len(portfolio):,}")
    print(f"Date range: {clean.index.min().date()} -> {clean.index.max().date()}")
    print(f"Removed incomplete price dates: {report['removed_rows']}")
    print("Review data/processed/data_quality_report.json before proceeding to risk models.")


if __name__ == "__main__":
    main()
