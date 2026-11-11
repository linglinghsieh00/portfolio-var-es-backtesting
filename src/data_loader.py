"""Acquire adjusted ETF closes and apply transparent quality checks."""
from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import TICKERS, START_DATE, END_DATE, LARGE_MOVE_THRESHOLD


def extract_adjusted_closes(downloaded: pd.DataFrame, tickers=TICKERS) -> pd.DataFrame:
    """Extract 'Close' from yfinance auto_adjust=True field-major/ticker-major data."""
    if downloaded is None or downloaded.empty:
        raise ValueError("Market data download returned no rows")
    if not isinstance(downloaded.columns, pd.MultiIndex):
        if len(tickers) != 1 or "Close" not in downloaded:
            raise ValueError("Expected MultiIndex columns for multiple tickers")
        closes = downloaded[["Close"]].rename(columns={"Close": tickers[0]})
    elif "Close" in downloaded.columns.get_level_values(0):
        closes = downloaded["Close"]
    elif "Close" in downloaded.columns.get_level_values(1):
        closes = downloaded.xs("Close", axis=1, level=1)
    else:
        raise ValueError("Missing Close field in downloaded data")
    missing = set(tickers) - set(closes.columns)
    if missing:
        raise ValueError(f"Missing ETF columns: {sorted(missing)}")
    closes = closes.loc[:, list(tickers)].copy()
    closes.index = pd.to_datetime(closes.index).tz_localize(None) if closes.index.tz is not None else pd.to_datetime(closes.index)
    closes.index.name = "Date"
    return closes


def validate_and_clean_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop invalid rows without filling prices; preserve a quality audit."""
    if prices.empty:
        raise ValueError("No prices supplied")
    if prices.index.has_duplicates:
        raise ValueError("Duplicate trading dates: inspect raw data before proceeding")
    if not prices.index.is_monotonic_increasing:
        prices = prices.sort_index()
    numeric = prices.apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(numeric) | (numeric <= 0)
    invalid_dates = bad.any(axis=1)
    cleaned = numeric.loc[~invalid_dates].copy()
    if len(cleaned) < 2:
        raise ValueError("Fewer than two complete price rows")
    daily = cleaned.pct_change(fill_method=None).iloc[1:]
    large_moves = (daily.abs() > LARGE_MOVE_THRESHOLD)
    flagged = {ticker: [d.strftime("%Y-%m-%d") for d in daily.index[large_moves[ticker]]] for ticker in daily.columns}
    audit = {
        "raw_rows": int(len(prices)),
        "clean_rows": int(len(cleaned)),
        "removed_rows": int(invalid_dates.sum()),
        "invalid_value_count_by_ticker": {k: int(v) for k,v in bad.sum().items()},
        "removed_dates_preview": [d.strftime("%Y-%m-%d") for d in prices.index[invalid_dates][:15]],
        "moves_over_20pct_by_ticker": flagged,
        "first_clean_date": cleaned.index.min().strftime("%Y-%m-%d"),
        "last_clean_date": cleaned.index.max().strftime("%Y-%m-%d"),
        "note": "Incomplete dates removed (not forward-filled); verify source errors and calendar gaps. Large moves flagged, not removed."
    }
    return cleaned, audit


def download_prices(tickers=TICKERS, start=START_DATE, end=END_DATE):
    import yfinance as yf
    df = yf.download(list(tickers), start=start, end=end, interval="1d", auto_adjust=True,
                     actions=False, group_by="column", threads=True, progress=False)
    return extract_adjusted_closes(df, tickers)


def save_data(prices: pd.DataFrame, clean_prices: pd.DataFrame, audit: dict, root: Path) -> None:
    raw_dir, processed_dir = root / "data" / "raw", root / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(raw_dir / "adjusted_close_unfiltered.csv", float_format="%.12g")
    clean_prices.to_csv(processed_dir / "adjusted_close.csv", float_format="%.12g")
    metadata = {**audit, "source": "yfinance / Yahoo Finance", "adjustment": "auto_adjust=True: Close already adjusted",
                "start_inclusive": START_DATE, "end_exclusive": END_DATE,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}
    (processed_dir / "data_quality_report.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
