"""Frozen Phase 1 portfolio specification."""
TICKERS = ("SPY", "QQQ", "IWM", "EFA", "EEM")
WEIGHTS = {"SPY": 0.40, "QQQ": 0.25, "IWM": 0.15, "EFA": 0.10, "EEM": 0.10}
START_DATE = "2010-01-01"  # inclusive
END_DATE = "2026-10-31"  # exclusive: includes 2026-10-30
INITIAL_VALUE = 1_000_000.0
ROLLING_WINDOW = 500
LARGE_MOVE_THRESHOLD = 0.20  # flag for review, don't silently remove
