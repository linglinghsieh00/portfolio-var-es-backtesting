import numpy as np
import pandas as pd
import pytest
from src.data_loader import extract_adjusted_closes, validate_and_clean_prices
from src.portfolio import calculate_portfolio


def dates(n):
    return pd.date_range("2020-01-01", periods=n, freq="B", name="Date")


def test_portfolio_returns_and_sign():
    prices = pd.DataFrame({"AAA": [100., 110., 121.], "BBB": [100., 100., 110.]}, index=dates(3))
    asset, result = calculate_portfolio(prices, {"AAA": 0.6, "BBB": 0.4}, 1000)
    assert result.portfolio_return.iloc[0] == pytest.approx(0.06)
    assert result.portfolio_return.iloc[1] == pytest.approx(0.10)
    assert result.realized_loss_usd.iloc[0] == pytest.approx(-60.)
    assert result.wealth_index.iloc[-1] == pytest.approx(1.06 * 1.10)
    assert len(asset) == 2


def test_loss_positive_on_drop():
    prices = pd.DataFrame({"A": [100., 90.], "B": [100., 100.]}, index=dates(2))
    _, result = calculate_portfolio(prices, {"A": .5, "B": .5}, 1000)
    assert result.realized_loss_usd.iloc[0] == pytest.approx(50.)


def test_invalid_row_removed_without_forward_fill():
    prices = pd.DataFrame({"A": [100., np.nan, 105., 110.], "B": [100., 101., 102., 103.]}, index=dates(4))
    clean, audit = validate_and_clean_prices(prices)
    assert len(clean) == 3 and audit["removed_rows"] == 1
    assert audit["invalid_value_count_by_ticker"]["A"] == 1
    assert dates(4)[1] not in clean.index


def test_multiindex_field_major():
    idx = dates(2)
    columns = pd.MultiIndex.from_tuples([("Close", "A"), ("Close", "B"), ("Open", "A")])
    raw = pd.DataFrame([[100., 200., 99.], [101., 199., 100.]], index=idx, columns=columns)
    closes = extract_adjusted_closes(raw, ("A", "B"))
    assert list(closes.columns) == ["A", "B"]
    assert closes.loc[idx[0], "A"] == 100.


def test_multiindex_ticker_major():
    idx = dates(2)
    columns = pd.MultiIndex.from_tuples([("A", "Close"), ("B", "Close")])
    raw = pd.DataFrame([[100., 200.], [101., 199.]], index=idx, columns=columns)
    closes = extract_adjusted_closes(raw, ("A", "B"))
    assert list(closes.columns) == ["A", "B"]


def test_invalid_weights():
    prices = pd.DataFrame({"A": [100., 101.], "B": [100., 101.]}, index=dates(2))
    with pytest.raises(ValueError):
        calculate_portfolio(prices, {"A": .8, "B": .3})


def test_duplicated_dates_rejected():
    prices = pd.DataFrame({"A": [100., 101., 102.]}, index=pd.DatetimeIndex(["2020-01-01", "2020-01-01", "2020-01-02"]))
    with pytest.raises(ValueError, match="Duplicate"):
        validate_and_clean_prices(prices)
