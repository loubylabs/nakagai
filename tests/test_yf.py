import pandas as pd
import pytest

from nakagai.data.yf import YFinanceProvider, _normalize_yf


def _raw_yf_frame():
    """Mimic yfinance.download(auto_adjust=True) output: naive index, capitalized cols."""
    idx = pd.date_range("2026-06-01", periods=3, freq="1D", name="Date")
    return pd.DataFrame(
        {"Open": [1.0, 2, 3], "High": [1.5, 2.5, 3.5], "Low": [0.5, 1.5, 2.5],
         "Close": [1.2, 2.2, 3.2], "Volume": [100, 200, 300]},
        index=idx,
    )


def test_normalize_yf():
    out = _normalize_yf(_raw_yf_frame())
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert str(out.index.tz) == "UTC"
    assert out.index[0] == pd.Timestamp("2026-06-01", tz="UTC")


def test_normalize_yf_flattens_multiindex_columns():
    raw = _raw_yf_frame()
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["SPY"]])
    out = _normalize_yf(raw)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_rejects_non_daily():
    with pytest.raises(ValueError, match="1d"):
        YFinanceProvider().fetch_bars("SPY", "15m", pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-02-01", tz="UTC"))
