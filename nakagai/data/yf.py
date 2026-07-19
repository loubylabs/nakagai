"""Deep daily history via yfinance (HTF bias context). Daily only."""

import pandas as pd

from nakagai.data.base import DataProvider
from nakagai.data.schema import validate_bars


def _normalize_yf(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    if df.index.tz is None:
        df.index = pd.DatetimeIndex(df.index).tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return validate_bars(df)


class YFinanceProvider(DataProvider):
    def fetch_bars(self, symbol, timeframe, start, end):
        if timeframe != "1d":
            raise ValueError("YFinanceProvider supports only timeframe '1d'")
        import yfinance as yf

        raw = yf.download(symbol, start=start.date(), end=end.date(), auto_adjust=True, progress=False)
        return _normalize_yf(raw)
