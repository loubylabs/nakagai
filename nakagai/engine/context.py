"""Point-in-time MarketContext assembly. The ONLY door strategies get to data."""

import pandas as pd

from nakagai.data.cache import BarCache
from nakagai.data.schema import DEFAULT_TIMEFRAMES, TimeframeSet
from nakagai.strategies.base import MarketContext

NY = "America/New_York"


class PreloadedBars:
    """In-memory, BarCache-shaped view of one symbol's timeframes.

    Engine.run builds one of these so replay does one parquet read per
    timeframe total instead of one per bar. Point-in-time filtering still
    happens per bar in closed_before; this only removes repeated disk I/O.
    """

    def __init__(self, cache, symbol: str, tfs: TimeframeSet = DEFAULT_TIMEFRAMES):
        self._frames = {tf: cache.load(symbol, tf) for tf in tfs.all}

    def load(self, symbol: str, timeframe: str):
        return self._frames[timeframe]


def closed_before(df: pd.DataFrame, timeframe: str, now: pd.Timestamp,
                  tfs: TimeframeSet = DEFAULT_TIMEFRAMES) -> pd.DataFrame:
    """Point-in-time prefix of a (sorted) bar frame: only bars fully closed at
    `now`. Binary search, not a boolean mask: this runs once per replayed bar,
    and a full-history mask here made replay O(history) per bar."""
    if not len(df.index):
        return df
    if timeframe in tfs.session_aligned:
        # Session bars carry a label whose UTC CALENDAR DATE is the session
        # date. That is what this depends on, and both producers satisfy it:
        # the cache's daily resample buckets on "1D" in UTC (midnight exactly),
        # and Alpaca's 1Day bars are stamped at midnight Eastern, which is
        # 04:00 UTC under EDT and 05:00 under EST, still inside the same UTC
        # date because Eastern never runs ahead of UTC. Under that convention the
        # bar's own UTC calendar date IS the session date, so a bar is visible
        # only strictly before its session date arrives in NY: ts.date() < NY
        # date, which for these labels is exactly ts < that date's UTC
        # midnight. Comparing NY-converted timestamps instead would shift a
        # midnight-UTC bar back a day and leak a bar into its own session.
        cutoff = pd.Timestamp(now.tz_convert(NY).date(), tz="UTC")
        return df.iloc[:df.index.searchsorted(cutoff, side="left")]
    delta = tfs.deltas[timeframe]
    return df.iloc[:df.index.searchsorted(now - delta, side="right")]


def build_context(cache: BarCache, symbol: str, now: pd.Timestamp,
                  tfs: TimeframeSet = DEFAULT_TIMEFRAMES) -> MarketContext:
    return MarketContext(
        symbol=symbol, now=now, tfs=tfs,
        bars={tf: closed_before(cache.load(symbol, tf), tf, now, tfs)
              for tf in tfs.all})
