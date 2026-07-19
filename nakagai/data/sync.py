"""Incremental bar sync: fetch only what the cache doesn't already hold.

Free-tier providers are the constraint (Alpaca allows 200 requests/min per
account, shared across every machine using the key). A full three-year 15m
backfill is dozens of paginated requests per symbol; resuming from the last
cached bar makes a routine sync cost about one request per symbol/timeframe.

Caveat: bars are split-adjusted at fetch time, so a split after the backfill
leaves the older cached bars on the pre-split basis. `nakagai sync --full`
(or deleting the parquet) forces a clean re-fetch.
"""

import pandas as pd

from nakagai.data.base import DataProvider
from nakagai.data.cache import BarCache


def fetch_incremental(cache: BarCache, provider: DataProvider, symbol: str,
                      timeframe: str, start: pd.Timestamp, end: pd.Timestamp,
                      full: bool = False) -> int:
    """Fetch bars into the cache, resuming from the last cached bar.

    The last cached bar is re-fetched (upsert keeps the newer copy) because it
    may have been written while the bar was still forming. Resuming happens
    from the last cached bar even when it predates the requested start, so a
    stale cache never ends up with an unfillable hole. A requested start
    earlier than the cache's first bar forces a full-range fetch, so widening
    the configured start still backfills the missing history."""
    fetch_start = start
    if not full:
        span = cache.coverage(symbol, timeframe)
        if span is not None and start >= span[0]:
            fetch_start = span[1]
    return cache.upsert(symbol, timeframe, provider.fetch_bars(symbol, timeframe, fetch_start, end))
