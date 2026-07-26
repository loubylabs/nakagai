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


def fetch_incremental_multi(cache: BarCache, provider, symbols: list[str],
                            timeframe: str, start: pd.Timestamp, end: pd.Timestamp,
                            full: bool = False) -> dict[str, int]:
    """Batch counterpart to fetch_incremental: batched provider round trips for
    many symbols, then a locked upsert per symbol. Returns symbol -> rows
    upserted.

    One request carries one start while symbols resume at different points, so
    the symbols are PARTITIONED rather than reduced to a single min():

    - warm (a cached span covering `start`) share one call beginning at the
      earliest of their last bars, and upsert dedupes the small overlap;
    - cold (no cache, or history starting after `start`) share one call over the
      full range, the same backfill fetch_incremental would do.

    Partitioning rather than min()-ing everything is not a micro-optimization,
    it is the difference between batching working and not. A single cold symbol
    contributes `start` itself, so one min() across the whole group drags every
    warm symbol back to the full window on EVERY cycle. Measured on the 101-name
    house universe: one symbol with no IEX bars (MMC) turned a near-empty
    incremental into a 40-day refetch for all 101, and the warm cycle cost the
    same as a cold backfill.

    Symbols are not bucketed more finely than warm/cold on purpose. The overlap
    within the warm group is bounded by how stale its laggard is, which is one
    cycle in steady state, and more groups means more requests.

    `provider` must expose fetch_bars_multi (AlpacaProvider does); this is
    deliberately not typed as DataProvider, whose contract is the single-symbol
    fetch_bars alone.
    """
    wanted = list(dict.fromkeys(s.upper() for s in symbols if s))
    if not wanted:
        return {}
    if full:
        groups = [(start, wanted)]
    else:
        warm: dict[str, pd.Timestamp] = {}
        cold: list[str] = []
        for sym in wanted:
            span = cache.coverage(sym, timeframe)
            if span is not None and start >= span[0]:
                warm[sym] = span[1]
            else:
                cold.append(sym)
        groups = []
        if warm:
            groups.append((min(warm.values()), list(warm)))
        if cold:
            groups.append((start, cold))
    written: dict[str, int] = dict.fromkeys(wanted, 0)
    for group_start, group in groups:
        frames = provider.fetch_bars_multi(group, timeframe, group_start, end)
        for sym in group:
            df = frames.get(sym)
            if df is not None and not df.empty:
                written[sym] = cache.upsert(sym, timeframe, df)
    return written
