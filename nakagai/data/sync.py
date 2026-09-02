"""Incremental bar sync: fetch only what the cache doesn't already hold.

Free-tier providers are the constraint (Alpaca allows 200 requests/min per
account, shared across every machine using the key). A full three-year 15m
backfill is dozens of paginated requests per symbol; resuming from the last
cached bar makes a routine sync cost about one request per symbol/timeframe.

Caveat: bars are split-adjusted at fetch time, so a split after the backfill
leaves the older cached bars on the pre-split basis. `nakagai sync --full`
(or deleting the parquet) forces a clean re-fetch.

Not every timeframe is fetched. derive_incremental materializes the derived
ones (nakagai/data/resample.py) from bars the cache already holds, for no
provider requests at all; it belongs here because a sync cycle owes it right
after the source timeframe lands.
"""

import pandas as pd

from nakagai.data.base import DataProvider
from nakagai.data.alpaca import AlpacaBarBatchResult, frame_from_rows
from nakagai.data.cache import BarCache
from nakagai.data.resample import DERIVED, resample_bars


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


def derive_incremental(cache: BarCache, symbol: str, timeframe: str = "4h",
                       full: bool = False) -> int:
    """Materialize a derived timeframe into the cache from its source bars.

    The fetch functions above spend provider requests; this one spends nothing
    but CPU, because a derived timeframe (see nakagai/data/resample.py) is
    aggregated from bars the cache already holds. It writes through the same
    cache.upsert, to the same SYMBOL_TF.parquet the backtester reads, so a
    derived bar is indistinguishable downstream from a fetched one and there is
    exactly one implementation behind both the live path and a replay.

    Only a bounded trailing window is re-derived, and the resume point is the
    last derived bar's own LABEL. That label is a bucket boundary by
    construction, which is the property the window turns on: starting anywhere
    else would hand resample_bars a truncated first bucket and write a partial
    bar over a complete one. Re-deriving the boundary bucket itself is
    deliberate, not waste; it is the bucket most likely to have been withheld
    as still forming last cycle, and upsert keeps the newer copy, so the
    operation is idempotent. In steady state this is a handful of source rows
    per symbol per cycle instead of three years of them.

    A cold derived cache re-derives everything, and so does a source whose
    history now begins EARLIER than the derived bars do, for the same reason
    fetch_incremental treats a widened start as a full fetch: otherwise the
    backfilled history would never be aggregated and the derived frame would
    keep a hole no later cycle could fill.

    Returns the number of rows upserted.
    """
    source_tf = DERIVED.get(timeframe)
    if source_tf is None:
        raise ValueError(f"{timeframe!r} is not a derived timeframe "
                         f"(derived: {sorted(DERIVED)})")
    src = cache.load(symbol, source_tf)
    if not len(src.index):
        return 0
    if not full:
        span = cache.coverage(symbol, timeframe)
        if span is not None and src.index[0] >= span[0]:
            src = src.loc[src.index >= span[1]]
    derived = resample_bars(src, timeframe)
    if derived.empty:
        return 0
    return cache.upsert(symbol, timeframe, derived)


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
    step = max(1, int(provider.max_symbols_per_request))
    for group_start, group in groups:
        for offset in range(0, len(group), step):
            chunk = group[offset:offset + step]
            result = provider.fetch_bars_multi(
                chunk, timeframe, group_start, end)
            _commit_batch(cache, timeframe, chunk, result, written)
    return written


def _commit_batch(cache: BarCache, timeframe: str, requested: list[str],
                  result: AlpacaBarBatchResult,
                  written: dict[str, int]) -> None:
    """Validate and commit one completed logical provider request."""
    expected = tuple(requested)
    if result.requested != expected:
        raise ValueError(
            f"provider returned requested={result.requested!r}, expected {expected!r}")
    symbols = tuple(member.symbol for member in result.members)
    if symbols != expected:
        raise ValueError(
            f"provider returned members={symbols!r}, expected {expected!r}")
    omitted = tuple(
        member.symbol for member in result.members if not member.present)
    if omitted:
        if len(omitted) == 1:
            detail = f"symbol {omitted[0]!r}"
        else:
            detail = f"symbols {omitted!r}"
        raise ValueError(f"provider omitted requested {detail}")
    for member in result.members:
        frame = frame_from_rows(list(member.rows))
        if not frame.empty:
            written[member.symbol] = cache.upsert(
                member.symbol, timeframe, frame)
