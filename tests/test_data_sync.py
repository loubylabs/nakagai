"""fetch_incremental: syncs resume from the last cached bar instead of
re-pulling the whole configured range on every run (Alpaca free tier is
200 req/min per account, shared by every machine using the key)."""

import httpx
import pandas as pd
import pytest

from nakagai.data.base import DataProvider
from nakagai.data.alpaca import (
    AlpacaBarBatchResult,
    AlpacaBarMember,
    AlpacaProvider,
)
from nakagai.data.cache import BarCache
from nakagai.data.schema import empty_bars
from nakagai.data.sync import fetch_incremental, fetch_incremental_multi


class CapturingProvider(DataProvider):
    name = "capturing"

    def __init__(self, make_bars):
        self._make = make_bars
        self.calls: list[tuple] = []

    def fetch_bars(self, symbol, timeframe, start, end):
        self.calls.append((symbol, timeframe, start, end))
        return self._make(4, start=start.strftime("%Y-%m-%d %H:%M"))


START = pd.Timestamp("2026-06-01 13:30", tz="UTC")
END = pd.Timestamp("2026-06-02", tz="UTC")


def test_empty_cache_fetches_the_full_range(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    p = CapturingProvider(make_bars)
    fetch_incremental(cache, p, "SPY", "15m", START, END)
    assert p.calls == [("SPY", "15m", START, END)]


def test_resumes_from_the_last_cached_bar(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    seeded = make_bars(10)
    cache.upsert("SPY", "15m", seeded)
    p = CapturingProvider(make_bars)
    fetch_incremental(cache, p, "SPY", "15m", START, END)
    # the last bar is re-fetched: it may have been cached while still forming
    assert p.calls[0][2] == seeded.index[-1]
    assert p.calls[0][3] == END


def test_stale_cache_resumes_from_last_bar_not_requested_start(tmp_path, make_bars):
    """A start after the cache's last bar must not win, or the cache gets a
    hole that later incremental syncs would never fill."""
    cache = BarCache(tmp_path)
    seeded = make_bars(4, start="2026-05-01 13:30")
    cache.upsert("SPY", "15m", seeded)
    p = CapturingProvider(make_bars)
    fetch_incremental(cache, p, "SPY", "15m", pd.Timestamp("2026-05-20", tz="UTC"), END)
    assert p.calls[0][2] == seeded.index[-1]


def test_start_before_cached_history_forces_a_full_fetch(tmp_path, make_bars):
    """Widening the configured start must backfill the older range."""
    cache = BarCache(tmp_path)
    cache.upsert("SPY", "15m", make_bars(4))
    p = CapturingProvider(make_bars)
    earlier = pd.Timestamp("2026-05-01", tz="UTC")
    fetch_incremental(cache, p, "SPY", "15m", earlier, END)
    assert p.calls[0][2] == earlier


def test_full_ignores_the_cache(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    cache.upsert("SPY", "15m", make_bars(10))
    p = CapturingProvider(make_bars)
    fetch_incremental(cache, p, "SPY", "15m", START, END, full=True)
    assert p.calls == [("SPY", "15m", START, END)]


# -- fetch_incremental_multi --------------------------------------------------
# One provider round trip for many symbols. One request carries ONE start, and
# symbols resume at different points, so the batch starts at the earliest resume
# point across the group and lets upsert dedupe the overlap. Refetching a few
# bars for the symbols already caught up is far cheaper than one request each,
# which is the cost this exists to remove.


def _provider_rows(frame):
    return tuple({
        "t": ts.isoformat(),
        "o": row.open,
        "h": row.high,
        "l": row.low,
        "c": row.close,
        "v": row.volume,
    } for ts, row in frame.iterrows())


class CapturingMultiProvider:
    """fetch_bars_multi-shaped. Not a DataProvider subclass: the batch method is
    an AlpacaProvider capability, not part of the one-method ABC contract."""

    def __init__(self, frames: dict, *, max_symbols_per_request: int = 100):
        self._frames = frames
        self.calls: list[tuple] = []
        self.max_symbols_per_request = max_symbols_per_request

    def fetch_bars_multi(self, symbols, timeframe, start, end):
        self.calls.append((tuple(symbols), timeframe, start, end))
        return AlpacaBarBatchResult(
            tuple(symbols),
            tuple(AlpacaBarMember(
                s, True, _provider_rows(self._frames.get(s, empty_bars())))
                for s in symbols),
        )


def test_multi_widens_to_the_earliest_resume_point(tmp_path, make_bars):
    """AAPL is caught up further than MSFT. A single request cannot carry two
    starts, so it must take the earlier one or MSFT gets an unfillable hole."""
    cache = BarCache(tmp_path)
    cache.upsert("AAPL", "15m", make_bars(4))                       # ...through 14:15
    cache.upsert("MSFT", "15m", make_bars(2))                       # ...through 13:45
    aapl_last = cache.coverage("AAPL", "15m")[1]
    msft_last = cache.coverage("MSFT", "15m")[1]
    assert msft_last < aapl_last                                    # premise of the test

    p = CapturingMultiProvider({"AAPL": make_bars(2, start="2026-06-01 14:30"),
                                "MSFT": make_bars(4, start="2026-06-01 13:45")})
    written = fetch_incremental_multi(cache, p, ["AAPL", "MSFT"], "15m", START, END)

    assert len(p.calls) == 1, "one round trip for the whole group"
    assert p.calls[0][2] == msft_last, "started at the EARLIEST resume point"
    assert written == {"AAPL": 2, "MSFT": 4}


def test_multi_partitions_cold_symbols_instead_of_widening_the_warm_batch(tmp_path, make_bars):
    """One uncached symbol must not drag the caught-up ones back to the full range.

    This was measured, not imagined. With min() taken across the whole group, a
    single symbol Alpaca has no bars for (MMC, on the 101-name house universe)
    contributed `start` itself, so EVERY cycle refetched 40 days for all 101
    symbols and the batching bought almost nothing. Two batched calls, one per
    partition, keeps the warm group's window tiny.
    """
    cache = BarCache(tmp_path)
    cache.upsert("AAPL", "15m", make_bars(4))
    cache.upsert("MSFT", "15m", make_bars(4))
    warm_resume = cache.coverage("AAPL", "15m")[1]

    p = CapturingMultiProvider({"AAPL": make_bars(1), "MSFT": make_bars(1),
                                "COLD": make_bars(1)})
    fetch_incremental_multi(cache, p, ["AAPL", "MSFT", "COLD"], "15m", START, END)

    assert len(p.calls) == 2, "one call for the warm group, one for the cold"
    by_start = {call[2]: set(call[0]) for call in p.calls}
    assert by_start[warm_resume] == {"AAPL", "MSFT"}, "warm group resumed at its own last bar"
    assert by_start[START] == {"COLD"}, "only the cold symbol pays the full range"


def test_multi_makes_one_call_when_every_symbol_is_warm(tmp_path, make_bars):
    """No cold partition means no second request, and the shared start is the
    EARLIEST warm resume point so no symbol is left with a hole."""
    cache = BarCache(tmp_path)
    cache.upsert("AAPL", "15m", make_bars(4))       # ...through 14:15
    cache.upsert("MSFT", "15m", make_bars(2))       # ...through 13:45, the laggard
    msft_last = cache.coverage("MSFT", "15m")[1]
    assert msft_last < cache.coverage("AAPL", "15m")[1]     # premise

    p = CapturingMultiProvider({"AAPL": make_bars(1), "MSFT": make_bars(1)})
    fetch_incremental_multi(cache, p, ["AAPL", "MSFT"], "15m", START, END)
    assert len(p.calls) == 1
    assert p.calls[0][2] == msft_last


def test_multi_makes_one_call_when_every_symbol_is_cold(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    p = CapturingMultiProvider({"A": make_bars(1), "B": make_bars(1)})
    written = fetch_incremental_multi(cache, p, ["A", "B"], "15m", START, END)
    assert len(p.calls) == 1
    assert p.calls[0][2] == START
    assert written == {"A": 1, "B": 1}


def test_multi_treats_a_start_before_cached_history_as_cold(tmp_path, make_bars):
    """Widening the configured start must still backfill, so a symbol whose
    history begins after the requested start belongs in the cold partition."""
    cache = BarCache(tmp_path)
    cache.upsert("AAPL", "15m", make_bars(4))                    # starts 13:30
    earlier = pd.Timestamp("2026-05-01", tz="UTC")
    p = CapturingMultiProvider({"AAPL": make_bars(1)})
    fetch_incremental_multi(cache, p, ["AAPL"], "15m", earlier, END)
    assert p.calls[0][2] == earlier


def test_multi_upserts_nothing_for_an_empty_frame(tmp_path):
    """A symbol the provider had no bars for must not raise and must not write."""
    cache = BarCache(tmp_path)
    p = CapturingMultiProvider({})          # every symbol comes back empty
    written = fetch_incremental_multi(cache, p, ["NOSUCH"], "15m", START, END)
    assert written == {"NOSUCH": 0}
    assert cache.load("NOSUCH", "15m").empty


def test_multi_raises_when_provider_omits_a_requested_member(tmp_path):
    """Omission is an operational failure, not the successful zero-row answer
    represented by a present member with an explicit empty row sequence."""
    cache = BarCache(tmp_path)

    class OmittingProvider:
        max_symbols_per_request = 100

        def fetch_bars_multi(self, symbols, timeframe, start, end):
            return AlpacaBarBatchResult(
                symbols,
                [AlpacaBarMember(symbols[0], False, [])],
            )

    with pytest.raises(ValueError, match="provider omitted requested symbol 'NOSUCH'"):
        fetch_incremental_multi(
            cache, OmittingProvider(), ["NOSUCH"], "15m", START, END)

    assert cache.load("NOSUCH", "15m").empty


def test_multi_full_ignores_cached_coverage(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    cache.upsert("AAPL", "15m", make_bars(10))
    p = CapturingMultiProvider({"AAPL": make_bars(1)})
    fetch_incremental_multi(cache, p, ["AAPL"], "15m", START, END, full=True)
    assert p.calls[0][2] == START


def test_multi_with_no_symbols_makes_no_call(tmp_path):
    cache = BarCache(tmp_path)
    p = CapturingMultiProvider({})
    assert fetch_incremental_multi(cache, p, [], "15m", START, END) == {}
    assert p.calls == []


def test_multi_uppercases_and_dedupes(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    p = CapturingMultiProvider({"AAPL": make_bars(1)})
    written = fetch_incremental_multi(cache, p, ["aapl", "AAPL"], "15m", START, END)
    assert p.calls[0][0] == ("AAPL",)
    assert written == {"AAPL": 1}


def test_multi_dedupes_the_refetched_overlap(tmp_path, make_bars):
    """The widened window refetches bars the cache already had. upsert keeps the
    newer copy of a duplicate ts, so the cache must not grow by the overlap."""
    cache = BarCache(tmp_path)
    cache.upsert("AAPL", "15m", make_bars(4))
    p = CapturingMultiProvider({"AAPL": make_bars(6)})   # 4 overlapping + 2 new
    fetch_incremental_multi(cache, p, ["AAPL"], "15m", START, END)
    assert len(cache.load("AAPL", "15m")) == 6


def test_multi_commits_a_completed_chunk_before_a_later_continuation_fails(
        tmp_path):
    """Buffering all chunks before writes would lose a completed independent
    request when the next logical request fails during pagination."""
    cache = BarCache(tmp_path)
    captured = []
    row = {
        "t": "2026-06-01T13:30:00Z", "o": 1.0, "h": 1.5,
        "l": 0.5, "c": 1.2, "v": 100,
    }

    def handler(request):
        captured.append(request)
        symbols = request.url.params["symbols"]
        token = request.url.params.get("page_token")
        if symbols == "AAPL,MSFT":
            return httpx.Response(200, json={
                "bars": {"AAPL": [row], "MSFT": [row]},
                "next_page_token": None,
            })
        if token == "qqq-continue":
            raise httpx.ReadTimeout("continuation timed out", request=request)
        return httpx.Response(200, json={
            "bars": {"QQQ": [row]},
            "next_page_token": "qqq-continue",
        })

    provider = AlpacaProvider(
        key_id="k", secret="s",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _seconds: None,
        max_symbols_per_request=2,
    )

    with pytest.raises(httpx.ReadTimeout, match="continuation timed out"):
        fetch_incremental_multi(
            cache, provider, ["AAPL", "MSFT", "QQQ"],
            "15m", START, END,
        )

    assert [request.url.params["symbols"] for request in captured] == [
        "AAPL,MSFT",
        "QQQ",
        "QQQ",
    ]
    assert len(cache.load("AAPL", "15m")) == 1
    assert len(cache.load("MSFT", "15m")) == 1
    assert cache.load("QQQ", "15m").empty
