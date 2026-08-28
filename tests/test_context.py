"""The unscheduled door: a point-in-time context reconstructed from labels.

A scanner and a screener have no `ReplaySchedule`, so this door answers both
causal questions from the bar labels themselves. Visibility is `closed_before`;
the emission gate is `strategies/util.label_freshness`. The portfolio replay
answers the same two questions from its schedule instead, and
`tests/test_portfolio_contexts.py` is where that door is pinned.

"""

import numpy as np
import pandas as pd

from nakagai.data.cache import BarCache
from nakagai.data.schema import DEFAULT_TIMEFRAMES, TimeframeSet
from nakagai.engine.context import build_context
from nakagai.strategies.rules import RuleStrategy


def _fill(cache, make_bars):
    cache.upsert("SPY", "15m", make_bars(20, "15m", start="2026-06-01 13:30"))
    cache.upsert("SPY", "1h", make_bars(8, "1h", start="2026-06-01 13:00"))
    # 2026-06-01 is EDT, so Eastern midnight is 04:00Z and the four-hour
    # buckets anchored on it land at 04:00, 08:00, 12:00 and 16:00Z. Real
    # labels, because an EMPTY 4h frame answers "not fresh" from the
    # empty-frame guard and would say nothing about the rule under test.
    cache.upsert("SPY", "4h", make_bars(4, "4h", start="2026-06-01 04:00"))
    cache.upsert("SPY", "1d", make_bars(5, "1d", start="2026-05-26 00:00"))


def test_no_future_bars(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    _fill(cache, make_bars)
    now = pd.Timestamp("2026-06-01 15:00", tz="UTC")  # 15m bar 14:45 just closed
    ctx = build_context(cache, "SPY", now, reference_pairs=())
    assert ctx.bars["15m"].index.max() == pd.Timestamp("2026-06-01 14:45", tz="UTC")
    assert ctx.bars["1h"].index.max() == pd.Timestamp("2026-06-01 14:00", tz="UTC")  # 14:00 bar closed at 15:00
    # daily: NY date of now is 2026-06-01 -> only bars strictly before that date
    assert ctx.bars["1d"].index.max() == pd.Timestamp("2026-05-30 04:00", tz="UTC")


def test_partial_hour_excluded(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    _fill(cache, make_bars)
    now = pd.Timestamp("2026-06-01 14:45", tz="UTC")
    ctx = build_context(cache, "SPY", now, reference_pairs=())
    # the 14:00 1h bar closes at 15:00; it must NOT be visible at 14:45
    assert ctx.bars["1h"].index.max() == pd.Timestamp("2026-06-01 13:00", tz="UTC")


def test_same_day_daily_bar_excluded(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    _fill(cache, make_bars)
    # Daily bars are labeled at New York midnight in UTC (see closed_before's
    # "1d" branch). Add a daily bar stamped 2026-06-01 04:00 UTC: it is today's
    # bar for the 2026-06-01 New York session.
    cache.upsert("SPY", "1d", make_bars(1, "1d", start="2026-06-01 00:00"))
    now = pd.Timestamp("2026-06-01 15:00", tz="UTC")  # NY date 2026-06-01
    ctx = build_context(cache, "SPY", now, reference_pairs=())
    # today's bar is look-ahead: it must NOT be visible (rule is strict <)
    assert pd.Timestamp("2026-06-01 04:00", tz="UTC") not in ctx.bars["1d"].index
    assert ctx.bars["1d"].index.max() == pd.Timestamp("2026-05-30 04:00", tz="UTC")


# ------------------------------------------------------------ the emission gate


def test_a_context_declares_which_higher_timeframes_may_be_decided_on(
        tmp_path, make_bars):
    """Readable and decidable are two different questions.

    The 14:00 hourly bar is readable at every close from 15:00 through 15:45,
    and it entitles an hourly play to decide at 15:00 alone. The daily gate is
    the driving bar that OPENS the session, which is 13:45 here rather than any
    close an hourly bar lands on, and the four-hour bucket labeled 12:00Z ends
    at 16:00Z, so no two of the three answer together. Nothing is declared for
    the driving frame itself: a play decided on the frame the engine replays is
    fresh on every step of it.

    Every False here is a frame with bars in it that are not newly complete,
    never an empty frame, so each one is the delta rule answering rather than
    an early return.
    """
    cache = BarCache(tmp_path)
    _fill(cache, make_bars)
    contexts = {
        stamp: build_context(
            cache, "SPY", pd.Timestamp(stamp, tz="UTC"), reference_pairs=())
        for stamp in ("2026-06-01 13:45", "2026-06-01 15:00",
                      "2026-06-01 15:15", "2026-06-01 16:00")
    }
    for stamp, ctx in contexts.items():
        assert len(ctx.bars["4h"]), f"{stamp} has no four-hour bar to judge"
    gates = {stamp: ctx.fresh for stamp, ctx in contexts.items()}
    assert gates["2026-06-01 13:45"] == {"1h": False, "4h": False, "1d": True}
    assert gates["2026-06-01 15:00"] == {"1h": True, "4h": False, "1d": False}
    assert gates["2026-06-01 15:15"] == {"1h": False, "4h": False, "1d": False}
    assert gates["2026-06-01 16:00"] == {"1h": True, "4h": True, "1d": False}


def test_an_hourly_play_decides_only_where_the_context_says_it_may(
        tmp_path, make_bars):
    """The gate reaches the play, which is the only reason it is computed.

    A rule play off the driving frame asks its context and emits nowhere else,
    so an hourly play walks four driving closes an hour and decides on one.
    """
    cache = BarCache(tmp_path)
    _fill(cache, make_bars)
    strategy = RuleStrategy({"spec": {
        "version": 2, "name": "hourly", "timeframe": "1h",
        "long": {"all": [{"lhs": {"src": "close"}, "op": ">", "rhs": 0}]},
        "risk": {"stop": {"kind": "percent", "pct": 2.0},
                 "target": {"kind": "rr", "rr": 2.0}},
    }})
    closes = pd.date_range("2026-06-01 14:45", periods=5, freq="15min", tz="UTC")
    decided = [now for now in closes
               if strategy._fresh(build_context(
                   cache, "SPY", now, reference_pairs=()))]
    assert decided == [pd.Timestamp("2026-06-01 15:00", tz="UTC")]


def test_point_in_time_context_loads_exact_pairs_once_and_keeps_driving_bars_only(
        tmp_path, monkeypatch):
    cache = BarCache(tmp_path)
    driving_index = pd.date_range(
        "2026-06-01 13:30", periods=4, freq="15min", tz="UTC")
    reference_index = pd.DatetimeIndex(
        [driving_index[0], driving_index[2]], tz="UTC")

    def frame(index, base):
        close = pd.Series(np.arange(len(index), dtype=float) + base, index=index)
        return pd.DataFrame({
            "open": close, "high": close + 1.0, "low": close - 1.0,
            "close": close, "volume": 1_000.0,
        }, index=index)

    cache.upsert("SPY", "15m", frame(driving_index, 100.0))
    cache.upsert("QQQ", "15m", frame(reference_index, 200.0))
    calls = []
    real_load = cache.load

    def recording_load(symbol, timeframe):
        calls.append((symbol, timeframe))
        return real_load(symbol, timeframe)

    monkeypatch.setattr(cache, "load", recording_load)
    tfs = TimeframeSet(
        driving="15m", higher=(), deltas=DEFAULT_TIMEFRAMES.deltas,
        session_aligned=DEFAULT_TIMEFRAMES.session_aligned,
    )

    context = build_context(
        cache, "SPY", driving_index[-1] + pd.Timedelta(minutes=15), tfs,
        reference_pairs=(("QQQ", "15m"),),
    )

    assert calls == [("SPY", "15m"), ("QQQ", "15m")]
    assert set(context.bars) == {"15m"}
    assert context.driving_bars is context.bars["15m"]
    reference = context.fe.on("QQQ", "15m")
    assert list(reference.index) == list(context.bars["15m"].index)
    assert reference.iloc[1].isna().all()
    assert reference.iloc[2]["close"] == 201.0


def test_point_in_time_reference_visibility_uses_its_own_close_time(tmp_path):
    cache = BarCache(tmp_path)
    driving_index = pd.DatetimeIndex([
        pd.Timestamp("2026-06-01 14:45", tz="UTC"),
        pd.Timestamp("2026-06-01 15:00", tz="UTC"),
    ])
    reference_index = pd.DatetimeIndex([
        pd.Timestamp("2026-06-01 14:00", tz="UTC"),
    ])

    def frame(index, close):
        values = pd.Series([close] * len(index), index=index, dtype=float)
        return pd.DataFrame({
            "open": values, "high": values + 1.0, "low": values - 1.0,
            "close": values, "volume": 1_000.0,
        }, index=index)

    cache.upsert("SPY", "15m", frame(driving_index, 100.0))
    cache.upsert("SPY", "1h", frame(reference_index, 100.0))
    cache.upsert("QQQ", "1h", frame(reference_index, 200.0))
    tfs = TimeframeSet(
        driving="15m", higher=("1h",), deltas=DEFAULT_TIMEFRAMES.deltas,
        session_aligned=DEFAULT_TIMEFRAMES.session_aligned,
    )

    before = build_context(
        cache, "SPY", pd.Timestamp("2026-06-01 14:59", tz="UTC"), tfs,
        reference_pairs=(("QQQ", "1h"),),
    )
    after = build_context(
        cache, "SPY", pd.Timestamp("2026-06-01 15:00", tz="UTC"), tfs,
        reference_pairs=(("QQQ", "1h"),),
    )

    assert before.fe.on("QQQ", "1h").empty
    assert after.fe.on("QQQ", "1h")["close"].tolist() == [200.0]
