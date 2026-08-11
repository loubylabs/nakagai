"""Point-in-time MarketContext assembly. The ONLY door strategies get to data.

Two doors, and they answer to different clocks. `build_context` serves the
live scanner and the screener, which have no schedule and reconstruct
visibility from the bar labels themselves through `closed_before`.
`build_scheduled_context` serves the portfolio replay, which has an embedded
`ReplaySchedule` and therefore asks it: a bar is visible when the schedule
says it became available, never because label arithmetic put it in the past.
"""

import numpy as np
import pandas as pd

from nakagai.data.cache import BarCache
from nakagai.data.schema import DEFAULT_TIMEFRAMES, EXCHANGE_TZ, TimeframeSet
from nakagai.engine.bars import (
    BASE_TIMEFRAME,
    ReplayDependencies,
    _ValidatedPortfolioBars,
)
from nakagai.engine.portfolio_types import (
    ReplayInputError,
    _require_instance,
    _require_symbol,
    _require_timestamp,
)
from nakagai.engine.schedule import ValidatedSchedule
from nakagai.strategies.base import MarketContext
from nakagai.strategies.rules.vocabulary import Vocabulary, resolve_vocabulary


class PreloadedBars:
    """In-memory, BarCache-shaped view of one symbol's timeframes, plus the
    replay's node cache.

    Engine.run builds one of these, so replay does one parquet read per
    timeframe total instead of one per bar, and every node in a spec is
    computed once per replay instead of once per bar. Point-in-time filtering
    still happens per bar in closed_before; `fe` holds the untruncated frames.
    """

    # Keyword-only `vocabulary`, as everywhere it sits behind an optional
    # `tfs`: passed positionally it would bind to `tfs` and the replay would
    # quietly evaluate against the core vocabulary instead of the injected one.
    def __init__(self, cache, symbol: str, tfs: TimeframeSet = DEFAULT_TIMEFRAMES,
                 *, vocabulary: Vocabulary | None = None):
        from nakagai.strategies.rules.frame_eval import FrameEval
        self._frames = {tf: cache.load(symbol, tf) for tf in tfs.all}
        self.fe = FrameEval(self._frames, tfs,
                            vocabulary=resolve_vocabulary(vocabulary))

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
        # Legacy providers may label daily rows at 00:00Z. BarCache.upsert
        # canonicalizes cached daily rows to midnight America/New_York, expressed
        # as 04:00Z or 05:00Z. All of those labels carry the same UTC calendar
        # date as the session because Eastern never runs ahead of UTC. Under that
        # convention the bar's own UTC calendar date IS the session date, so a
        # bar is visible
        # only strictly before its session date arrives in NY: ts.date() < NY
        # date, which for these labels is exactly ts < that date's UTC
        # midnight. Comparing NY-converted timestamps instead would shift a
        # midnight-UTC bar back a day and leak a bar into its own session.
        cutoff = pd.Timestamp(now.tz_convert(EXCHANGE_TZ).date(), tz="UTC")
        return df.iloc[:df.index.searchsorted(cutoff, side="left")]
    delta = tfs.deltas[timeframe]
    return df.iloc[:df.index.searchsorted(now - delta, side="right")]


def visible_counts(src_index: pd.DatetimeIndex, dst_close_times: pd.DatetimeIndex,
                   timeframe: str, tfs: TimeframeSet = DEFAULT_TIMEFRAMES) -> np.ndarray:
    """How many `timeframe` bars are fully closed at each of `dst_close_times`.

    The vectorized form of calling closed_before once per replayed bar: entry i
    is exactly len(closed_before(src, timeframe, dst_close_times[i], tfs)). One
    searchsorted per (timeframe, replay) replaces one slice per bar.

    The session-aligned branch reuses closed_before's own NY rule rather than a
    label-plus-one-day approximation, so it does not depend on the cache being
    RTH-only.
    """
    if not len(src_index):
        return np.zeros(len(dst_close_times), dtype=np.int64)
    if timeframe in tfs.session_aligned:
        # A sweep replays this map thousands of times (specs x symbols x
        # windows), so a Python-level loop building one pd.Timestamp per
        # destination row here is not a stylistic nicety, it is millions of
        # constructions per pair. tz_convert + normalize + relocalize computes
        # the same "NY calendar date, as midnight UTC" cutoff for every row in
        # one vectorized pass: convert to NY wall time, floor to that day's
        # midnight (still NY-tz-aware, so it is correct across DST changes),
        # drop the tz, then relabel the naive midnight as UTC. Do not
        # re-simplify this back into a per-row comprehension.
        cutoffs = (dst_close_times.tz_convert(EXCHANGE_TZ).normalize()
                   .tz_localize(None).tz_localize("UTC"))
        return src_index.searchsorted(cutoffs, side="left").astype(np.int64)
    delta = tfs.deltas[timeframe]
    return src_index.searchsorted(dst_close_times - delta, side="right").astype(np.int64)


def build_context(cache: BarCache, symbol: str, now: pd.Timestamp,
                  tfs: TimeframeSet = DEFAULT_TIMEFRAMES, *,
                  vocabulary: Vocabulary | None = None) -> MarketContext:
    """Point-in-time context at `now`.

    closed_before still runs per timeframe per call. It is a searchsorted plus
    a zero-copy .iloc slice, measured at 1.5% of replay, and ctx.bars[tf] has to
    stay a real prefix frame because _fresh, rr_signal, stop_target and every
    non-rule strategy read it. Removing it would trade that 1.5% for a new
    invariant to defend.
    """
    from nakagai.strategies.rules.frame_eval import FrameEval
    frames = {tf: cache.load(symbol, tf) for tf in tfs.all}
    bars = {tf: closed_before(frames[tf], tf, now, tfs) for tf in tfs.all}
    # A replay hands its own FrameEval over the untruncated frames (PreloadedBars);
    # a scanner or screener has no replay, so it gets one over the cut frames, whose
    # last row IS `now`. Both index the same way, so there is one walker and one set
    # of semantics rather than a point-in-time walker beside a whole-frame one.
    fe = getattr(cache, "fe", None)
    if fe is None:
        fe = FrameEval(bars, tfs, vocabulary=resolve_vocabulary(vocabulary))
        # The span is not optional here. A point-in-time caller can only ever
        # read the LAST row of each frame, because the frames were just cut at
        # `now`; without a span the end-anchored primitives default to the whole
        # frame and walk every row of history one at a time to produce values
        # nobody asks for. Measured on a three-year 15m SPY cache that took one
        # spec, one symbol, one bar from 0.001s to 11.4s, and the scan registry
        # holds three end-anchored specs run every 15 minutes.
        for tf in tfs.all:
            n = len(bars[tf])
            fe.set_span(tf, max(n - 1, 0), n)
    return MarketContext(symbol=symbol, now=now, tfs=tfs, bars=bars, fe=fe,
                         cursor={tf: len(bars[tf]) - 1 for tf in tfs.all})


def _scheduled_timeframes(dependencies: ReplayDependencies) -> TimeframeSet:
    """The declared timeframes as a `TimeframeSet`, for the grammar's use only.

    `FrameEval` and `ctx.driving_bars` both need one. Nothing in the scheduled
    path reads its deltas or its session-aligned set to decide visibility;
    that answer comes from the schedule and only from the schedule.
    """
    return TimeframeSet(
        driving=BASE_TIMEFRAME,
        higher=tuple(tf for tf in dependencies.timeframes if tf != BASE_TIMEFRAME),
        deltas=DEFAULT_TIMEFRAMES.deltas,
        session_aligned=DEFAULT_TIMEFRAMES.session_aligned,
    )


def build_scheduled_context(prepared: _ValidatedPortfolioBars, symbol: str,
                            now: pd.Timestamp, schedule: ValidatedSchedule,
                            dependencies: ReplayDependencies) -> MarketContext:
    """One symbol's causal context at the scheduled base close `now`.

    Each frame is cut to the rows the schedule has released: base bars that
    have fully closed, and context bars whose `available_at` has arrived. The
    cut is a prefix because a prepared frame's labels ARE the schedule's
    labels, in the schedule's order.

    The prefixes are views rather than copies, which is safe and deliberate.
    Under copy-on-write, writing into one of them copies first, so a strategy
    that mutates `ctx.bars[tf]` changes its own view and never the engine's
    frame. Copying per bar per timeframe would cost a replay dearly for a
    guarantee pandas already gives.
    """
    from nakagai.strategies.rules.frame_eval import FrameEval
    _require_instance(prepared, "prepared", _ValidatedPortfolioBars)
    _require_instance(schedule, "schedule", ValidatedSchedule)
    _require_instance(dependencies, "dependencies", ReplayDependencies)
    symbol = _require_symbol(symbol, "symbol")
    now = _require_timestamp(now, "now")
    closed = schedule.closed_base_count(now)
    if not closed or schedule.base_intervals[closed - 1].close_ts != now:
        raise ReplayInputError(
            "invalid_context_time",
            "a context is built at a scheduled base interval close",
            {"field": "now", "now": now.isoformat()},
        )
    bars = {
        tf: prepared.frame(symbol, tf).iloc[:(
            closed if tf == BASE_TIMEFRAME
            else schedule.available_context_count(tf, now))]
        for tf in dependencies.timeframes
    }
    tfs = _scheduled_timeframes(dependencies)
    fe = FrameEval(bars, tfs, vocabulary=resolve_vocabulary(None))
    # The span is not optional, for the same reason it is not optional in
    # build_context: without one, the end-anchored primitives walk the whole
    # frame to produce values no caller can read.
    for tf in tfs.all:
        rows = len(bars[tf])
        fe.set_span(tf, max(rows - 1, 0), rows)
    return MarketContext(symbol=symbol, now=now, tfs=tfs, bars=bars, fe=fe,
                         cursor={tf: len(bars[tf]) - 1 for tf in tfs.all})
