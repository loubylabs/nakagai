"""Point-in-time MarketContext assembly. The ONLY door strategies get to data.

Two doors, and they answer to different clocks. `build_context` serves the
live scanner and the screener, which have no schedule and reconstruct
visibility from the bar labels themselves through `closed_before`.
`build_scheduled_context` serves the portfolio replay, which has an embedded
`ReplaySchedule` and therefore asks it: a bar is visible when the schedule
says it became available, never because label arithmetic put it in the past.

Each door answers two questions, not one. VISIBILITY is which rows a strategy
may read, and the EMISSION GATE is which close a play decided off the driving
frame may signal at, which `ctx.fresh` carries. They are genuinely different:
an hourly bar is readable for every base close of the hour after it and
entitles a decision at exactly one of them. A strategy asks the context for
both and derives neither, so a schedule cannot be overruled downstream of it.
"""

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import pandas as pd

from nakagai.data.cache import BarCache
from nakagai.data.schema import DEFAULT_TIMEFRAMES, EXCHANGE_TZ, TimeframeSet
from nakagai.engine.bars import (
    BASE_TIMEFRAME,
    ReplayDependencies,
    _ValidatedPortfolioBars,
    _normalized_frame,
    _require_prepared_closure,
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
from nakagai.strategies.util import label_freshness


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
                  reference_pairs: tuple[tuple[str, str], ...],
                  vocabulary: Vocabulary | None = None,
                  facts: Mapping[str, float | int | None] | None = None,
                  ) -> MarketContext:
    """Point-in-time context at `now`.

    closed_before still runs per timeframe per call. It is a searchsorted plus
    a zero-copy .iloc slice, measured at 1.5% of replay, and ctx.bars[tf] has to
    stay a real prefix frame because _fresh, rr_signal, stop_target and every
    non-rule strategy read it. Removing it would trade that 1.5% for a new
    invariant to defend.
    """
    from nakagai.strategies.rules.frame_eval import FrameEval
    references = ReplayDependencies(
        timeframes=tuple(tfs.all), reference_pairs=reference_pairs,
    ).reference_pairs
    driving_frames = {
        (symbol, tf): cache.load(symbol, tf) for tf in tfs.all
    }
    pair_frames = dict(driving_frames)
    for pair in references:
        if pair in pair_frames:
            continue
        loaded = cache.load(*pair)
        expected = driving_frames[(symbol, pair[1])].index
        normalized = _normalized_frame(loaded, pair)
        pair_frames[pair] = normalized.reindex(expected)
    visible = {
        pair: closed_before(frame, pair[1], now, tfs)
        for pair, frame in pair_frames.items()
    }
    bars = {tf: visible[(symbol, tf)] for tf in tfs.all}
    # The evaluator sits over the CUT frames, whose last row is `now`, so this
    # door and `build_scheduled_context` index the same way and there is one
    # walker with one set of semantics rather than a point-in-time walker
    # beside a whole-frame one.
    fe = FrameEval(
        symbol, MappingProxyType(visible), tfs,
        vocabulary=resolve_vocabulary(vocabulary),
        facts=facts,
    )
    # The span is not optional here. A point-in-time caller can only ever read
    # the LAST row of each frame, because the frames were just cut at `now`;
    # without a span the end-anchored primitives default to the whole frame and
    # walk every row of history one at a time to produce values nobody asks
    # for. Measured on a three-year 15m SPY cache that took one spec, one
    # symbol, one bar from 0.001s to 11.4s, and the scan registry holds three
    # end-anchored specs run every 15 minutes.
    for pair, frame in visible.items():
        n = len(frame)
        fe.set_span(*pair, max(n - 1, 0), n)
    ctx = MarketContext(
        symbol=symbol, now=now, tfs=tfs, bars=MappingProxyType(bars), fe=fe,
        cursor=MappingProxyType({tf: len(bars[tf]) - 1 for tf in tfs.all}))
    # Assigned after the context exists because the label rule reads a context:
    # the session gate asks the driving frame which bar opened the session. The
    # door owns the value either way, and nothing downstream may replace it.
    ctx.fresh = MappingProxyType(label_freshness(ctx))
    return ctx


def _scheduled_timeframes(dependencies: ReplayDependencies) -> TimeframeSet:
    """The declared timeframes as a `TimeframeSet`, for the grammar's use only.

    `FrameEval` and `ctx.driving_bars` both need one. Which rows a strategy
    sees, and which close it may decide on, come from the schedule and only
    from the schedule; the deltas here are the grammar's, for lifting one
    frame's series onto another's index, and no visibility or freshness rule
    reads them.
    """
    return TimeframeSet(
        driving=BASE_TIMEFRAME,
        higher=tuple(tf for tf in dependencies.timeframes if tf != BASE_TIMEFRAME),
        deltas=DEFAULT_TIMEFRAMES.deltas,
        session_aligned=DEFAULT_TIMEFRAMES.session_aligned,
    )


def build_scheduled_context(prepared: _ValidatedPortfolioBars, symbol: str,
                            now: pd.Timestamp, schedule: ValidatedSchedule,
                            dependencies: ReplayDependencies, *,
                            vocabulary: Vocabulary) -> MarketContext:
    """One symbol's causal context at the scheduled base close `now`.

    Each frame is cut to the rows the schedule has released: base bars that
    have fully closed, and context bars whose `available_at` has arrived. The
    cut is a prefix because a prepared frame's labels ARE the schedule's
    labels, in the schedule's order.

    `vocabulary` is the grammar the evaluator behind `ctx.fe` reads a spec
    under, and it is REQUIRED and keyword-only rather than defaulted. The
    caller is one play's runtime construction, which knows the definition's own
    grammar; a default here would be silently taken by every replay and would
    decide entries under the core grammar while the IC lens graded the same
    play under the definition's, with no digest moving to say so. Keyword-only
    for the reason every `vocabulary` in this codebase is: bound positionally
    it would land in an earlier parameter and nothing would raise.

    `now` must be a scheduled base close inside the test range. The schedule
    runs on to `ic_tail_end`, and those tail closes are scheduled closes too,
    but tail bars belong to the IC lens after the trading replay finishes and
    no strategy, management call, order, benchmark, or equity point may read
    them. This is the only door that hands bars to a strategy, so it is where
    that rule is enforced.

    The prefixes are views rather than copies, which is safe and deliberate.
    Copying per bar per timeframe would cost a replay dearly for a guarantee
    pandas already gives. It gives that guarantee to the ENGINE, though, and
    not between two consumers: call this once per consumer, so a write lands
    inside the caller that made it.
    """
    from nakagai.strategies.rules.frame_eval import FrameEval
    _require_instance(prepared, "prepared", _ValidatedPortfolioBars)
    _require_instance(schedule, "schedule", ValidatedSchedule)
    _require_instance(dependencies, "dependencies", ReplayDependencies)
    _require_instance(vocabulary, "vocabulary", Vocabulary)
    _require_prepared_closure(prepared, schedule.request, dependencies)
    symbol = _require_symbol(symbol, "symbol")
    now = _require_timestamp(now, "now")
    closed = schedule.closed_base_count(now)
    if (not closed or schedule.base_intervals[closed - 1].close_ts != now
            or now > schedule.request.window.test_end):
        raise ReplayInputError(
            "invalid_context_time",
            "a context is built at a scheduled base interval close inside the "
            "test range",
            {"field": "now", "now": now.isoformat()},
        )
    # These slices ALIAS the engine's frames. What makes that safe is the
    # `pandas>=3` floor in pyproject.toml: copy-on-write is unconditional
    # there, so a strategy writing into one of them copies first and the
    # engine's own prices cannot move. Do not lower that floor, and do not
    # let this become a plain `.iloc` on a frame the engine still trusts
    # under an older pandas.
    #
    # Copy-on-write protects the ENGINE and nothing else. A write copies away
    # from the engine's frame and then mutates the object it was made on, so
    # two consumers holding one slice still read each other's writes. What
    # separates them is a slice each, which is why the replay calls this once
    # per runtime rather than once per symbol.
    def visible_rows(timeframe: str) -> int:
        return (closed if timeframe == BASE_TIMEFRAME
                else schedule.available_context_count(timeframe, now))

    pair_bars = {
        (symbol, tf): prepared.frame(symbol, tf).iloc[:visible_rows(tf)]
        for tf in dependencies.timeframes
    }
    for pair in dependencies.reference_pairs:
        if pair not in pair_bars:
            pair_bars[pair] = prepared.frame(*pair).iloc[:visible_rows(pair[1])]
    bars = {tf: pair_bars[(symbol, tf)] for tf in dependencies.timeframes}
    tfs = _scheduled_timeframes(dependencies)
    fe = FrameEval(
        symbol, MappingProxyType(pair_bars), tfs, vocabulary=vocabulary)
    # The span is not optional, for the same reason it is not optional in
    # build_context: without one, the end-anchored primitives walk the whole
    # frame to produce values no caller can read.
    for pair, frame in pair_bars.items():
        rows = len(frame)
        fe.set_span(*pair, max(rows - 1, 0), rows)
    # Read-only mappings, which is a narrower claim than it looks and is not
    # what isolates two runtimes: that is one context each, decided by the
    # caller. These stop a strategy REPLACING an answer the door owns, and the
    # one that matters is `fresh`. A play that could rebind its own gate could
    # decide on a close the schedule never released it for, which is the whole
    # rule this door exists to enforce.
    return MarketContext(
        symbol=symbol, now=now, tfs=tfs, bars=MappingProxyType(bars), fe=fe,
        cursor=MappingProxyType({tf: len(bars[tf]) - 1 for tf in tfs.all}),
        fresh=MappingProxyType(_scheduled_freshness(schedule, tfs, now)))


def _scheduled_freshness(schedule: ValidatedSchedule, tfs: TimeframeSet,
                         now: pd.Timestamp) -> dict[str, bool]:
    """Which higher timeframes the SCHEDULE says may be decided on at `now`.

    The newest bar released at `now`, and whether the schedule calls it fresh
    here. Freshness is the emission gate and it is not availability: an hourly
    bar is readable for every base close of the hour after it and entitles a
    decision at exactly one of them, the close its own `fresh_context_at`
    names. A bar whose `fresh_context_at` is null, which is what an early close
    does to the noon four-hour bucket, entitles no decision at all.

    Only the newest released bar can be the one: freshness sits inside
    `[period_end, period_end + one base bar)` and a bar is released at its own
    period end, so a bar fresh at `now` was released at or within one base bar
    of it and nothing later can have been released yet.

    This is why the replay does not reconstruct the instant. A four-hour bucket
    is four EASTERN WALL-CLOCK hours, so across a daylight-saving change it is
    three absolute hours or five, and `label + 4h` names an instant an hour
    away from the one the bucket actually ended at. The schedule already
    carries the answer; asking it is the whole point of carrying it.
    """
    fresh: dict[str, bool] = {}
    for tf in tfs.higher:
        released = schedule.available_context(tf, now)
        fresh[tf] = bool(released) and released[-1].fresh_context_at == now
    return fresh
