"""Shared strategy plumbing: freshness gates + signal builder.

The engine replays driving-timeframe bars, so a condition computed on a
higher timeframe stays true for every driving-timeframe step inside that bar.
A play decided off the driving frame therefore needs a gate that fires exactly
once per higher-timeframe bar, and `MarketContext.fresh` carries the answer.

Everything here reconstructs that answer from bar labels, which is what a
caller with no schedule has to do and what `engine/context.build_context`
calls. It is NOT what a portfolio replay does: a replay's schedule names
`fresh_context_at` for every context bar, and that is the only correct answer
once early closes, holidays, and daylight saving are in play. Do not reach for
these from a scheduled path, and do not reintroduce a strategy-side call: a
gate a strategy recomputes is a gate the schedule cannot decide.
"""

import math

import pandas as pd

from nakagai.data.schema import session_open
from nakagai.engine.portfolio_types import Signal
from nakagai.strategies.base import Direction, MarketContext


def label_freshness(ctx: MarketContext) -> dict[str, bool]:
    """Every higher timeframe's emission gate, reconstructed from labels.

    The unscheduled door's whole answer, in one place, so the two ways a
    timeframe can be gated live beside each other rather than inside a
    strategy. A session-aligned timeframe carries a date rather than a close
    time, so it is gated on the driving bar that opens the session; every other
    one is gated on its own label plus its delta.

    The driving timeframe is absent on purpose. It has no gate: a play decided
    on the frame it is replayed on is fresh on every step of it, and an entry
    here would invite a caller to ask a question with only one answer.
    """
    return {
        tf: (first_bar_of_session(ctx) if tf in ctx.tfs.session_aligned
             else fresh_bar(ctx, tf))
        for tf in ctx.tfs.higher
    }


def fresh_bar(ctx: MarketContext, timeframe: str) -> bool:
    """True only on the first driving step after a `timeframe` bar completes.

    Label plus a fixed absolute delta, which is exact for an hourly bar and
    exact for a four-hour bucket that does not span a daylight-saving change.
    A schedule answers this from `fresh_context_at` instead and is authoritative
    wherever one exists; see the module docstring.
    """
    bars = ctx.bars[timeframe]
    if bars.empty:
        return False
    close_ts = bars.index[-1] + ctx.tfs.deltas[timeframe]
    return (ctx.now - close_ts) < ctx.tfs.step


def first_bar_of_session(ctx: MarketContext) -> bool:
    """True on the driving bar that OPENS a new regular session: the once-a-day
    gate for strategies driven by session-aligned bars.

    Anchored on the 09:30 open, not on the calendar date changing. Those look
    equivalent and are not, because the caches are not RTH-only: providers
    return sporadic pre-market prints, and on roughly half of the sessions in a
    three-year SPY cache the first bar of the date is one of them, at 08:00 or
    08:15. Under the date-change reading the gate fired on THAT bar, which no
    live scanner ever visits (it wakes 09:45-16:00), so every daily play went
    dark for the whole session while the same play entered normally in replay.
    The two agree again now, and they agree on the bar a daily signal actually
    means: the one the session opens on.

    A session whose 09:30 bar is missing fires on the first bar that is there,
    one bar late, which is the honest answer rather than skipping the day.

    Cheaper than what it replaces, which matters because a replay calls this
    once per bar per daily play: converting the whole index to New York was
    O(history) per bar, and this is a binary search plus one session-open
    lookup, which data/schema.session_open memoizes per session. Keep both
    halves cheap. This gate is what every bar of a session that is NOT the
    open stops at, so whatever it costs is paid on the bars that do no other
    work, and it is the one place where a per-bar cost hides best.
    """
    idx = ctx.driving_bars.index
    if not len(idx):
        return False
    return idx.searchsorted(session_open(idx[-1]), side="left") == len(idx) - 1


def rr_signal(ctx: MarketContext, direction: Direction, stop: float, rr: float,
              tags: tuple[str, ...], rationale: str, confidence: float = 0.55,
              target: float | None = None) -> Signal | None:
    """Build a market-entry signal with an explicit stop and either a fixed
    reward:risk target (default) or a structural target price. Returns None
    when the geometry is degenerate (stop on the wrong side, NaNs).

    The deciding raw close is the signal's `entry_ref`, and it is the same
    reference the geometry below is checked against. Phase 1 fills market at
    the next scheduled open, so this is a reference, never a limit price."""
    if ctx.driving_bars.empty:
        return None
    ref = float(ctx.driving_bars["close"].iloc[-1])
    if pd.isna(stop) or pd.isna(ref):
        return None
    if direction == Direction.LONG:
        if stop >= ref:
            return None
        tgt = target if target is not None else ref + rr * (ref - stop)
        if tgt <= ref:
            return None
    else:
        if stop <= ref:
            return None
        tgt = target if target is not None else ref - rr * (stop - ref)
        if tgt >= ref:
            return None
    # Every level has to be a price. A wide reward:risk on a percent stop
    # walks a SHORT target down through zero (rr 20 on a 5% stop lands on it
    # exactly), and zero is not a level the market can reach. Degenerate, so
    # it returns None here rather than reaching the boundary as a signal the
    # replay would have to abort on.
    if not all(math.isfinite(float(v)) and float(v) > 0.0 for v in (ref, stop, tgt)):
        return None
    return Signal(symbol=ctx.symbol, direction=direction, entry_ref=ref,
                  stop=float(stop), target=float(tgt), confidence=confidence,
                  setup_tags=tags, rationale=rationale)
