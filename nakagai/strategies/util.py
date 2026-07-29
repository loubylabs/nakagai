"""Shared strategy plumbing: freshness gates + signal builder.

The engine replays driving-timeframe bars, so a condition computed on a
higher timeframe stays true for every driving-timeframe step inside that bar.
Templates gate on these freshness checks to fire exactly once per
driving-timeframe bar.
"""

import pandas as pd

from nakagai.data.schema import EXCHANGE_TZ, SESSION_OPEN
from nakagai.strategies.base import Direction, MarketContext, Signal


def fresh_bar(ctx: MarketContext, timeframe: str) -> bool:
    """True only on the first driving step after a `timeframe` bar completes."""
    bars = ctx.bars[timeframe]
    if bars.empty:
        return False
    close_ts = bars.index[-1] + ctx.tfs.deltas[timeframe]
    return (ctx.now - close_ts) < ctx.tfs.step


def session_open(ts: pd.Timestamp) -> pd.Timestamp:
    """The regular open of the NY session `ts` falls in, in UTC.

    Built from the exchange-local calendar fields rather than by adding 9h30m
    to midnight, so it stays the 09:30 WALL time across a DST change instead of
    drifting an hour on the two Sundays a year the offset moves.
    """
    ny = ts.tz_convert(EXCHANGE_TZ)
    return pd.Timestamp(year=ny.year, month=ny.month, day=ny.day,
                        hour=SESSION_OPEN[0], minute=SESSION_OPEN[1],
                        tz=EXCHANGE_TZ).tz_convert("UTC")


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
    O(history) per bar, and this is a binary search plus one construction.
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
    when the geometry is degenerate (stop on the wrong side, NaNs)."""
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
    return Signal(symbol=ctx.symbol, direction=direction, entry=None,
                  stop=float(stop), target=float(tgt), confidence=confidence,
                  setup_tags=tags, rationale=rationale)
