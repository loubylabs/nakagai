"""Shared strategy plumbing: freshness gates + signal builder.

The engine replays driving-timeframe bars, so a condition computed on a
higher timeframe stays true for every driving-timeframe step inside that bar.
Templates gate on these freshness checks to fire exactly once per
driving-timeframe bar.
"""

import pandas as pd

from nakagai.strategies.base import Direction, MarketContext, Signal

NY = "America/New_York"


def fresh_bar(ctx: MarketContext, timeframe: str) -> bool:
    """True only on the first driving step after a `timeframe` bar completes."""
    bars = ctx.bars[timeframe]
    if bars.empty:
        return False
    close_ts = bars.index[-1] + ctx.tfs.deltas[timeframe]
    return (ctx.now - close_ts) < ctx.tfs.step


def first_bar_of_session(ctx: MarketContext) -> bool:
    """True on the first completed driving bar of a new NY session: the
    once-a-day gate for strategies driven by session-aligned bars."""
    b = ctx.driving_bars
    if len(b) < 2:
        return len(b) == 1
    ny = b.index.tz_convert(NY)
    return ny[-1].date() != ny[-2].date()


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
