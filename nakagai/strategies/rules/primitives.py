"""Stateful/session-aware primitives for RuleSpec v2 expressions.

Each primitive is fn(ctx, bars, **args) -> Series aligned to bars.index (or a
float). `bars` is the spec's driving-timeframe frame. The registry maps the
grammar name to an arg schema (validated in spec.py) and the function.
"""

import numpy as np
import pandas as pd

from nakagai.strategies.base import Direction, MarketContext
from nakagai.strategies.ict.fvg import find_fvgs
from nakagai.strategies.ict.primitives import _strict_extrema, atr as _ict_atr
from nakagai.data.schema import EXCHANGE_TZ


def _ny_dates(bars: pd.DataFrame) -> np.ndarray:
    return np.asarray(bars.index.tz_convert(EXCHANGE_TZ).date)


def _session_groups(bars: pd.DataFrame):
    return bars.groupby(_ny_dates(bars))


def opening_range_high(ctx: MarketContext, bars: pd.DataFrame, minutes: int = 30) -> pd.Series:
    return _opening_range(bars, int(minutes), "high", "max")


def opening_range_low(ctx: MarketContext, bars: pd.DataFrame, minutes: int = 30) -> pd.Series:
    return _opening_range(bars, int(minutes), "low", "min")


def _opening_range(bars: pd.DataFrame, minutes: int, col: str, how: str) -> pd.Series:
    """The session's opening-range level, NaN until the range has fully elapsed.

    Fully vectorized, and it has to stay that way. This runs once per replayed
    bar, so the per-session Python loop it replaced made a window replay
    O(sessions x bars) and grew heavier every month as history accumulated. At
    three years of 15m bars that loop cost ~90ms per call against ~2.6ms here,
    which is what put permutation testing out of reach: a single 638-bar window
    spent minutes inside this function alone. tests/test_primitives.py guards
    both the shape of the answer and the cost.
    """
    if not len(bars.index):
        return pd.Series(np.nan, index=bars.index, dtype="float64")
    days = _ny_dates(bars)
    ts = pd.Series(bars.index, index=bars.index)
    # Each session's range is measured from ITS OWN first bar, not from a wall
    # clock: a late open or a half day has to measure from where it actually
    # started, which is why this is a groupby-first rather than a fixed time.
    edge = ts.groupby(days).transform("first") + pd.Timedelta(minutes=minutes)
    level = bars[col].where(ts < edge).groupby(days).transform(how)
    # No lookahead: the level is invisible until its own window has elapsed, so
    # a session that closes inside the range never gets one.
    return level.where(ts >= edge).rename(None)


def _prev_session(bars: pd.DataFrame, col: str, agg) -> pd.Series:
    per_day = _session_groups(bars)[col].agg(agg)
    prev = per_day.shift(1)
    return pd.Series(prev.loc[_ny_dates(bars)].to_numpy(), index=bars.index)


def prev_session_high(ctx: MarketContext, bars: pd.DataFrame) -> pd.Series:
    return _prev_session(bars, "high", "max")


def prev_session_low(ctx: MarketContext, bars: pd.DataFrame) -> pd.Series:
    return _prev_session(bars, "low", "min")


def prev_session_close(ctx: MarketContext, bars: pd.DataFrame) -> pd.Series:
    return _prev_session(bars, "close", "last")


def gap_pct(ctx: MarketContext, bars: pd.DataFrame) -> pd.Series:
    """Today's session open vs the prior session close, in percent."""
    opens = _session_groups(bars)["open"].transform("first")
    prev_close = prev_session_close(ctx, bars)
    return 100 * (opens - prev_close) / prev_close


def swing_high(ctx: MarketContext, bars: pd.DataFrame, k: int = 3) -> pd.Series:
    return _swing(bars, "high", int(k), find_max=True)


def swing_low(ctx: MarketContext, bars: pd.DataFrame, k: int = 3) -> pd.Series:
    return _swing(bars, "low", int(k), find_max=False)


def _swing(bars: pd.DataFrame, col: str, k: int, find_max: bool) -> pd.Series:
    values = bars[col].to_numpy()
    mask = _strict_extrema(values, k, find_max)
    # a swing at i is only KNOWN k bars later; stamp it there, then carry forward
    out = pd.Series(np.nan, index=bars.index)
    idx = np.flatnonzero(mask)
    confirm = idx + k
    keep = confirm < len(bars)
    out.iloc[confirm[keep]] = values[idx[keep]]
    return out.ffill()


def leg_retrace(ctx: MarketContext, bars: pd.DataFrame,
                direction: str = "long", k: int = 3) -> pd.Series:
    """Position of the close inside the last confirmed swing range.
    long: (H - close) / (H - L), so 0 = at the swing high, 0.5 = equilibrium,
    1 = full retrace to the swing low; the ICT OTE band is [0.62, 0.79].
    short mirrors from the low. NaN until both swings exist or when H <= L."""
    hi = _swing(bars, "high", int(k), find_max=True)
    lo = _swing(bars, "low", int(k), find_max=False)
    rng = (hi - lo).where((hi - lo) > 0)
    if direction == "long":
        return (hi - bars["close"]) / rng
    return (bars["close"] - lo) / rng


def order_block(ctx: MarketContext, bars: pd.DataFrame,
                direction: str = "long", field: str = "top",
                body_atr: float = 1.5, lookback: int = 40) -> float:
    """Range boundary of the last opposing candle before the most recent
    displacement candle (body >= body_atr * ATR) in the lookback window: the
    ICT order block. NaN when no displacement or no opposing candle exists."""
    df = bars.tail(int(lookback))
    a = _ict_atr(df)
    if len(df) < 2 or not a or np.isnan(a):
        return float("nan")
    body = (df["close"] - df["open"]).to_numpy()
    disp = body >= body_atr * a if direction == "long" else body <= -body_atr * a
    disp_idx = np.flatnonzero(disp)
    if not disp_idx.size:
        return float("nan")
    i = disp_idx[-1]
    opp = np.flatnonzero(body[:i] < 0) if direction == "long" else np.flatnonzero(body[:i] > 0)
    if not opp.size:
        return float("nan")
    ob = df.iloc[opp[-1]]
    top, bottom = float(ob["high"]), float(ob["low"])
    return {"top": top, "bottom": bottom, "mid": (top + bottom) / 2}[field]


def day_of_week(ctx: MarketContext, bars: pd.DataFrame) -> pd.Series:
    """Weekday of each bar's session, 0 = Monday. Intraday labels convert to
    NY time; session-aligned daily bars are labeled midnight UTC of their
    session date, where the NY conversion would land on the prior evening,
    so exact-midnight labels read the UTC calendar day instead."""
    idx = bars.index
    midnight_utc = (idx.hour == 0) & (idx.minute == 0)
    dow = np.where(midnight_utc, idx.dayofweek, idx.tz_convert(EXCHANGE_TZ).dayofweek)
    return pd.Series(dow.astype(float), index=idx)


def minutes_into_session(ctx: MarketContext, bars: pd.DataFrame) -> pd.Series:
    starts = bars.index.to_series().groupby(_ny_dates(bars)).transform("first")
    return ((bars.index.to_series() - starts).dt.total_seconds() / 60).astype(float)


def bars_since(ctx: MarketContext, bars: pd.DataFrame, cond: dict, eval_fn=None) -> pd.Series:
    """Bars elapsed since `cond` was last elementwise-true. NaN before the
    first True. eval_fn(cond, bars) -> boolean Series is injected by the
    evaluator to avoid a circular import."""
    if eval_fn is None:
        raise ValueError("bars_since needs the evaluator's eval_fn")
    mask = eval_fn(cond, bars).astype(bool)
    pos = pd.Series(np.arange(len(bars), dtype=float), index=bars.index)
    last_true = pos.where(mask).ffill()
    return pos - last_true


def fvg_nearest(ctx: MarketContext, bars: pd.DataFrame,
                direction: str = "long", field: str = "top",
                state: str = "open", min_size_atr: float = 0.25,
                lookback: int = 40) -> float:
    """Boundary of the qualifying FVG nearest the last close, for the given
    trade direction. state "open" = unfilled gap in that direction; state
    "inverted" = a gap of the OPPOSITE original direction whose far boundary
    a later bar closed through, so the zone now supports this direction.
    Returns NaN when none exists (condition reads False)."""
    want = Direction.LONG if direction == "long" else Direction.SHORT
    if state == "open":
        gaps = [f for f, s in find_fvgs(bars, min_size_atr, int(lookback))
                if s == "open" and f.direction == want]
    else:
        origin = Direction.SHORT if want == Direction.LONG else Direction.LONG
        gaps = [f for f, s in find_fvgs(bars, min_size_atr, int(lookback))
                if s == "inverted" and f.direction == origin]
    if not gaps or bars.empty:
        return float("nan")
    ref = float(bars["close"].iloc[-1])
    best = min(gaps, key=lambda f: min(abs(ref - f.top), abs(ref - f.bottom)))
    return float({"top": best.top, "bottom": best.bottom,
                  "mid": (best.top + best.bottom) / 2}[field])


PRIMITIVES: dict[str, dict] = {
    "opening_range_high": {"args": {"minutes": (5, 120)}, "fn": opening_range_high},
    "opening_range_low": {"args": {"minutes": (5, 120)}, "fn": opening_range_low},
    "prev_session_high": {"args": {}, "fn": prev_session_high},
    "prev_session_low": {"args": {}, "fn": prev_session_low},
    "prev_session_close": {"args": {}, "fn": prev_session_close},
    "gap_pct": {"args": {}, "fn": gap_pct},
    "swing_high": {"args": {"k": (1, 10)}, "fn": swing_high},
    "swing_low": {"args": {"k": (1, 10)}, "fn": swing_low},
    "day_of_week": {"args": {}, "fn": day_of_week},
    "minutes_into_session": {"args": {}, "fn": minutes_into_session},
    "bars_since": {"args": {"cond": "condition"}, "fn": bars_since},
    "fvg_nearest": {"args": {"direction": ("long", "short"),
                             "field": ("top", "bottom", "mid"),
                             "state": ("open", "inverted"),
                             "min_size_atr": (0.05, 2.0),
                             "lookback": (10, 200)}, "fn": fvg_nearest},
    "leg_retrace": {"args": {"direction": ("long", "short"), "k": (1, 10)},
                    "fn": leg_retrace},
    "order_block": {"args": {"direction": ("long", "short"),
                             "field": ("top", "bottom", "mid"),
                             "body_atr": (0.5, 5.0), "lookback": (10, 200)},
                    "fn": order_block},
}
# Session/calendar-scoped primitives read the driving frame's own session or
# calendar structure (session-open windows, elapsed session minutes, the bar's
# calendar weekday) rather than plain OHLCV structure. Feeding them a `tf` swaps
# in a different frame's bars, which silently degenerates: opening_range_* on
# "1d" bars (one bar per session) never sees its window elapse and is NaN
# forever, on "1h" bars a minutes=30 window is measured in whole-hour steps;
# minutes_into_session on "1d" bars is 0 everywhere (one bar per session group);
# day_of_week reads calendar identity that belongs to the spec's own session,
# so answering it from a foreign frame is a category error even though the
# primitive itself now handles midnight-UTC daily labels. These primitives
# must always run on the spec's own driving bars, so `tf` is rejected outright.
SESSION_SCOPED_PRIMS = frozenset({
    "opening_range_high", "opening_range_low", "minutes_into_session", "day_of_week",
})
ARG_DEFAULTS: dict[str, dict] = {
    "opening_range_high": {"minutes": 30}, "opening_range_low": {"minutes": 30},
    "swing_high": {"k": 3}, "swing_low": {"k": 3},
    "fvg_nearest": {"direction": "long", "field": "top", "state": "open",
                    "min_size_atr": 0.25, "lookback": 40},
    "leg_retrace": {"direction": "long", "k": 3},
    "order_block": {"direction": "long", "field": "top",
                    "body_atr": 1.5, "lookback": 40},
}

# Primitives whose value is anchored to the END of the frame they are handed:
# one float from the tail, not a causal series. A whole-frame pass may not
# broadcast that across history (it would be lookahead), so they are evaluated
# row by row over a bounded span instead. Bounded by `lookback`, so this costs
# what the per-bar path always cost.
END_ANCHORED = frozenset({"fvg_nearest", "order_block"})


def end_anchored_series(name: str, ctx, bars: pd.DataFrame, lo: int, hi: int,
                        **args) -> pd.Series:
    """Row `i` holds exactly what PRIMITIVES[name] returns for bars[:i+1].

    Calling the scalar function per row rather than reimplementing it makes the
    equivalence a tautology: there is no second implementation to drift.
    """
    fn = PRIMITIVES[name]["fn"]
    idx = bars.index[lo:hi]
    if not len(idx):
        return pd.Series(dtype="float64", index=idx)
    return pd.Series([float(fn(ctx, bars.iloc[: i + 1], **args))
                      for i in range(lo, hi)], index=idx, dtype="float64")
