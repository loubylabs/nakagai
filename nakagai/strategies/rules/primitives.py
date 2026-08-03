"""Stateful/session-aware primitives for RuleSpec v2 expressions.

Each primitive is fn(ctx, bars, **args) -> Series aligned to bars.index (or a
float). `bars` is the spec's driving-timeframe frame. Vocabulary declarations
live in vocabulary.py.
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


def rvol(ctx: MarketContext, bars: pd.DataFrame, sessions: int = 20) -> pd.Series:
    """This bar's volume over the MEDIAN volume of the same clock time across
    the trailing `sessions` sessions, the current session excluded.

    What this replaces is the catalog's own idiom, a volume threshold written
    against `sma(50, of=volume)`, which on intraday bars reads the clock rather
    than the tape. Volume has a session shape: the bars just after the open and
    just before the close carry a multiple of what midday carries, every single
    day, whether or not anything is happening. Measured on three years of 15m
    bars, the median 09:30 bar already sits at 2.78x its own rolling 50-bar
    mean for NVDA, 2.21x for AAPL and 1.27x for JNJ, while the 12:30 bar sits
    near 0.55x for all three. So one written threshold asks two different
    questions: `> 3x mean` fires on roughly half of NVDA's ordinary mornings,
    and at midday demands about 5.7x the genuine normal. Comparing a bar only
    against other bars from its own place in the session takes the shape out,
    so the threshold means the same thing at 09:30 as it does at 12:30.

    Three choices worth naming, because each has a plausible-looking
    alternative that is wrong:

    The baseline is a median, not a mean. Ordinary is the thing being measured,
    and one halted or news-driven session inside the window drags a mean far
    enough to hide the next genuine surge behind it.

    It is shifted by one occurrence, so a session is never part of its own
    baseline. Without that, a large bar lifts the median it is measured against
    and reads tamer than it is, exactly when it matters.

    It buckets on the EXCHANGE clock, not on the UTC label. The 09:30 bar is
    14:30 UTC in winter and 13:30 UTC in summer, so a UTC bucket splits in two
    every March and November and blanks the primitive for `sessions` days each
    time.

    NaN until a bucket holds `sessions` earlier occurrences: half a window is a
    different measurement, not a rough one.
    """
    n = int(sessions)
    ny = bars.index.tz_convert(EXCHANGE_TZ)
    clock = ny.hour * 60 + ny.minute
    # Shift INSIDE the bucket, so the step back is one occurrence of this clock
    # time rather than one calendar session. Half days and late opens are
    # ordinary: a session with no 15:45 bar should simply not be in the 15:45
    # bucket, rather than punch a hole through the next `sessions` days of it.
    # The lambda is a Python loop over buckets, but a bucket is one clock time,
    # so their number is bounded by the bars in a session and does not grow as
    # history accumulates, which is the property _opening_range's cost guard
    # exists to protect.
    baseline = bars["volume"].groupby(clock).transform(
        lambda v: v.shift(1).rolling(n).median())
    # A zero baseline means the symbol did not trade at this clock time in most
    # of the trailing window. That is not an infinite surge, it is an absence of
    # anything to compare against, and a condition reads NaN as False.
    return (bars["volume"] / baseline.where(baseline > 0)).rename(None)


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


def end_anchored_series(term, ctx, bars: pd.DataFrame, lo: int, hi: int,
                        **args) -> pd.Series:
    """Row `i` holds exactly what term.fn returns for bars[:i+1].

    Calling the scalar function per row rather than reimplementing it makes the
    equivalence a tautology: there is no second implementation to drift.
    """
    idx = bars.index[lo:hi]
    if not len(idx):
        return pd.Series(dtype="float64", index=idx)
    return pd.Series([float(term.fn(ctx, bars.iloc[: i + 1], **args))
                      for i in range(lo, hi)], index=idx, dtype="float64")
