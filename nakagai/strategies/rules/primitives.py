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
from nakagai.data.schema import (EXCHANGE_TZ, _is_session_frame, rth_mask,
                                 session_opens)


def _session_keys(index: pd.DatetimeIndex) -> np.ndarray:
    """Which session each bar belongs to, as a groupby key.

    The session's own 09:30 open, as the index's int64 values. One open per New
    York calendar date, so this is exactly the partition the New York date
    gives and exactly the one the Pine side's nk_session_key builds; it is the
    same fact spelled in the units the rest of the file already needs.

    As asi8 rather than as the timestamps themselves, which is not a
    micro-optimization. Handed datetime64 keys pandas rebuilds a DatetimeIndex
    for the grouper on every call, and handed the New York `date` objects it
    hashes an object array in Python. Measured over three years of 15m bars on
    one machine, gap_pct's three groupbys took 30ms on date keys and 12ms on
    these, and it runs once per replay per spec and once per scanned symbol.
    """
    return session_opens(index).asi8


def opening_range_high(ctx: MarketContext, bars: pd.DataFrame, minutes: int = 30) -> pd.Series:
    return _opening_range(bars, int(minutes), "high", "max")


def opening_range_low(ctx: MarketContext, bars: pd.DataFrame, minutes: int = 30) -> pd.Series:
    return _opening_range(bars, int(minutes), "low", "min")


def _opening_range(bars: pd.DataFrame, minutes: int, col: str, how: str) -> pd.Series:
    """The session's opening-range level, NaN until the range has fully elapsed.

    The window is [09:30, 09:30 + minutes) of each bar's OWN session, taken
    from data/schema.session_opens rather than from whichever bar happens to
    open the calendar date. Those are not the same thing and the difference is
    the whole defect: the caches are not RTH-only, so the first bar of a date
    is ordinarily an 08:00 or 08:30 pre-market print, and measuring from there
    aggregated a thin, barely-traded pre-market band and called it the opening
    range. Every breakout of it fired on a level no opening-range play means,
    on bars that had already printed before the bell.

    Only bars inside the window contribute, which is what makes a genuinely
    late open answer NaN rather than answer something. A name that first trades
    at 10:30 has no [09:30, 10:00) range, and saying so lets the condition read
    False; measuring from 10:30 instead would hand a breakout play a level
    built out of a different half-hour than the one its author wrote down. The
    old code did exactly that, on purpose, and the reasoning was sound for a
    cache that started at the bell and wrong for the cache that exists.

    Fully vectorized, and it has to stay that way. This runs once per replayed
    bar, so the per-session Python loop it replaced made a window replay
    O(sessions x bars) and grew heavier every month as history accumulated. At
    three years of 15m bars that loop cost ~90ms per call against ~2.6ms here,
    which is what put permutation testing out of reach: a single 638-bar window
    spent minutes inside this function alone. Anchoring on the bell costs
    nothing: session_opens is pure index arithmetic and its answer doubles as
    the session grouping key, so it replaces both the object array of New York
    dates and the groupby-transform that used to find each session's first bar.
    Timed side by side on one machine it comes out slightly ahead of the
    calendar-date version it replaces. tests/test_primitives.py guards both the
    shape of the answer and the cost.
    """
    if not len(bars.index):
        return pd.Series(np.nan, index=bars.index, dtype="float64")
    ts = pd.Series(bars.index, index=bars.index)
    opens = session_opens(bars.index)
    start = pd.Series(opens, index=bars.index)
    edge = start + pd.Timedelta(minutes=minutes)
    # `opens.asi8` is _session_keys, spelled out because this function already
    # holds `opens` for the window bounds and asking for them twice would be
    # the expensive half of the call. See _session_keys for why the key is
    # int64 rather than a timestamp or a New York date.
    level = bars[col].where((ts >= start) & (ts < edge)).groupby(
        opens.asi8).transform(how)
    # No lookahead: the level is invisible until its own window has elapsed, so
    # a session that closes inside the range never gets one. Pre-market bars
    # fall under the same clause, since a bar before the bell is before the
    # edge too, so the range cannot be read before it exists.
    return level.where(ts >= edge).rename(None)


def _prev_session(bars: pd.DataFrame, col: str, agg) -> pd.Series:
    """The previous session's REGULAR-HOURS aggregate, on every bar of this one.

    Only bars inside [09:30, 16:00) reach the aggregate. The caches are not
    RTH-only, so the calendar-date aggregate this replaces drew its extremes
    from the pre-market and the post-market as well, where a few hundred shares
    set a price nobody could have traded in size, and its "close" was the last
    post-market print at 19:45 rather than the 15:45 bar that every "yesterday's
    close" in the catalog means. A level the whole catalog leans on as support
    or resistance was a level nothing had defended.

    Masked to NaN and left to the aggregate to skip, rather than filtered out
    of the frame first. `max`, `min` and `last` all skip NaN, so one masked
    column answers all three, and a session with no regular bars at all comes
    back NaN on every one of them. Filtering would drop such a session from the
    grouping instead, and `shift(1)` would then hand the next session the last
    session that HAPPENED to trade: a reading from two or five days ago wearing
    the name "previous session", and nothing in the answer to say so. NaN is
    the honest answer and a condition over it reads False.

    rth_mask carries the session-frame branch, which is what keeps a 1d spec
    working. A daily bar is labeled at midnight under either convention, UTC or
    Eastern, and both sit outside [09:30, 16:00), so a plain wall-clock mask
    would blank every session of a daily frame rather than raise anywhere.
    There the mask is all True and one row is its own session's aggregate.
    """
    keys = _session_keys(bars.index)
    inside = bars[col].where(rth_mask(bars.index))
    prev = inside.groupby(keys).agg(agg).shift(1)
    return pd.Series(prev.loc[keys].to_numpy(), index=bars.index)


def prev_session_high(ctx: MarketContext, bars: pd.DataFrame) -> pd.Series:
    return _prev_session(bars, "high", "max")


def prev_session_low(ctx: MarketContext, bars: pd.DataFrame) -> pd.Series:
    return _prev_session(bars, "low", "min")


def prev_session_close(ctx: MarketContext, bars: pd.DataFrame) -> pd.Series:
    return _prev_session(bars, "close", "last")


def gap_pct(ctx: MarketContext, bars: pd.DataFrame) -> pd.Series:
    """This session's REGULAR open against the prior session's regular close.

    Both sides used to read the wrong print, and they compounded. The open was
    the first bar of the New York date, which on a cache that is not RTH-only
    is an 08:00 or 08:30 pre-market trade; the close came from
    prev_session_close, which was the last post-market print of the day before.
    So an overnight gap was measured from 19:45 to 08:00, a span that already
    contains most of the move the term is meant to name, and `gap_pct > 2` read
    a fraction of the gap a chart shows. It is now 09:30 against 15:45, which
    is the number a gap play means. The close half arrives through the
    corrected prev_session_close rather than being applied a second time here,
    so there is one definition of a session close and not two.

    NaN until the session's own first regular bar has printed, and that is the
    causal half rather than a tidy-up. `transform("first")` broadcasts one
    value over the WHOLE group, backwards included, so without the `started`
    mask an 08:00 bar would read the 09:30 open an hour and a half before it
    existed. That is lookahead on exactly the bars a gap play is most tempted
    to trade, and it would fire in replay and never live. A condition over NaN
    reads False, so the pre-market simply cannot act on a gap that has not
    opened yet.

    No zero guard on the denominator, deliberately: a zero prior close is not a
    price an equity prints, and the Pine lowering records what the two engines
    would answer if one ever did.
    """
    keys = _session_keys(bars.index)
    rth = rth_mask(bars.index)
    # Whether this session's first regular bar has printed yet: a cumulative
    # any over the mask, which is False through the pre-market, True from the
    # 09:30 bar onward, and True on every bar of a session frame, where the one
    # row IS its own regular session.
    started = rth.groupby(keys).cummax()
    opens = bars["open"].where(rth).groupby(keys).transform("first")
    prev_close = prev_session_close(ctx, bars)
    return (100 * (opens.where(started) - prev_close) / prev_close).rename(None)


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
    """Weekday of each bar's session, 0 = Monday.

    Which clock the weekday is read on is the FRAME's to decide, never the
    individual label's. An intraday bar belongs to the New York calendar day it
    falls in, which is the session _session_keys groups every other
    session-scoped primitive on. A session bar is one whole session in one row,
    and its label's
    UTC calendar date IS that session's date, where converting to New York
    would land on the prior evening and shift every weekday back by one.

    Branching on the label's own clock instead is the bug this must not go back
    to. The caches are not RTH-only (see data/schema.SESSION_OPEN), so a 19:00
    EST post-market bar carries exactly the 00:00 UTC label a resampled daily
    bar carries, and reading its UTC date answered Tuesday for a Monday
    evening: a divergence that appeared on one bar of a session and nowhere
    else, which is the shape of thing a spec author never catches.
    """
    idx = bars.index
    dow = (idx.dayofweek if _is_session_frame(idx)
           else idx.tz_convert(EXCHANGE_TZ).dayofweek)
    return pd.Series(np.asarray(dow, dtype="float64"), index=idx)


def minutes_into_session(ctx: MarketContext, bars: pd.DataFrame) -> pd.Series:
    """Minutes elapsed since the 09:30 bell of each bar's own session.

    Measured from data/schema.session_opens and not from the first bar of the
    calendar date. The caches are not RTH-only, so that first bar is ordinarily
    an 08:00 or 08:30 pre-market print, and counting from it ran every reading
    an hour and a half fast: a spec writing `minutes_into_session < N` named one
    wall-clock time and got another an hour and a half earlier.

    A bar BEFORE the bell answers NaN, and that is the deliberate half. On the
    right anchor a pre-market bar is NEGATIVE, and a negative silently passes
    every `< N` gate ever written over this term, so a play meant for the first
    part of the session would start firing at 08:00, before any of its premises
    exist. An engine condition reads NaN as False, so NaN fails closed and no
    gated play can fire pre-market, in a live scan or in replay. One `.where()`
    here does what a lower bound would have cost in every spec that reads this,
    and it cannot be forgotten in the next one.

    A bar AFTER the close keeps counting rather than going NaN, and that is not
    an oversight to tidy later. A session play exits on a `>= N` that is
    satisfied INSIDE the session, so the exit is already taken by the time a
    post-market bar arrives; and an entry gate is a `< N` that a value past the
    close already fails. Blanking the post-market would change neither, and
    would take away the one reading that says how far past the bell a bar sits.

    On a session frame the answer is 0 on every bar: there one row IS the whole
    session, so it opens at its own open and no other number is available.
    vocabulary.py documents that reading and spec.py refuses a 1d DRIVING frame
    over it (driving_frame_intraday), while session_scoped keeps a foreign `tf`
    out, so nothing shipped reaches this branch. It routes through
    data/schema._is_session_frame anyway, which is the same rule rth_mask and
    day_of_week read, so a frame is read one way house-wide instead of three.
    """
    idx = bars.index
    if _is_session_frame(idx):
        return pd.Series(0.0, index=idx, dtype="float64")
    elapsed = pd.Series((idx - session_opens(idx)).total_seconds() / 60.0,
                        index=idx, dtype="float64")
    return elapsed.where(elapsed >= 0)


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
