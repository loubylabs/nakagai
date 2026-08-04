"""Every primitive helper RUN, bar by bar, against the engine it translates.

The other primitive file asserts what a helper's source says. This one asserts
what it DOES, and the two catch different things. A causal lowering's whole
content is when a value becomes readable, which lives in the ORDER of a
helper's lines and in whether a variable survives the bar; neither is visible
to `"..." in source`. Dropping `var` from the swing's carried level, or rolling
the previous session's aggregate after the running one has been reset, changes
what every bar reads and changes no substring any assertion here or there
matches.

So the helper's own emitted text is executed (tests/pine_interpreter.py) over
one synthetic frame, and compared with the Python primitive's output over the
same frame, row for row. Nothing restates an algorithm: the comparison is the
emitted Pine against the shipped Python, so it moves when either does. A
hand-written Python model of what a helper is believed to do would agree with
itself forever and catch nothing.

The last test is the net checking itself. It applies the two mutations by hand,
in memory, and asserts the comparison notices: a net nobody has seen fail is a
claim rather than a check.
"""

import numpy as np
import pandas as pd
import pytest

from nakagai.strategies.ict.primitives import atr as window_atr
from nakagai.strategies.rules import primitives as prim
from nakagai.strategies.rules.pine.lowerings import HELPERS
from tests.pine_interpreter import as_series, run_helper

SOURCES = {helper_id: helper.source for helper_id, helper in HELPERS.items()}
# Short enough that an end-anchored scan is exercised over ordinary history,
# long enough that the ATR inside it is the full 14 true ranges.
LOOKBACK = 20


def _bars() -> pd.DataFrame:
    """Seven New York sessions of hourly bars, 08:00 through 20:00 Eastern.

    Extended hours on purpose. The caches are not RTH-only, so a session runs
    from a pre-market bar to a post-market one, and the 19:00 Eastern bar of
    each session carries a 00:00 UTC label: exactly the bar that used to take
    day_of_week's daily branch and read the next day's weekday. Both engines
    have to group it with the session it belongs to.

    The prices are a seeded walk with three displacement runs pushed into it,
    which is what leaves swings for the pivots to confirm, three-candle
    imbalances for the gap scan, and bodies large enough for an order block.
    """
    stamps = []
    for day in ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
                "2026-01-09", "2026-01-12", "2026-01-13"):
        start = pd.Timestamp(f"{day} 08:00", tz="America/New_York")
        stamps.extend(start + pd.Timedelta(hours=hour) for hour in range(13))
    index = pd.DatetimeIndex(stamps).tz_convert("UTC")
    rng = np.random.default_rng(7)
    steps = rng.normal(0, 0.6, len(index))
    steps[[5, 6, 7]] += [2.5, 3.0, 2.0]
    steps[[30, 31, 32]] -= [3.0, 3.5, 2.0]
    steps[[55, 56, 57]] += [3.0, 2.5, 3.0]
    close = 100 + np.cumsum(steps)
    return pd.DataFrame(
        {"open": close - steps * 0.7,
         "high": close + np.abs(rng.normal(0, 0.3, len(index))) + 0.1,
         "low": close - np.abs(rng.normal(0, 0.3, len(index))) - 0.1,
         "close": close,
         "volume": np.abs(rng.normal(1000, 200, len(index))) + 100},
        index=index)


BARS = _bars()
# Rows whose window is genuinely `LOOKBACK` bars deep. Before that the engine
# scans the short frame it has and the helper reads the buffer TradingView
# gives it, which the program already carries as a standing assumption about
# the first bars of a chart. Everything after is compared exactly.
FULL_WINDOW = np.arange(len(BARS)) >= LOOKBACK - 1


def _pine(entry: str, *args) -> pd.Series:
    return as_series(run_helper(SOURCES, entry, BARS, args), BARS)


def _pine_pair(entry: str, *args) -> tuple[pd.Series, pd.Series]:
    rows = run_helper(SOURCES, entry, BARS, args)
    return (as_series([row[0] for row in rows], BARS),
            as_series([row[1] for row in rows], BARS))


def _end_anchored(fn, **args) -> pd.Series:
    """What the engine answers at each row: the scalar over bars[:i + 1]."""
    return pd.Series([float(fn(None, BARS.iloc[:i + 1], **args))
                      for i in range(len(BARS))], index=BARS.index)


def _wrong(pine: pd.Series, engine: pd.Series, where=None):
    """The rows where the two disagree, with both readings, na counted equal."""
    got, want = pine.to_numpy(), np.asarray(engine, dtype="float64")
    agree = (np.isclose(got, want, rtol=1e-9, atol=1e-9)
             | (np.isnan(got) & np.isnan(want)))
    mask = np.ones(len(got), dtype=bool) if where is None else where
    return np.flatnonzero(~agree & mask), got, want


def _assert_same(pine: pd.Series, engine: pd.Series, where=None) -> None:
    wrong, got, want = _wrong(pine, engine, where)
    assert not len(wrong), (
        f"rows {wrong[:5].tolist()}: Pine {got[wrong[:5]]} against engine "
        f"{want[wrong[:5]]}")
    # A run where everything is na would agree with anything, so say outright
    # that the frame exercised the helper.
    mask = np.ones(len(got), dtype=bool) if where is None else where
    assert np.count_nonzero(~np.isnan(want[mask])) > 5


# -- the session state machines --------------------------------------------


@pytest.mark.parametrize("minutes", [60, 120, 240])
def test_the_opening_range_becomes_readable_on_the_bar_the_engine_says(minutes):
    _assert_same(_pine("nk_opening_range_high", minutes),
                 prim.opening_range_high(None, BARS, minutes=minutes))
    _assert_same(_pine("nk_opening_range_low", minutes),
                 prim.opening_range_low(None, BARS, minutes=minutes))


def test_the_previous_session_levels_are_the_engine_s_on_every_bar():
    # The roll and the running aggregate are two lines whose ORDER is the whole
    # rule: swapped, every bar reads its own session's running high, which is a
    # value the engine does not expose until the session is over.
    _assert_same(_pine("nk_prev_session_high"), prim.prev_session_high(None, BARS))
    _assert_same(_pine("nk_prev_session_low"), prim.prev_session_low(None, BARS))
    _assert_same(_pine("nk_prev_session_close"),
                 prim.prev_session_close(None, BARS))


def test_the_gap_is_the_engine_s_gap():
    _assert_same(_pine("nk_gap_pct"), prim.gap_pct(None, BARS))


def test_minutes_into_session_counts_the_engine_s_minutes():
    _assert_same(_pine("nk_minutes_into_session"),
                 prim.minutes_into_session(None, BARS))


def test_the_weekday_agrees_on_the_post_market_bar_that_carries_a_utc_midnight():
    midnight = (BARS.index.hour == 0) & (BARS.index.minute == 0)
    assert midnight.sum() == 7, "the frame must hold the bar this rule turns on"
    engine = prim.day_of_week(None, BARS)
    _assert_same(_pine("nk_day_of_week"), engine)
    # Named outright, because it is the one bar where reading the label's own
    # clock instead of the frame's cadence answered the next day's weekday.
    assert set(engine[midnight]) <= {0.0, 1.0, 2.0, 3.0, 4.0}
    assert (engine[midnight].to_numpy()
            == engine.shift(1)[midnight].to_numpy()).all()


@pytest.mark.parametrize("sessions", [3, 5])
def test_relative_volume_is_the_engine_s_same_clock_median(sessions):
    _assert_same(_pine("nk_rvol", sessions),
                 prim.rvol(None, BARS, sessions=sessions))


# -- structure -------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 5])
def test_a_swing_carries_the_engine_s_last_confirmed_level(k):
    # The level is stamped on the confirming bar and carried forward by `var`.
    # Without the carry it reads na on every bar between confirmations, which
    # is the opposite of what the engine's ffill does.
    _assert_same(_pine("nk_swing_high", k), prim.swing_high(None, BARS, k=k))
    _assert_same(_pine("nk_swing_low", k), prim.swing_low(None, BARS, k=k))


@pytest.mark.parametrize("direction", ["long", "short"])
def test_the_leg_retrace_sits_where_the_engine_puts_it(direction):
    _assert_same(_pine("nk_leg_retrace", 3, direction == "long"),
                 prim.leg_retrace(None, BARS, direction=direction, k=3))


def test_bars_since_counts_the_engine_s_bars():
    mask = (BARS["close"] > BARS["open"]).to_numpy()
    engine = prim.bars_since(None, BARS, cond={},
                             eval_fn=lambda _cond, _bars:
                             pd.Series(mask, index=BARS.index))
    _assert_same(_pine("nk_bars_since", mask), engine)


# -- the end-anchored scans ------------------------------------------------


def test_the_window_atr_is_the_engine_s_mean_true_range():
    engine = pd.Series([window_atr(BARS.iloc[max(0, i - LOOKBACK + 1):i + 1])
                        for i in range(len(BARS))], index=BARS.index)
    _assert_same(_pine("nk_window_atr", LOOKBACK), engine, where=FULL_WINDOW)


@pytest.mark.parametrize("direction", ["long", "short"])
@pytest.mark.parametrize("state", ["open", "inverted"])
def test_the_nearest_gap_is_the_engine_s_gap(direction, state):
    # Both boundaries, so a scan that identified a different gap and happened
    # to agree on one edge is still caught.
    want_long = (direction == "long") == (state == "open")
    top, bottom = _pine_pair("nk_fvg_nearest", 0.25, LOOKBACK, want_long,
                             state == "inverted")
    for pine, field in ((top, "top"), (bottom, "bottom")):
        _assert_same(pine, _end_anchored(
            prim.fvg_nearest, direction=direction, field=field, state=state,
            min_size_atr=0.25, lookback=LOOKBACK), where=FULL_WINDOW)


@pytest.mark.parametrize("direction", ["long", "short"])
def test_the_order_block_is_the_engine_s_block(direction):
    top, bottom = _pine_pair("nk_order_block", 1.5, LOOKBACK,
                             direction == "long")
    for pine, field in ((top, "top"), (bottom, "bottom")):
        _assert_same(pine, _end_anchored(
            prim.order_block, direction=direction, field=field, body_atr=1.5,
            lookback=LOOKBACK), where=FULL_WINDOW)


# -- the net, checked against itself ---------------------------------------

# Each row is a one-line edit that leaves every substring assertion in
# tests/test_pine_primitives.py true and changes what the helper answers.
MUTATIONS = [
    ("the swing's level stops carrying forward", "nk_swing_high",
     "var float level = na", "float level = na",
     lambda: (_pine("nk_swing_high", 3), prim.swing_high(None, BARS, k=3))),
    ("the session rolls after the running high is reset",
     "nk_prev_session_high",
     "        previous := current\n        current := high",
     "        current := high\n        previous := current",
     lambda: (_pine("nk_prev_session_high"),
              prim.prev_session_high(None, BARS))),
]


@pytest.mark.parametrize("label, helper_id, before, after, compare", MUTATIONS,
                         ids=[row[0] for row in MUTATIONS])
def test_the_net_fails_when_the_causal_rule_is_broken(label, helper_id, before,
                                                      after, compare):
    original = SOURCES[helper_id]
    assert before in original, "the mutation no longer describes the helper"
    SOURCES[helper_id] = original.replace(before, after)
    try:
        pine, engine = compare()
    finally:
        SOURCES[helper_id] = original
    wrong, _got, _want = _wrong(pine, engine)
    assert len(wrong), \
        f"{label} changed nothing this file can see, so it guards nothing"
