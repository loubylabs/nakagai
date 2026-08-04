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
from types import SimpleNamespace

from nakagai.data.schema import DEFAULT_TIMEFRAMES
from nakagai.strategies.ict.primitives import atr as window_atr
from nakagai.strategies.rules import compile_pine, lower_pine, primitives as prim
from nakagai.strategies.rules.frame_eval import FrameEval
from nakagai.strategies.rules.pine.lowerings import HELPERS
from nakagai.strategies.util import first_bar_of_session, fresh_bar
from tests.pine_interpreter import as_series, run_helper, run_program

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


def test_the_session_open_bar_is_the_one_the_engine_gates_a_daily_play_on():
    # util.first_bar_of_session, which is deliberately NOT "the calendar day
    # changed". This frame opens at 08:00 Eastern, and the engine's gate waits
    # for 09:30 because the caches are not RTH-only: firing on the first bar of
    # the DATE would put a daily play's signal on a pre-market print that no
    # live scanner ever visits.
    engine = np.array([first_bar_of_session(
        SimpleNamespace(driving_bars=BARS.iloc[:i + 1]))
        for i in range(len(BARS))])
    pine = np.array([bool(value) for value
                     in run_helper(SOURCES, "nk_session_open_bar", BARS)])
    assert (pine == engine).all(), \
        f"rows {np.flatnonzero(pine != engine)[:5].tolist()}"
    assert int(engine.sum()) == 7, "one gate per session in the frame"
    fired = BARS.index[engine].tz_convert("America/New_York")
    assert set(fired.hour) == {10} and not set(fired.minute) - {0}
    # Named outright: the 08:00 bar opens each date and must never be the gate.
    assert not engine[BARS.index.tz_convert("America/New_York").hour == 8].any()


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


# -- the decision bar, against the engine's own two gates -------------------
#
# This is the one comparison the whole cross-timeframe lowering rests on. A
# play off the driving frame is a request, a gate and a latch, and none of them
# means anything alone: the request carries no offset, which is sound only
# because the gate reads it where the requested bar closes, and the latch is
# what keeps every other bar off that value. Whether the three together fire on
# the bars the ENGINE fires on is not visible in any line of the artifact, so
# the artifact is executed and the bar sets are compared.
#
# The engine's answer has two halves and both are taken from the engine itself:
# frame_eval.driving_group says which spec-timeframe bar a driving bar reads,
# and RuleStrategy._fresh (util.fresh_bar, util.first_bar_of_session) says
# whether that driving bar may signal at all. Comparing against only the first
# would pass on a script that marked all four 15m bars of a 1h bar.

CHART_STEP = pd.Timedelta(minutes=15)
# The condition, deliberately built from sources alone: the interpreter models
# no `ta.*`, and a probe that needed one would be testing arithmetic rather
# than the alignment this file is about.
PROBE_GROUPS = {
    "long": {"all": [{"lhs": {"src": "close"}, "op": ">", "rhs": {"src": "open"}}]},
    "short": {"all": [{"lhs": {"src": "close"}, "op": "<", "rhs": {"src": "open"}}]},
}


def _probe_spec(timeframe: str) -> dict:
    return {"version": 2, "name": "probe", "timeframe": timeframe,
            **PROBE_GROUPS,
            # Percent geometry, so no ATR reaches the interpreter and the
            # decision is the gate and the condition with nothing else in it.
            "risk": {"stop": {"kind": "percent", "pct": 2.0},
                     "target": {"kind": "rr", "rr": 2.0}}}


def _stamps(days: int) -> pd.DatetimeIndex:
    """`days` New York sessions of 15m bars, 09:00 through 15:45 Eastern.

    Whole hours on purpose: the 1h bars derived from these start on the hour,
    so the last 15m bar of each hour is the one whose close IS the 1h close,
    which is the bar the whole defect turned on.
    """
    out = []
    for day in ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
                "2026-01-09", "2026-01-12", "2026-01-13", "2026-01-14")[:days]:
        start = pd.Timestamp(f"{day} 09:00", tz="America/New_York")
        out.extend(start + CHART_STEP * k for k in range(28))
    return pd.DatetimeIndex(out).tz_convert("UTC")


def _market(index: pd.DatetimeIndex, close: np.ndarray, step: np.ndarray):
    """One chart frame and the higher-timeframe frames a request of it reads."""
    chart = pd.DataFrame(
        {"open": close - step, "high": np.maximum(close, close - step) + 0.2,
         "low": np.minimum(close, close - step) - 0.2, "close": close,
         "volume": np.full(len(index), 1000.0)}, index=index)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum"}
    frames = {"15m": chart,
              **{tf: chart.resample(rule).agg(agg).dropna()
                 for tf, rule in (("1h", "1h"), ("4h", "4h"), ("1d", "1D"))}}
    requested = {"60": (frames["1h"], pd.Timedelta(hours=1)),
                 "D": (frames["1d"], pd.Timedelta(days=1))}
    return chart, frames, requested


def _trending():
    index = _stamps(6)
    rng = np.random.default_rng(11)
    step = rng.normal(0, 0.5, len(index))
    # A drift that alternates by SESSION, big enough to decide a day's
    # direction and small enough to leave the hours inside it mixed. Without it
    # a daily probe can run six sessions the same way and compare a short side
    # that never fired, which would agree with anything.
    step += np.where((np.arange(len(index)) // 28) % 2 == 0, 0.25, -0.25)
    return _market(index, 100 + np.cumsum(step), step)


def _oscillating():
    """A price that keeps recrossing the previous day's close.

    A cross against a foreign operand is the sparsest thing this file measures:
    the reference moves once a day, so a trending walk supplies two crossings in
    six sessions and a mutation can miss both and look harmless. Noise around a
    level supplies a dozen, which is what makes the net able to fail.
    """
    index = _stamps(6)
    rng = np.random.default_rng(5)
    close = 100 + rng.normal(0, 1.2, len(index))
    return _market(index, close, rng.normal(0, 0.4, len(index)))


CHART, FRAMES, REQUESTED = _trending()
CROSS_CHART, CROSS_FRAMES, CROSS_REQUESTED = _oscillating()


def _body(source: str) -> list[str]:
    """The artifact's inputs, helpers, calculations and decisions.

    Everything above is the header and the chart guard, and everything below
    places markers or orders. What is left is the whole of what decides.
    """
    start = source.index("// --- Inputs ---")
    return source[start:source.index("// --- Markers ---")].splitlines()


def _engine_decisions(spec: dict, side: str, chart=None, frames=None) -> np.ndarray:
    """The driving bars RuleStrategy would signal `side` on, from the engine."""
    chart = CHART if chart is None else chart
    frames = FRAMES if frames is None else frames
    timeframe = spec["timeframe"]
    lifted = FrameEval(frames, DEFAULT_TIMEFRAMES).driving_group(
        spec[side], timeframe).to_numpy()
    out = np.zeros(len(chart), dtype=bool)
    for i in range(len(chart)):
        now = chart.index[i] + CHART_STEP
        visible = frames[timeframe][frames[timeframe].index
                                    + DEFAULT_TIMEFRAMES.deltas.get(
                                        timeframe, pd.Timedelta(0)) <= now]
        ctx = SimpleNamespace(bars={timeframe: visible}, now=now,
                              tfs=DEFAULT_TIMEFRAMES,
                              driving_bars=chart.iloc[:i + 1])
        fresh = (first_bar_of_session(ctx)
                 if timeframe in DEFAULT_TIMEFRAMES.session_aligned
                 else fresh_bar(ctx, timeframe))
        out[i] = bool(lifted[i]) and fresh
    return out


@pytest.mark.parametrize("timeframe", ["1h", "1d"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_the_decision_lands_on_the_bars_the_engine_decides_on(timeframe, side):
    spec = _probe_spec(timeframe)
    source = compile_pine(spec).indicator
    rows = run_program({h.id: h.source for h in HELPERS.values()},
                       _body(source), CHART, CHART_STEP, REQUESTED)
    pine = np.array([bool(row[f"nk_{side}_decision"]) for row in rows])
    engine = _engine_decisions(spec, side)
    wrong = np.flatnonzero(pine != engine)
    assert not len(wrong), (
        f"{timeframe} {side}: rows {wrong[:8].tolist()} at "
        f"{[str(t) for t in CHART.index[wrong[:8]]]}, Pine "
        f"{pine[wrong[:8]].tolist()} against engine {engine[wrong[:8]].tolist()}")
    # A run that decided nothing would agree with anything, and one that
    # decided more often than its own timeframe has bars would mean the gate
    # had gone.
    assert 2 <= int(engine.sum()) <= len(FRAMES[timeframe])


@pytest.mark.parametrize("timeframe", ["1h", "1d"])
def test_the_decision_fires_once_per_bar_of_the_play_s_own_timeframe(timeframe):
    # The freshness gate, on its own terms. Without it the lifted condition
    # stands for every chart bar underneath one spec-timeframe bar: four for a
    # 1h play, twenty-eight for a daily one.
    spec = _probe_spec(timeframe)
    rows = run_program({h.id: h.source for h in HELPERS.values()},
                       _body(compile_pine(spec).indicator), CHART, CHART_STEP,
                       REQUESTED)
    decided = [i for i, row in enumerate(rows)
               if row["nk_long_decision"] or row["nk_short_decision"]]
    assert decided
    per_bar = FRAMES[timeframe].index.searchsorted(CHART.index[decided],
                                                   side="right")
    assert len(set(per_bar)) == len(per_bar), (
        f"{timeframe}: two chart bars decided inside one {timeframe} bar")


def test_the_comparison_notices_the_defect_it_was_written_for():
    # The net, checked against itself. Dropping the gate is the shape of the
    # original bug, and reading the confirmed form instead of the gated one is
    # the shape of the fix I nearly shipped. Both must be caught here, or this
    # file is a claim rather than a check.
    spec = _probe_spec("1h")
    lines = _body(compile_pine(spec).indicator)
    engine = _engine_decisions(spec, "long")
    sources = {h.id: h.source for h in HELPERS.values()}
    for label, before, after in (
            ("the freshness gate is dropped",
             "nk_long_decision = nk_visible_60 and nk_long_entry and",
             "nk_long_decision = nk_long_entry and"),
            ("the request is read one bar back instead of on its close",
             "[nk_long_entry_native, nk_short_entry_native]",
             "[nk_long_entry_native[1], nk_short_entry_native[1]]")):
        mutated = [line.replace(before, after) for line in lines]
        assert mutated != lines, f"{label} no longer describes the artifact"
        rows = run_program(sources, mutated, CHART, CHART_STEP, REQUESTED)
        pine = np.array([bool(row["nk_long_decision"]) for row in rows])
        assert (pine != engine).any(), \
            f"{label} changed nothing this file can see, so it guards nothing"


# -- a play that reads two timeframes at once -------------------------------
#
# The shape of discount_pullback and mfi_bounce: a 1h play with a 1d operand.
# The engine composes 1d -> 1h -> 15m, and the artifact requests the 1d
# straight onto the chart, so the two routes have to agree at every bar the
# gate admits. They do because a gated chart bar closes at the same instant its
# 1h bar does, and both routes ask which daily bar had closed by then. That is
# an argument; these tests are the measurement.

MIXED_GROUPS = {
    "long": {"all": [{"lhs": {"src": "close"}, "op": ">",
                      "rhs": {"src": "open", "tf": "1d"}},
                     {"lhs": {"src": "close"}, "op": ">", "rhs": {"src": "open"}}]},
    "short": {"all": [{"lhs": {"src": "close"}, "op": "<",
                       "rhs": {"src": "open", "tf": "1d"}},
                      {"lhs": {"src": "close"}, "op": "<", "rhs": {"src": "open"}}]},
}
# The same, with the foreign operand INSIDE a cross, which is the case that
# needs the gate-cadence snapshots. No catalog play is shaped this way yet, so
# it is pinned here rather than left to be discovered.
CROSS_GROUPS = {
    "long": {"all": [{"lhs": {"src": "close"}, "op": "crosses_above",
                      "rhs": {"src": "close", "tf": "1d"}}]},
    "short": {"all": [{"lhs": {"src": "close"}, "op": "crosses_below",
                       "rhs": {"src": "close", "tf": "1d"}}]},
}


def _spec_with(groups: dict) -> dict:
    return {"version": 2, "name": "probe", "timeframe": "1h", **groups,
            "risk": {"stop": {"kind": "percent", "pct": 2.0},
                     "target": {"kind": "rr", "rr": 2.0}}}


@pytest.mark.parametrize("groups, market", [(MIXED_GROUPS, "trending"),
                                            (CROSS_GROUPS, "oscillating")],
                         ids=["a plain comparison", "a cross"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_a_foreign_operand_decides_where_the_engine_decides(groups, market,
                                                            side):
    spec = _spec_with(groups)
    chart, frames, requested = ((CHART, FRAMES, REQUESTED)
                                if market == "trending" else
                                (CROSS_CHART, CROSS_FRAMES, CROSS_REQUESTED))
    rows = run_program({h.id: h.source for h in HELPERS.values()},
                       _body(compile_pine(spec).indicator), chart, CHART_STEP,
                       requested)
    pine = np.array([bool(row[f"nk_{side}_decision"]) for row in rows])
    engine = _engine_decisions(spec, side, chart, frames)
    wrong = np.flatnonzero(pine != engine)
    assert not len(wrong), (
        f"{market} {side}: rows {wrong[:8].tolist()} at "
        f"{[str(t) for t in chart.index[wrong[:8]]]}, Pine "
        f"{pine[wrong[:8]].tolist()} against engine {engine[wrong[:8]].tolist()}")
    assert 2 <= int(engine.sum()) <= len(frames["1h"])


def test_a_chart_composed_cross_reads_the_previous_hour_not_the_previous_bar():
    # The net for the snapshots. Reading the chart's own history where the
    # engine reads the previous SPEC bar is the whole family of bug this file
    # exists to catch, so both wrong readings are applied and both must show.
    spec = _spec_with(CROSS_GROUPS)
    lines = _body(compile_pine(spec).indicator)
    engine = _engine_decisions(spec, "long", CROSS_CHART, CROSS_FRAMES)
    sources = {h.id: h.source for h in HELPERS.values()}
    assert any("_prior" in line for line in lines), \
        "the artifact no longer takes gate-cadence snapshots"
    for label, edits in (
            ("the cross compares chart bars rather than the play's own",
             [("nk_close_1_prior <=", "close[1] <="),
              ("nk_close_1_gated >", "close >")]),
            # The two snapshot lines swapped, so the pair samples this bar
            # twice. Ordering is the whole content of a carry-forward, and no
            # substring assertion anywhere can see it.
            ("the snapshot is taken after the latch it samples",
             [("    nk_close_1_prior := nk_close_1_gated", "    SWAP"),
              ("    nk_close_1_gated := nk_close_1",
               "    nk_close_1_prior := nk_close_1_gated"),
              ("    SWAP", "    nk_close_1_gated := nk_close_1")]),
            ("previous means the current value",
             [("nk_close_1_prior", "nk_close_1_gated"),
              ("nk_close_2_prior", "nk_close_2_gated")])):
        mutated = list(lines)
        for before, after in edits:
            mutated = [line.replace(before, after) for line in mutated]
        assert mutated != lines, f"{label} no longer describes the artifact"
        rows = run_program(sources, mutated, CROSS_CHART, CHART_STEP,
                           CROSS_REQUESTED)
        pine = np.array([bool(row["nk_long_decision"]) for row in rows])
        assert (pine != engine).any(), \
            f"{label} changed nothing this file can see, so it guards nothing"


# The shape that lost its freshness gate: every condition on a foreign
# timeframe, so the play requests nothing of its own, and a percent stop so no
# ATR distance requests one either. Nothing shipped is shaped this way, but
# lower_pine is a general compiler and "on 1h, when the daily close is above
# the daily open, with a 2 percent stop" reaches it directly.
ALL_FOREIGN = {
    "long": {"all": [{"lhs": {"src": "close", "tf": "1d"}, "op": ">",
                      "rhs": {"src": "open", "tf": "1d"}}]},
    "short": {"all": [{"lhs": {"src": "close", "tf": "1d"}, "op": "<",
                       "rhs": {"src": "open", "tf": "1d"}}]},
}


@pytest.mark.parametrize("side", ["long", "short"])
def test_a_play_that_requests_nothing_of_its_own_frame_is_still_gated(side):
    spec = _spec_with(ALL_FOREIGN)
    rows = run_program({h.id: h.source for h in HELPERS.values()},
                       _body(compile_pine(spec).indicator), CHART, CHART_STEP,
                       REQUESTED)
    pine = np.array([bool(row[f"nk_{side}_decision"]) for row in rows])
    engine = _engine_decisions(spec, side)
    wrong = np.flatnonzero(pine != engine)
    assert not len(wrong), (
        f"all-foreign {side}: rows {wrong[:8].tolist()} at "
        f"{[str(t) for t in CHART.index[wrong[:8]]]}, Pine "
        f"{pine[wrong[:8]].tolist()} against engine {engine[wrong[:8]].tolist()}")
    # Ungated, this fired on all four 15m bars of every hour the daily
    # condition held. The engine fires once per hour, so the count is the
    # defect's own signature and not only the bar set.
    assert 2 <= int(engine.sum()) <= len(FRAMES["1h"])
    assert int(pine.sum()) == int(engine.sum())


def test_the_gate_alone_pins_the_premise_it_rests_on():
    # Same shape as above, read for what the artifact SAYS rather than what it
    # decides. The gate is `time_close("60") == time_close`, which is
    # TradingView's hourly aggregation and is therefore anchored to the chart's
    # session exactly as a request.security of 1h would be. The premise used to
    # be written down only for timeframes something asked a value of, so this
    # play, whose only 1h fact is its gate, shipped an artifact that never named
    # the assumption its every signal rests on.
    program = lower_pine(_spec_with(ALL_FOREIGN))
    premise = [line for line in program.assumptions
               if "nobody has measured on a chart" in line]
    assert len(premise) == 1
    assert 'plot time("60")' in premise[0]
    # A 15m play has no such gate and must not carry the sentence.
    flat = dict(_spec_with(ALL_FOREIGN), timeframe="15m")
    assert not [line for line in lower_pine(flat).assumptions
                if "nobody has measured on a chart" in line]
