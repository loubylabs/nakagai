"""FrameEval: whole-frame node values that agree with prefix evaluation."""

import numpy as np
import pandas as pd
import pytest

from nakagai.data.cache import MemoryBars
from nakagai.data.schema import DEFAULT_TIMEFRAMES as TFS
from nakagai.engine.context import build_context, closed_before
from nakagai.strategies.indicators import crossed_above
from nakagai.strategies.rules.frame_eval import FrameEval
from nakagai.strategies.rules.vocabulary import Term, core_vocabulary
from tests.whole_frame_oracle import prefix_value

NODES = [
    {"src": "close"},
    {"ind": "sma", "n": 20},
    {"ind": "ema", "n": 50},
    {"ind": "rsi", "n": 14},
    {"ind": "atr", "n": 14},
    {"ind": "vwap"},
    {"ind": "bb", "n": 20, "k": 2.0, "field": "upper"},
    {"ind": "macd", "fast": 12, "slow": 26, "signal": 9, "field": "hist"},
    {"prim": "opening_range_high", "minutes": 30},
    {"prim": "minutes_into_session"},
    {"prim": "prev_session_high"},
    {"prim": "swing_high", "k": 3},
    {"op": "-", "args": [{"src": "close"}, {"ind": "sma", "n": 10}]},
]


def _frames(n=400):
    rng = np.random.default_rng(11)
    idx = pd.date_range("2026-01-05 14:30", periods=n, freq="15min", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.4, n))
    b15 = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                        "close": close, "volume": 1000.0}, index=idx)
    b1h = b15.resample("1h").agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "volume": "sum"}).dropna()
    b1d = b15.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "volume": "sum"}).dropna()
    return {"15m": b15, "1h": b1h, "1d": b1d}


@pytest.mark.parametrize("node", NODES, ids=lambda n: str(n))
def test_whole_frame_row_equals_prefix_last_row(node):
    frames = _frames()
    fe = FrameEval(frames, TFS)
    whole = fe.series(node, "15m")
    driving = frames["15m"]
    for i in (120, 200, 311, len(driving) - 1):
        now = driving.index[i] + TFS.step
        want = prefix_value(node, frames, "15m", now, TFS)
        got = float(whole.iloc[i]) if isinstance(whole, pd.Series) else float(whole)
        assert (pd.isna(got) and pd.isna(want)) or got == want, f"row {i}"


def test_cross_timeframe_node_never_sees_an_unclosed_bar():
    frames = _frames()
    fe = FrameEval(frames, TFS)
    node = {"src": "close", "tf": "1h"}
    got = fe.series(node, "15m")
    driving, hourly = frames["15m"], frames["1h"]
    for i in range(80, len(driving)):
        now = driving.index[i] + TFS.step
        visible = hourly[hourly.index + TFS.deltas["1h"] <= now]
        want = visible["close"].iloc[-1] if len(visible) else float("nan")
        assert (pd.isna(got.iloc[i]) and pd.isna(want)) or got.iloc[i] == want


def test_nothing_visible_yet_is_nan_not_the_last_bar():
    """The rows before the first higher bar has closed.

    The visibility map counts closed bars, so an empty history is 0, and the
    position it converts to is -1. Read as an index that is the LAST row, which
    would hand the opening rows of a replay a value from the end of history.
    That is the worst-shaped lookahead bug there is: silent, and profitable.
    """
    frames = _frames()
    fe = FrameEval(frames, TFS)
    got = fe.series({"src": "close", "tf": "1d"}, "15m")
    driving = frames["15m"]
    blind = [i for i in range(len(driving))
             if not len(closed_before(frames["1d"], "1d",
                                      driving.index[i] + TFS.step, TFS))]
    assert blind, "fixture must contain rows with no closed daily bar yet"
    assert got.iloc[blind].isna().all()
    assert not (got.iloc[blind] == frames["1d"]["close"].iloc[-1]).any()
    seeing = [i for i in range(len(driving)) if i not in set(blind)]
    assert got.iloc[seeing].notna().all(), "the visible rows must still carry a value"


@pytest.mark.parametrize("node", [
    {"prim": "fvg_nearest", "direction": "long", "field": "top"},
    {"prim": "order_block", "direction": "long", "field": "top"},
], ids=lambda n: n["prim"])
def test_end_anchored_primitive_matches_prefix_over_its_span(node):
    """End-anchored primitives read the tail of the frame they are handed, so
    they are the one node kind a whole-frame pass may not broadcast. Inside the
    span they must equal the prefix answer row for row; outside it, on BOTH
    sides, they are NaN rather than a value carried from somewhere else in
    history. The rows before the span matter as much as the rows after: each row
    is computed on its own full prefix, so the span needs no warm-up margin, and
    a margin would only cost calls no reader can reach."""
    frames = _frames()
    fe = FrameEval(frames, TFS)
    fe.set_span("15m", 300, 340)
    got = fe.series(node, "15m")
    for i in range(300, 340):
        now = frames["15m"].index[i] + TFS.step
        want = prefix_value(node, frames, "15m", now, TFS)
        assert (pd.isna(got.iloc[i]) and pd.isna(want)) or got.iloc[i] == want, f"row {i}"
    assert got.iloc[:300].isna().all()
    assert got.iloc[340:].isna().all()


def test_a_point_in_time_context_walks_one_row_not_the_whole_frame(monkeypatch):
    """build_context's non-replay branch has to declare its span.

    The scanner and the screener build a context per bar against a raw cache,
    with no replay to bound anything. build_context has already cut the frames
    at `now`, so the only row those callers can read is the last one. An
    undeclared span defaults to the whole frame, and the end-anchored branch
    then calls the primitive once per row of history to produce values nobody
    reads. Measured on the real three-year 15m SPY cache that took one spec, one
    symbol, one bar from 0.001s to 11.4s; three end-anchored specs sit in the
    scan registry, on a 15-minute cron. Every fixture in this suite is a few
    hundred rows, where the cost is invisible, so assert the row COUNT.
    """
    frames = _frames(900)
    cache = MemoryBars({("SPY", tf): f for tf, f in frames.items()})
    now = frames["15m"].index[-1] + TFS.step
    node = {"prim": "fvg_nearest", "direction": "long", "field": "top"}
    want = prefix_value(node, frames, "15m", now, TFS)

    calls = []
    base = core_vocabulary()
    real = base.primitives["fvg_nearest"]
    terms = [term for term in base.all_terms() if term.name != "fvg_nearest"]
    terms.append(Term(real.name, real.kind, real.args, real.defaults,
                      lambda *a, **k: (calls.append(1), real.fn(*a, **k))[1],
                      doc=real.doc, end_anchored=real.end_anchored,
                      session_scoped=real.session_scoped,
                      driving_frame_intraday=real.driving_frame_intraday,
                      pine=real.pine))
    vocabulary = type(base)(
        {term.name: term for term in terms if term.kind != "primitive"},
        {term.name: term for term in terms if term.kind == "primitive"})
    ctx = build_context(cache, "SPY", now, vocabulary=vocabulary)
    got = float(ctx.fe.series(node, "15m").iloc[-1])

    assert len(calls) == 1, (
        f"walked {len(calls)} rows of a {len(frames['15m'])}-row frame; "
        "a point-in-time context can only read the last one")
    assert (pd.isna(got) and pd.isna(want)) or got == want


def test_series_is_memoized_per_node():
    fe = FrameEval(_frames(), TFS)
    a = fe.series({"ind": "sma", "n": 20}, "15m")
    b = fe.series({"ind": "sma", "n": 20}, "15m")
    assert a is b


def test_moving_the_span_after_a_node_is_cached_is_refused():
    """The span is a correctness input the memo key does not carry.

    Every end_anchored node's value depends on the span, and the node cache is
    keyed on (tf, node) alone. Reusing one FrameEval across walk-forward
    windows is the obvious next optimization and it would silently serve window
    one's rows for window two. Latent today only because each FrameEval is
    built for a single span.
    """
    frames = _frames()
    fe = FrameEval(frames, TFS)
    fe.set_span("15m", 300, 340)
    fe.series({"prim": "fvg_nearest", "direction": "long", "field": "top"}, "15m")
    with pytest.raises(ValueError, match="span moved"):
        fe.set_span("15m", 100, 200)


def test_declaring_a_span_after_the_default_was_used_is_refused():
    """The whole-frame default is a span too, and a node computed under it is
    just as stale once a real one is declared."""
    fe = FrameEval(_frames(), TFS)
    fe.series({"ind": "sma", "n": 20}, "15m")
    with pytest.raises(ValueError, match="span moved"):
        fe.set_span("15m", 100, 200)


def test_restating_the_same_span_is_allowed():
    """Only a MOVE is a lie about the cache. Restating what is already in force
    changes nothing, and refusing it would make set_span order-sensitive for no
    reason."""
    frames = _frames()
    fe = FrameEval(frames, TFS)
    fe.set_span("15m", 300, 340)
    fe.series({"ind": "sma", "n": 20}, "15m")
    fe.set_span("15m", 300, 340)
    fe.set_span("1h", 0, len(frames["1h"]))   # the default, said out loud
    assert fe._span("15m") == (300, 340)


def test_memo_key_separates_evaluation_timeframes():
    """The same node at two timeframes is two different series.

    _eval evaluates `of` sub-nodes at src_tf, so one spec can ask for
    {'src': 'close'} on 15m and again on 1h within a single replay. A memo key
    that keyed on the node alone would hand the second caller the first
    caller's series, indexed on the wrong timeframe, and every later read would
    be off by whatever the two indexes disagree about.
    """
    frames = _frames()
    fe = FrameEval(frames, TFS)
    node = {"src": "close"}
    at15, at1h = fe.series(node, "15m"), fe.series(node, "1h")
    assert at15 is not at1h
    assert not at15.index.equals(at1h.index)
    assert at15.index.equals(frames["15m"].index)
    assert at1h.index.equals(frames["1h"].index)


def _closes(values) -> dict:
    idx = pd.date_range("2026-01-05 14:30", periods=len(values), freq="15min",
                        tz="UTC")
    c = pd.Series(values, index=idx, dtype="float64")
    return {"15m": pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5,
                                 "close": c, "volume": 1000.0}, index=idx)}


def test_cross_reads_the_bar_before_never_the_bar_after():
    """crosses_above compares row i against row i-1, and row i-1 only.

    The shift direction is the whole causality of the operator. A .shift(-1)
    here still produces plausible-looking crosses, just ones that consult the
    NEXT bar, which is lookahead that no metric would flag.
    """
    frames = _closes([1.0, 2.0, 4.0, 5.0, 2.0, 6.0])
    fe = FrameEval(frames, TFS)
    cond = {"lhs": {"src": "close"}, "op": "crosses_above", "rhs": 3.0}
    got = fe.condition_series(cond, "15m")
    assert got.dtype == bool
    assert list(got) == [False, False, True, False, False, True]

    close = frames["15m"]["close"]
    for i in range(len(close)):
        assert bool(got.iloc[i]) == crossed_above(close.iloc[: i + 1], 3.0), \
            f"row {i} disagrees with the prefix-and-iloc[-2] semantics"


def test_cross_below_reads_the_bar_before():
    frames = _closes([6.0, 5.0, 2.0, 1.0, 4.0, 1.0])
    fe = FrameEval(frames, TFS)
    cond = {"lhs": {"src": "close"}, "op": "crosses_below", "rhs": 3.0}
    got = fe.condition_series(cond, "15m")
    assert list(got) == [False, False, True, False, False, True]


def test_cross_against_an_end_anchored_level_keeps_the_scalar_broadcast():
    """Both bars of the cross see the level known at the CURRENT bar.

    fvg_nearest returned one float per replayed bar, and crossed_above's scalar
    branch compared close[i-1] AND close[i] against that single level. Shifting
    the level series instead tests the earlier bar against the level as it stood
    a bar ago, which fires and suppresses different trades. This is the fixture
    that says so: `shifted` is what a plain .shift(1) on both operands would
    give, and it must not match.
    """
    frames = _frames()
    fe = FrameEval(frames, TFS)
    lo, hi = 200, 400
    fe.set_span("15m", lo, hi)
    node = {"prim": "fvg_nearest", "direction": "long", "field": "top"}
    got = fe.condition_series(
        {"lhs": {"src": "close"}, "op": "crosses_above", "rhs": node}, "15m")

    driving, level = frames["15m"], fe.series(node, "15m")
    close = driving["close"]
    want, shifted = [], []
    for i in range(lo, hi):
        now = driving.index[i] + TFS.step
        prefix = closed_before(driving, "15m", now, TFS)
        want.append(bool(crossed_above(
            prefix["close"], prefix_value(node, frames, "15m", now, TFS))))
        shifted.append(bool(close.iloc[i - 1] <= level.iloc[i - 1]
                            and close.iloc[i] > level.iloc[i]))
    assert [bool(v) for v in got.iloc[lo:hi]] == want
    assert any(want), "fixture must contain at least one firing cross"
    assert shifted != want, \
        "fixture must contain a row where shifting the level changes the answer"


def test_a_higher_timeframe_condition_is_false_until_its_bar_has_closed():
    """The lift onto the driving index keeps booleans boolean.

    strategy.py reads this series as bool(series.iloc[i]). Lifting through the
    float branch would leave the not-yet-visible rows NaN, and bool(nan) is
    True, so a condition on a bar that has not closed yet would read SATISFIED
    for every row of the opening blind window.
    """
    frames = _frames()
    frames["1h"] = frames["1h"].iloc[8:]
    fe = FrameEval(frames, TFS)
    group = {"all": [{"lhs": {"src": "close"}, "op": ">", "rhs": 0.0},
                     {"lhs": {"src": "high"}, "op": ">=", "rhs": {"src": "low"}}]}
    out = fe.driving_group(group, "1h")

    driving = frames["15m"]
    blind = [i for i in range(len(driving))
             if not len(closed_before(frames["1h"], "1h",
                                      driving.index[i] + TFS.step, TFS))]
    assert len(blind) > 8, "fixture must open with a real blind window"
    assert out.dtype == bool
    assert not out.iloc[blind].any()
    assert not any(bool(out.iloc[i]) for i in blind)
    seeing = sorted(set(range(len(driving))) - set(blind))
    assert out.iloc[seeing].all(), "the visible rows must still read satisfied"


def _hand_bars(closes, start, freq):
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5,
                         "close": c, "volume": 1000.0}, index=idx)


def test_a_daily_bar_is_invisible_within_its_own_session():
    """Jan 6 rows see Jan 5's daily close, never Jan 6's own.

    A daily bar is labeled at midnight UTC of the session it summarizes, so a
    plain label-ordered ffill hands every row of Jan 6 the close of the session
    it is still living through. That is the sharpest lookahead the grammar can
    express, and the one a session-aligned visibility rule exists to refuse.
    """
    frames = {"15m": _hand_bars([100.0] * 26, "2026-01-06 14:30", "15min"),
              "1h": _hand_bars([100.0] * 8, "2026-01-06 14:00", "1h"),
              "1d": _hand_bars([95.0, 96.0], "2026-01-05 00:00", "1D")}
    fe = FrameEval(frames, TFS)
    assert (fe.series({"src": "close", "tf": "1d"}, "15m") == 95.0).all()


def test_a_tf_qualified_primitive_evaluates_on_its_native_frame():
    """prev_session_high[1h] reads the 1h frame, then lands by visibility.

    The primitive has to see hourly bars to have an hourly answer; evaluating
    it against the driving frame and relabeling would give a 15m answer wearing
    an hourly name. The Jan 5 session's hourly high is 202.0, and every Jan 6
    driving row carries it once the hourly bars that establish it have closed.

    The Jan 5 bars are 15:00 and 16:00 UTC, which is 10:00 and 11:00 New York,
    and the hours are load-bearing rather than arbitrary. They used to be 13:00
    and 14:00, an 08:00 and an 09:00 pre-market print, so the 202.0 this
    asserts was a pre-market high wearing the name "the previous session's
    high": the fixture agreed with chrvsd/nakagai#276 rather than measuring
    anything. The Jan 6 bars stay in the pre-market on purpose, because a bar
    reads the previous session's level whatever time of day it prints; it is
    the AGGREGATE that is scoped to regular hours, not the reading.
    """
    b1h = pd.concat([_hand_bars([199.5, 201.5], "2026-01-05 15:00", "1h"),
                     _hand_bars([190.0, 191.0], "2026-01-06 13:00", "1h")])
    frames = {"15m": _hand_bars([100.0] * 8, "2026-01-06 14:30", "15min"),
              "1h": b1h,
              "1d": _hand_bars([95.0, 96.0], "2026-01-05 00:00", "1D")}
    fe = FrameEval(frames, TFS)
    out = fe.series({"prim": "prev_session_high", "tf": "1h"}, "15m")
    assert out.index.equals(frames["15m"].index)
    assert (out == 202.0).all()


def test_session_aligned_destination_timeframe_is_refused():
    """A daily bar's label carries no close time, so its visibility cutoff
    would have to be guessed. _positions refuses rather than guess."""
    fe = FrameEval(_frames(), TFS)
    with pytest.raises(ValueError, match="session-aligned"):
        fe.series({"src": "close", "tf": "15m"}, "1d")


KLEENE_ROWS = [
    ("all", [True, pd.NA], pd.NA),
    ("all", [False, pd.NA], False),
    ("all", [True, False], False),
    ("all", [pd.NA, pd.NA], pd.NA),
    ("any", [False, pd.NA], pd.NA),
    ("any", [True, pd.NA], True),
    ("any", [True, False], True),
    ("any", [pd.NA, pd.NA], pd.NA),
]


def _kleene_frames():
    """15m with one row, plus an EMPTY 1h frame. A reference to 1h is then
    unknown on every row, which is how pd.NA is manufactured for the table
    without a second code path."""
    return {**_closes([1.0]),
            "1h": pd.DataFrame({"open": [], "high": [], "low": [], "close": [],
                                "volume": []},
                               index=pd.DatetimeIndex([], tz="UTC"))}


def _kleene_group(key, operands):
    leaf = {True: {"lhs": {"src": "close"}, "op": ">", "rhs": -1.0},
            False: {"lhs": {"src": "close"}, "op": "<", "rhs": -1.0}}
    unknown = {"lhs": {"src": "close"}, "op": ">",
               "rhs": {"src": "close", "tf": "1h"}}
    return {key: [leaf[v] if v is True or v is False else unknown
                  for v in operands]}


@pytest.mark.parametrize("key,operands,want", KLEENE_ROWS,
                         ids=[f"{k}-{o}" for k, o, _ in KLEENE_ROWS])
def test_the_private_reducer_matches_the_eight_row_kleene_table(key, operands, want):
    """D8's table, asserted against the private reducer, which is where Kleene
    lives (N3-D4). Asserting it against group_series would be a different and
    wrong test: that method resolves unknown to False."""
    fe = FrameEval(_kleene_frames(), TFS)
    got = fe._group_reduce_na(_kleene_group(key, operands), "15m").iloc[0]
    if want is pd.NA:
        assert pd.isna(got), f"{key} {operands}: want NA, got {got!r}"
    else:
        assert not pd.isna(got) and bool(got) == want, \
            f"{key} {operands}: want {want}, got {got!r}"


def test_the_private_reducer_carries_the_nullable_dtype_not_plain_bool():
    """The representation itself (N3-D1). A reducer that resolved unknown
    early would still satisfy the two `all` rows above by accident on a
    single-row frame; this pins the dtype the table is read from."""
    fe = FrameEval(_kleene_frames(), TFS)
    out = fe._group_reduce_na(_kleene_group("all", [True, pd.NA]), "15m")
    assert out.dtype == pd.BooleanDtype()


def test_group_series_resolves_an_unknown_reducer_result_to_false_and_bool():
    """Acceptance item 4's public half, and item 8. The SAME group whose
    reducer result is NA reads False, dtype bool, at the boundary."""
    fe = FrameEval(_kleene_frames(), TFS)
    group = _kleene_group("all", [True, pd.NA])
    assert pd.isna(fe._group_reduce_na(group, "15m").iloc[0])
    out = fe.group_series(group, "15m")
    assert not bool(out.iloc[0])
    assert out.dtype == bool


def test_driving_group_stays_bool_over_an_unknown_group():
    """Acceptance item 8 for the other public reader. pd.NA reaching this
    return value raises TypeError in every one of its three call sites."""
    frames = _frames()
    frames["1h"] = frames["1h"].iloc[8:]
    fe = FrameEval(frames, TFS)
    group = {"all": [{"lhs": {"src": "close", "tf": "1h"}, "op": ">",
                      "rhs": -1.0}]}
    out = fe.driving_group(group, "15m")
    assert all(isinstance(bool(v), bool) for v in out)
    assert out.dtype == bool


def test_condition_series_resolves_an_unknown_operand_to_false_and_stays_bool():
    """The public leaf boundary (N3-D4): unchanged from before this node.

    The unknown rows are the point, so the fixture builds them: a 1h operand
    on a 15m condition is NaN until the first hourly bar has closed, which is
    NA one level down and False here.
    """
    frames = _frames()
    frames["1h"] = frames["1h"].iloc[8:]
    fe = FrameEval(frames, TFS)
    cond = {"lhs": {"src": "close", "tf": "1h"}, "op": ">", "rhs": -1.0}
    blind = [i for i, v in enumerate(fe.series(cond["lhs"], "15m").isna()) if v]
    assert len(blind) > 8, "fixture must open with a real blind window"

    got = fe.condition_series(cond, "15m")
    # bool() on the element is what strategy.py:81 and screen/runner.py:79 do,
    # and it is what raises rather than lies if pd.NA ever reaches here.
    assert bool(got.iloc[blind[0]]) is False, "an unknown operand must read not-fired"
    assert not got.iloc[blind].any()
    seeing = sorted(set(range(len(got))) - set(blind))
    assert got.iloc[seeing].all(), "the visible rows must still read satisfied"
    assert got.dtype == bool


def test_not_over_a_warming_indicator_does_not_fire_on_the_first_bars():
    """D8's headline case (acceptance item 6), with the bar index named."""
    idx = pd.date_range("2026-01-05 09:30", periods=60, freq="15min", tz="UTC")
    # Oscillating, so that once sma(20) is warm the condition is genuinely
    # false on some rows and the negation genuinely fires there. A monotone
    # ramp would leave the negation false everywhere after warmup too, and
    # this test would then pass on a series that is False for the wrong
    # reason from row 0 to the end.
    close = pd.Series(100 + np.sin(np.linspace(0, 6 * np.pi, 60)) * 2,
                      index=idx)
    bars = pd.DataFrame({"open": close, "high": close + 0.2, "low": close - 0.2,
                         "close": close, "volume": 1000.0}, index=idx)
    fe = FrameEval({"15m": bars}, TFS)
    sma = {"ind": "sma", "n": 20}
    assert fe.series(sma, "15m").iloc[:19].isna().all(), \
        "fixture must open with rows where the indicator is still warming"
    group = {"not": {"all": [{"lhs": {"src": "close"}, "op": ">", "rhs": sma}]}}
    out = fe.group_series(group, "15m")
    assert not out.iloc[:19].any(), \
        "not over unknown must not read as fired on bars 0..18"
    assert out.iloc[19:].any(), "fixture must fire once the indicator is warm"


def test_a_not_nested_as_an_item_of_a_list_is_recognized_as_a_group():
    """"X and not Y", the ordinary spelling, evaluated rather than described.

    Every other `not` in this module sits at the TOP of its tree, where
    `_group_reduce_na` reaches it through the `key == "not"` branch. That
    branch is not the one that makes `not` composable. The recognizer inside
    the list comprehension is, and until this test nothing in the suite
    evaluated a `not` that was an ITEM of an all/any list: the nested cases
    elsewhere go to margins, to describe, to canon and to the Pine refusal,
    none of which is the evaluator.

    So the one line making `not` usable in a real spec was guarded by nothing.
    Narrowing `is_group_node(i)` back to the pre-node `("all" in i or "any" in
    i)` left all 1976 tests passing while `group_series` raised
    `KeyError: 'lhs'` on this spec, the item having fallen through to the
    condition leaf. That exception is not loud where it lands: the platform
    scanner catches Exception around `detect_events` and publishes zero market
    events with a log warning.

    Asserted against the hand-computed complement rather than against a
    constant, so the test says what "and not" means rather than that it
    returned something.

    Compared on the WARM rows only, and that restriction is the point rather
    than a convenience. `group_series` resolves unknown to False at its public
    boundary, so composing two of its results negates a False and reads True,
    while the evaluator composes underneath that boundary where ~NA is NA. The
    two disagree on exactly the warm-up rows, the evaluator is the one that is
    right, and which of them is right is N3-D4's question rather than this
    test's; `test_not_over_a_warming_operand_does_not_fire` owns it. Comparing
    everywhere would make this test fail for that reason instead of for its
    own.
    """
    fe = FrameEval(_frames(), TFS)
    x = {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 5}}
    y = {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 20}}
    nested = {"all": [x, {"not": {"all": [y]}}]}

    warm = (fe.series({"ind": "sma", "n": 20}, "15m").notna()
            & fe.series({"ind": "sma", "n": 5}, "15m").notna())
    assert warm.any() and not warm.all(), "fixture must carry both phases"

    out = fe.group_series(nested, "15m")
    want = (fe.group_series({"all": [x]}, "15m")
            & ~fe.group_series({"all": [y]}, "15m"))
    # Both sides of the conjunction have to do work, or the equality holds
    # vacuously over a series that is False everywhere.
    assert out[warm].any() and not out[warm].all(), \
        out[warm].value_counts().to_dict()
    pd.testing.assert_series_equal(out[warm], want[warm], check_names=False)


def test_not_over_a_not_yet_closed_higher_timeframe_operand_does_not_fire():
    """Acceptance item 7, through N3-D3's OPERAND path.

    The group is evaluated on 15m and its operand carries tf="1h", so _align
    leaves float NaN on every 15m row where no 1h bar has closed yet. That NaN
    becomes pd.NA inside the group tree, BELOW the negation, which is the only
    place cross-timeframe unknown can reach a `not`. Evaluating the group on
    "1h" instead would short-circuit _align on src_tf == dst_tf, put no NaN in
    the tree at all, and read False on the blind rows only because the
    boolean lift zeroes invisible rows, which sits ABOVE the negation and is
    explicitly not what this is aimed at.
    """
    frames = _frames()
    frames["1h"] = frames["1h"].iloc[8:]
    fe = FrameEval(frames, TFS)
    operand = {"src": "close", "tf": "1h"}
    blind = [i for i, v in enumerate(fe.series(operand, "15m").isna()) if v]
    assert len(blind) > 8, "fixture must open with a real blind window"
    group = {"not": {"any": [{"lhs": operand, "op": "<", "rhs": -1.0}]}}
    out = fe.group_series(group, "15m")
    assert not out.iloc[blind].any(), \
        "a negation over an unknown higher-timeframe operand must not fire"
    seeing = sorted(set(range(len(out))) - set(blind))
    assert out.iloc[seeing].all(), "the visible rows must still read satisfied"
    assert out.dtype == bool


def test_not_evaluates_to_kleene_negation():
    frames = _closes([1.0, 2.0, 4.0, 5.0, 2.0, 6.0])
    fe = FrameEval(frames, TFS)
    group = {"not": {"any": [
        {"lhs": {"src": "close"}, "op": ">", "rhs": 1000.0},
        {"lhs": {"src": "close"}, "op": "<", "rhs": -1000.0}]}}
    out = fe.group_series(group, "15m")
    assert out.dtype == bool
    assert out.all()  # neither disjunct ever true, so not(false or false)


# The leaf is a still-warming indicator so the unknown rows exist by
# construction: sma over 20 bars is NA for the first 19. The arg is `n`; a
# spelling _eval does not know (`len`) is merged into the arg dict without
# refusal and the term then runs on its DEFAULT length, so the fixture would
# have no NA rows at all and the assertion below would be the only thing
# saying so.
_WARMING = {"lhs": {"ind": "sma", "n": 20}, "op": ">", "rhs": {"src": "close"}}


def test_double_negation_evaluates_to_the_original_group():
    """N3-D7's middle verb. Validating and canonicalizing a double negation
    were already covered; evaluating one was not, so a reducer that accepted
    only all/any beneath a negation would pass both and raise the moment a
    saved spec with {"not": {"not": G}} was actually run.

    All three Kleene values, because the interesting one is NA: under D8 an
    unknown must survive both negations as unknown. An implementation that
    collapsed NA to False anywhere in the pair round-trips True and False
    correctly and fails only here.
    """
    fe = FrameEval(_frames(), TFS)
    inner = {"all": [_WARMING]}
    base = fe._group_reduce_na(inner, "15m")
    once = fe._group_reduce_na({"not": inner}, "15m")
    twice = fe._group_reduce_na({"not": {"not": inner}}, "15m")

    assert base.isna().any(), (
        "no unknown rows in the fixture, so the NA leg below proves nothing")
    known = base.dropna()
    assert known.any() and not known.all(), \
        "fixture must carry all three Kleene values, not only NA and one other"
    assert twice.equals(base), "double negation did not round-trip"
    assert not twice.equals(once), "double negation collapsed to a single one"
    assert twice[base.isna()].isna().all(), "unknown did not survive both negations"


def test_double_negation_is_bool_at_the_public_boundary():
    """N3-D4 still holds through a nested negation: the private reducer is
    nullable, group_series is not."""
    fe = FrameEval(_frames(), TFS)
    out = fe.group_series({"not": {"not": {"all": [_WARMING]}}}, "15m")
    assert out.dtype == bool
