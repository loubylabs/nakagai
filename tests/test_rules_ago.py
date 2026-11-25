"""`ago`: a series read n bars back, end to end through the grammar.

The grammar had no way to say "the previous bar", so a candle pattern (an
engulfing bar, an inside bar) or "above yesterday's high" could not be written.
`ago` is that one operator: its `of` series shifted back `n` bars of the frame
it is evaluated on. These tests pin it through every reader the grammar has:
evaluation, validation, the DSL readback, the plain sentence, identity and Pine.
"""

import copy
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from nakagai.data.schema import DEFAULT_TIMEFRAMES as TFS
from nakagai.nlbuilder.prompt import render_system_prompt
from nakagai.screen.prompt import render_screen_prompt
from nakagai.strategies import indicators as ind
from nakagai.strategies.rules import (
    describe_spec, lower_pine, past_fires, rule_sentence, spec_hash,
    validate_spec,
)
from nakagai.strategies.rules.frame_eval import FrameEval
from nakagai.strategies.rules.vocabulary import core_vocabulary, is_whole_rule
from tests.whole_frame_oracle import prefix_value

OPEN, HIGH, LOW, CLOSE = ({"src": s} for s in ("open", "high", "low", "close"))


def ago(of=None, n=None, **extra):
    node = {"ind": "ago", **({"of": of} if of is not None else {}), **extra}
    return {**node, "n": n} if n is not None else node


def _spec(conds, timeframe="1d"):
    return {"version": 2, "name": "ago", "timeframe": timeframe,
            "long": {"all": conds}}


def _random_bars(timeframe="15m", sessions=6, seed=7):
    """RTH-shaped bars whose candle bodies take both signs."""
    rng = np.random.default_rng(seed)
    if timeframe == "1d":
        idx = pd.bdate_range("2026-01-05", periods=sessions * 26, tz="UTC",
                             name="ts")
    else:
        days = pd.bdate_range("2026-01-05", periods=sessions, tz="UTC")
        idx = pd.DatetimeIndex(
            [d + pd.Timedelta(hours=14, minutes=30) + i * TFS.deltas["15m"]
             for d in days for i in range(26)], name="ts")
    n = len(idx)
    close = 100 + np.cumsum(rng.normal(0, 0.4, n))
    open_ = close + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) + 0.1,
         "low": np.minimum(open_, close) - 0.1, "close": close,
         "volume": 1000.0 + rng.integers(0, 500, n)}, index=idx)


def _fe(bars, tf):
    frames = {("SPY", tf): bars}
    if tf == "15m":
        agg = {"open": "first", "high": "max", "low": "min", "close": "last",
               "volume": "sum"}
        frames[("SPY", "1h")] = bars.resample("1h").agg(agg).dropna()
    return FrameEval("SPY", frames, TFS)


# --- evaluation --------------------------------------------------------------

@pytest.mark.parametrize("src", ["open", "high", "low", "close", "volume"])
@pytest.mark.parametrize("n", [1, 3])
def test_ago_is_its_source_shifted_back_n_bars(src, n):
    bars = _random_bars()
    got = _fe(bars, "15m").series(ago({"src": src}, n), "15m")
    pd.testing.assert_series_equal(got, bars[src].shift(n), check_names=False)


def test_ago_defaults_to_one_bar_of_the_close():
    bars = _random_bars()
    got = _fe(bars, "15m").series({"ind": "ago"}, "15m")
    pd.testing.assert_series_equal(got, bars["close"].shift(1),
                                   check_names=False)


def test_ago_of_an_indicator_shifts_the_indicator():
    bars = _random_bars()
    got = _fe(bars, "15m").series(ago({"ind": "sma", "n": 20}, 3), "15m")
    pd.testing.assert_series_equal(got, ind.sma(bars["close"], 20).shift(3),
                                   check_names=False)


def test_ago_of_a_math_expression_shifts_the_whole_expression():
    bars = _random_bars()
    body = {"op": "-", "args": [CLOSE, OPEN]}
    got = _fe(bars, "15m").series(ago(body, 2), "15m")
    pd.testing.assert_series_equal(got, (bars["close"] - bars["open"]).shift(2),
                                   check_names=False)


def test_ago_is_a_source_another_term_can_measure():
    # The highest high of the 20 bars BEFORE this one is the Donchian upper
    # edge, which excludes the current bar by construction.
    bars = _random_bars()
    node = {"ind": "highest", "n": 20, "of": ago(HIGH, 1)}
    got = _fe(bars, "15m").series(node, "15m")
    pd.testing.assert_series_equal(got, ind.donchian(bars, 20)["upper"],
                                   check_names=False)


def test_the_first_n_bars_have_no_value_and_never_fire():
    bars = _random_bars()
    got = _fe(bars, "15m").series(ago(CLOSE, 4), "15m")
    assert got.iloc[:4].isna().all() and got.iloc[4:].notna().all()
    # A condition every bar with history satisfies: close > close 4 bars ago
    # minus a mile. Unknown on the first 4 bars, so it is not-fired there.
    cond = {"lhs": CLOSE, "op": ">",
            "rhs": {"op": "-", "args": [ago(CLOSE, 4), 1e6]}}
    fired = _fe(bars, "15m").condition_series(cond, "15m")
    assert not fired.iloc[:4].any() and fired.iloc[4:].all()


def test_on_an_intraday_frame_the_previous_bar_can_be_yesterday_s_last():
    bars = _random_bars("15m", sessions=2)
    got = _fe(bars, "15m").series(ago(CLOSE, 1), "15m")
    first_of_day_two = 26
    assert got.iloc[first_of_day_two] == bars["close"].iloc[first_of_day_two - 1]


def test_with_its_own_tf_ago_shifts_by_that_frame_s_bars():
    bars = _random_bars("15m", sessions=3)
    fe = _fe(bars, "15m")
    hourly = fe.on("SPY", "1h")
    got = fe.series(ago(CLOSE, 1, tf="1h"), "15m")
    # At each 15m bar, the last 1h bar closed by its close, then one more back.
    want = fe.to_driving(hourly["close"].shift(1), "SPY", "1h")
    pd.testing.assert_series_equal(got, want, check_names=False)
    # A tf on the SOURCE instead keeps the shift on the 15m frame: one 15m bar
    # back of the hourly close as each 15m bar saw it, a different reading.
    on_source = fe.series(ago({"src": "close", "tf": "1h"}, 1), "15m")
    pd.testing.assert_series_equal(
        on_source, fe.series({"src": "close", "tf": "1h"}, "15m").shift(1),
        check_names=False)
    assert not got.equals(on_source)


# --- an engulfing rule, by hand ------------------------------------------------

ENGULFING = [
    {"lhs": ago(CLOSE, 1), "op": "<", "rhs": ago(OPEN, 1)},
    {"lhs": CLOSE, "op": ">", "rhs": OPEN},
    {"lhs": CLOSE, "op": ">", "rhs": ago(OPEN, 1)},
    {"lhs": OPEN, "op": "<", "rhs": ago(CLOSE, 1)},
]


def _candles(rows):
    idx = pd.bdate_range("2026-03-02", periods=len(rows), tz="UTC", name="ts")
    o = np.array([r[0] for r in rows], dtype=float)
    c = np.array([r[1] for r in rows], dtype=float)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.1,
                         "low": np.minimum(o, c) - 0.1, "close": c,
                         "volume": 1000.0}, index=idx)


# (open, close) per day. Engulfing on 1 and 6; 4 is bullish after a bearish
# bar but its close stays under the prior open, so it does not engulf.
CANDLES = _candles([(10, 9), (8.5, 10.5), (10.5, 11), (11, 10), (10.2, 10.8),
                    (10.8, 10.3), (10.1, 11), (11, 11.5)])


def test_a_bullish_engulfing_rule_validates_and_fires_where_expected():
    spec = _spec(ENGULFING)
    assert validate_spec(spec) == []
    fired = _fe(CANDLES, "1d").group_series(spec["long"], "1d")
    assert np.flatnonzero(fired.to_numpy()).tolist() == [1, 6]


def test_past_fires_reads_the_engulfing_rule_on_the_same_bars():
    frames = {("AAA", "1d"): CANDLES,
              ("SPY", "1d"): CANDLES.assign(close=100.0)}
    out = past_fires(_spec(ENGULFING), frames, "AAA", benchmark="SPY",
                     horizon=1, warmup=0)
    assert out.long.fire_bars == (1, 6)


def test_whole_frame_and_bar_by_bar_replay_fire_on_the_same_bars():
    """The whole-frame series against the per-bar prefix, condition by
    condition, at every row: the property replay rests on."""
    bars = _random_bars("15m", sessions=4)
    frames = {("SPY", "15m"): bars}
    fe = FrameEval("SPY", frames, TFS)
    spec = _spec(ENGULFING, timeframe="15m")
    whole = fe.group_series(spec["long"], "15m").to_numpy()
    ops = {"<": np.less, ">": np.greater}
    replay = []
    for i in range(len(bars)):
        now = bars.index[i] + TFS.deltas["15m"]
        held = True
        for cond in ENGULFING:
            lhs, rhs = (prefix_value(cond[side], frames, "SPY", "15m", now, TFS)
                        for side in ("lhs", "rhs"))
            held &= bool(ops[cond["op"]](lhs, rhs))
        replay.append(held)
    assert whole.tolist() == replay
    assert not whole[0]
    assert 2 <= whole.sum() < len(bars) // 2, "the rule must actually fire"


# --- validation ----------------------------------------------------------------

def _errs(node):
    return validate_spec(_spec([{"lhs": CLOSE, "op": ">", "rhs": node}]))


@pytest.mark.parametrize("n", [0, 501, -1])
def test_n_outside_its_bounds_is_refused(n):
    assert _errs(ago(OPEN, n)) == [
        f"long.all[0].rhs: ago.n must be a number in [1, 500], got {n!r}"]


@pytest.mark.parametrize("n", [1.5, 2.25])
def test_a_fractional_n_is_refused(n):
    assert _errs(ago(OPEN, n)) == [
        f"long.all[0].rhs: ago.n must be a whole number in [1, 500], "
        f"got {n!r}"]


def test_a_whole_float_n_is_the_same_whole_number():
    assert _errs(ago(OPEN, 2.0)) == []


@pytest.mark.parametrize("n", ["1", True, None])
def test_a_non_number_n_is_refused(n):
    assert _errs({"ind": "ago", "of": OPEN, "n": n}) == [
        f"long.all[0].rhs: ago.n must be a number in [1, 500], got {n!r}"]


def test_an_unknown_arg_is_refused():
    assert _errs({"ind": "ago", "of": OPEN, "k": 2}) == [
        "long.all[0].rhs: ago takes no arg 'k' (valid: ['n'])"]


def test_ago_takes_no_window():
    assert _errs({"ind": "ago", "window": "x"}) == [
        "long.all[0].rhs: ago does not support window aggregation"]


def test_the_whole_range_rule_survives_pickle_and_deepcopy():
    rule = core_vocabulary().indicators["ago"].args["n"]
    for copy_of in (lambda r: pickle.loads(pickle.dumps(r)), copy.deepcopy):
        again = copy_of(rule)
        assert again == (1, 500) and is_whole_rule(again)


def test_ago_is_admitted_on_daily_and_weekly_specs():
    for tf in ("1d", "1w"):
        assert validate_spec(_spec(ENGULFING, timeframe=tf)) == []


# --- readback ------------------------------------------------------------------

def test_describe_spec_renders_ago_in_the_dsl():
    text = describe_spec(_spec([
        {"lhs": ago(CLOSE, 1), "op": "<", "rhs": ago(OPEN, 2)},
        {"lhs": CLOSE, "op": ">", "rhs": ago({"ind": "sma", "n": 20}, 3)},
    ]))
    assert "  - ago(1) is below ago(2, of=open)\n" in text
    assert "  - close is above ago(3, of=sma(20))\n" in text


@pytest.mark.parametrize(("node", "tf", "words"), [
    (ago(OPEN, 1), "1d", "the open 1 day ago"),
    (ago(n=3), "1d", "the close 3 days ago"),
    ({"ind": "ago"}, "1h", "the close 1 hour ago"),
    (ago(HIGH, 2), "15m", "the high 2 bars ago"),
    (ago({"src": "volume"}, 1), "1w", "volume 1 week ago"),
    (ago({"ind": "sma", "n": 20}, 5), "1d",
     "the 20-day simple moving average 5 days ago"),
    (ago(CLOSE, 1, tf="1h"), "15m", "the close 1 hour ago"),
    (ago(CLOSE, 1, sym="QQQ"), "1d", "QQQ's close 1 day ago"),
])
def test_the_plain_sentence_says_how_far_back(node, tf, words):
    text = rule_sentence(_spec([{"lhs": CLOSE, "op": ">", "rhs": node}], tf))
    assert f" when the close is above {words}. " in text


def test_the_engulfing_rule_reads_as_a_sentence():
    assert rule_sentence(_spec(ENGULFING)).startswith(
        "Long on the daily chart when the close 1 day ago is below the open "
        "1 day ago, the close is above the open, the close is above the open "
        "1 day ago and the open is below the close 1 day ago.")


# --- identity ------------------------------------------------------------------

def test_n_is_part_of_a_spec_s_identity():
    one = _spec([{"lhs": CLOSE, "op": ">", "rhs": ago(OPEN, 1)}])
    two = _spec([{"lhs": CLOSE, "op": ">", "rhs": ago(OPEN, 2)}])
    assert spec_hash(one) != spec_hash(two)


def test_the_default_n_and_its_explicit_spellings_are_one_spec():
    hashes = {spec_hash(_spec([{"lhs": CLOSE, "op": ">", "rhs": node}]))
              for node in (ago(OPEN), ago(OPEN, 1), ago(OPEN, 1.0))}
    assert len(hashes) == 1


def test_the_default_source_is_the_close_in_identity_too():
    implicit = _spec([{"lhs": OPEN, "op": ">", "rhs": {"ind": "ago"}}])
    explicit = _spec([{"lhs": OPEN, "op": ">", "rhs": ago(CLOSE, 1)}])
    assert spec_hash(implicit) == spec_hash(explicit)


# --- prompts -------------------------------------------------------------------

@pytest.mark.parametrize("render", [render_system_prompt, render_screen_prompt])
def test_both_compilers_teach_ago_with_an_example_that_validates(render):
    prompt = render()
    assert "- ago(n=(1, 500)) [takes of=<expr>]" in prompt
    rule = prompt.split("- ago is its `of` series", 1)[1].split("\n\n", 1)[0]
    rule = rule.split(" is\n", 1)[1]
    example = json.loads(rule[:rule.rindex("}") + 1])
    assert validate_spec(_spec([example])) == []


# --- Pine ----------------------------------------------------------------------

LHS = "nk_long_all_0_lhs"


def _program(lhs):
    return lower_pine(_spec([{"lhs": lhs, "op": ">", "rhs": 0}], "15m"))


def test_pine_indexes_the_source_s_history_by_the_n_input():
    program = _program(ago(OPEN, 1))
    assert f"nk_ago_1 = open[{LHS}_ago_n]" in program.calculations
    (n_input,) = [i for i in program.inputs if i.name == f"{LHS}_ago_n"]
    assert (n_input.kind, n_input.default, n_input.bounds) == ("int", 1, (1, 500))
    # The offset is an input, so TradingView cannot infer the buffer: the
    # script owes max_bars_back, one more than the deepest offset.
    assert program.max_bars_back == 501


def test_pine_names_a_compound_source_before_indexing_it():
    program = _program(ago({"op": "-", "args": [HIGH, LOW]}, 2))
    assert "nk_ago_1_source = (high - low)" in program.calculations
    assert f"nk_ago_1 = nk_ago_1_source[{LHS}_ago_n]" in program.calculations


def test_pine_indexes_a_nested_indicator_s_own_identifier():
    program = _program(ago({"ind": "sma", "n": 20}, 3))
    assert f"nk_sma_1 = ta.sma(close, {LHS}_of_sma_n)" in program.calculations
    assert f"nk_ago_1 = nk_sma_1[{LHS}_ago_n]" in program.calculations
