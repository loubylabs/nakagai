"""rule_sentence: a rule spec in plain sentences for a person.

The DSL readback (describe_spec) is untouched and has its own goldens; these
freeze the people-facing words.
"""

from datetime import time

import pytest

from nakagai.strategies.rules import (
    PRIOR_DAY, Term, WindowSpec, core_vocabulary, describe_spec,
    rule_sentence, validate_spec,
)
from nakagai.strategies.rules.spec import LOW_IEX_DISCLOSURE

LONDON = WindowSpec("london", "Europe/London", time(8), time(16, 30),
                    "weekday", "low_iex")
NY_OPEN_30 = WindowSpec("ny_open_30", "America/New_York", time(9, 30),
                        time(10), "xnys_session", "standard")
WINDOWS = core_vocabulary().with_windows(PRIOR_DAY, LONDON, NY_OPEN_30)


def _spec(long=None, *, short=None, timeframe="1d", **rest) -> dict:
    spec = {"version": 2, "name": "plain", "timeframe": timeframe, **rest}
    if long is not None:
        spec["long"] = {"all": long}
    if short is not None:
        spec["short"] = {"all": short}
    return spec


def _when(lhs, op, rhs) -> dict:
    return {"lhs": lhs, "op": op, "rhs": rhs}


CLOSE = {"src": "close"}


def _long_text(spec, vocabulary=None) -> str:
    """The text between `when ` and the first sentence end."""
    text = rule_sentence(spec, vocabulary)
    return text.split(" when ", 1)[1].split(". Stop ", 1)[0]


def test_the_issue_example_reads_as_a_sentence():
    spec = _spec([_when(CLOSE, "crosses_above", {"ind": "sma", "n": 20})])
    assert validate_spec(spec) == []
    assert rule_sentence(spec) == (
        "Long on the daily chart when the close crosses above the 20-day "
        "simple moving average. Stop 2x the 14-day average true range below "
        "entry. Target 2x the risk.")


def test_the_dsl_readback_is_unchanged():
    spec = _spec([_when(CLOSE, "crosses_above", {"ind": "sma", "n": 20})])
    assert describe_spec(spec) == (
        'Strategy "plain" on 1d bars.\n'
        "Enter long when ALL of:\n"
        "  - close crosses above sma(20)\n"
        "Stop: 2x ATR(14) from entry. Target: 2x the risked distance.")


@pytest.mark.parametrize(("node", "words"), [
    ({"ind": "sma", "n": 50}, "the 50-day simple moving average"),
    ({"ind": "ema", "n": 9}, "the 9-day exponential moving average"),
    ({"ind": "rsi"}, "the 14-day RSI"),
    ({"ind": "roc", "n": 10}, "the 10-day rate of change"),
    ({"ind": "zscore"}, "the 20-day z-score"),
    ({"ind": "stdev"}, "the 20-day standard deviation"),
    ({"ind": "sma", "n": 20, "of": {"src": "volume"}},
     "the 20-day simple moving average of volume"),
    ({"ind": "highest", "n": 20}, "the highest value of the close over the last 20 days"),
    ({"ind": "lowest", "n": 10, "of": {"src": "low"}},
     "the lowest value of the low over the last 10 days"),
    ({"ind": "macd", "field": "signal"}, "the MACD signal line (12, 26, 9)"),
    ({"ind": "macd", "field": "hist"}, "the MACD histogram (12, 26, 9)"),
    ({"ind": "bb", "field": "upper"},
     "the upper Bollinger band (20-day, 2 standard deviations)"),
    ({"ind": "atr"}, "the 14-day average true range"),
    ({"ind": "donchian", "field": "lower", "n": 55},
     "the lower line of the 55-day Donchian channel"),
    ({"ind": "supertrend"}, "the Supertrend line (10-day, 3x ATR)"),
    ({"ind": "vwap"}, "the session VWAP"),
    ({"ind": "stoch", "field": "d"}, "the 14-day stochastic %D (3-bar smoothing)"),
    ({"ind": "adx"}, "the 14-day ADX"),
    ({"ind": "obv"}, "on-balance volume"),
    ({"ind": "ichimoku", "field": "kijun"}, "the Ichimoku base line (9, 26, 52, 26)"),
    ({"ind": "keltner", "field": "upper"},
     "the upper Keltner channel line (20-day, 2x ATR)"),
    ({"ind": "cci"}, "the 20-day CCI"),
    ({"ind": "mfi"}, "the 14-day money flow index"),
    ({"ind": "wpr"}, "the 14-day Williams %R"),
    ({"prim": "gap_pct"}, "the opening gap from the prior close (%)"),
    ({"prim": "swing_high"}, "the latest swing high (3 days each side)"),
    ({"prim": "swing_low", "k": 1}, "the latest swing low (1 day each side)"),
    ({"prim": "day_of_week"}, "the day of the week (0 is Monday)"),
    ({"prim": "fvg_nearest", "direction": "short", "field": "mid"},
     "the middle of the nearest open bearish fair value gap "
     "(at least 0.25 ATR, last 40 days)"),
    ({"prim": "leg_retrace"},
     "how far the close has retraced the latest up swing (0 to 1, 3 days each side)"),
    ({"prim": "order_block"},
     "the top of the nearest bullish order block (body at least 1.5 ATR, last 40 days)"),
])
def test_each_term_reads_in_words(node, words):
    assert _long_text(_spec([_when(CLOSE, ">", node)])) == f"the close is above {words}"


def test_intraday_terms_read_in_bars_and_sessions():
    spec = _spec([_when({"prim": "rvol"}, ">", 2),
                  _when({"prim": "minutes_into_session"}, "<", 30),
                  _when({"ind": "rsi"}, "<", 30)], timeframe="15m")
    assert validate_spec(spec) == []
    assert rule_sentence(spec).startswith(
        "Long on the 15-minute chart when relative volume against the same "
        "time of day over the last 20 sessions is above 2, the minutes since "
        "the session opened is below 30 and the 14-bar RSI is below 30.")


def test_hourly_and_weekly_lengths_name_their_unit():
    hourly = _spec([_when(CLOSE, ">", {"ind": "sma"})], timeframe="1h")
    weekly = _spec([_when(CLOSE, ">", {"ind": "sma"})], timeframe="1w")
    assert rule_sentence(hourly).startswith(
        "Long on the hourly chart when the close is above the 20-hour simple")
    assert rule_sentence(weekly).startswith(
        "Long on the weekly chart when the close is above the 20-week simple")


def test_groups_nest_with_and_or_and_not():
    spec = _spec([
        _when({"ind": "rsi"}, "<", 30),
        {"any": [_when(CLOSE, ">", {"ind": "sma"}),
                 {"not": {"all": [_when({"prim": "gap_pct"}, ">=", 2)]}}]},
    ])
    assert validate_spec(spec) == []
    assert _long_text(spec) == (
        "the 14-day RSI is below 30 and (the close is above the 20-day simple "
        "moving average or not (the opening gap from the prior close (%) is "
        "at or above 2))")


def test_a_top_level_not_reads_as_a_sentence():
    spec = _spec()
    spec["long"] = {"not": {"any": [_when(CLOSE, "<", 10),
                                    _when(CLOSE, ">", 500)]}}
    assert validate_spec(spec) == []
    assert _long_text(spec) == (
        "it is not true that the close is below 10 or the close is above 500")


def test_every_operator_has_words():
    spec = _spec([_when(CLOSE, op, 1) for op in
                  (">", "<", ">=", "<=")]
                 + [_when(CLOSE, op, {"ind": "ema"}) for op in
                    ("crosses_above", "crosses_below")])
    assert _long_text(spec) == (
        "the close is above 1, the close is below 1, the close is at or "
        "above 1, the close is at or below 1, the close crosses above the "
        "20-day exponential moving average and the close crosses below the "
        "20-day exponential moving average")


def test_math_reads_in_words():
    spec = _spec([
        _when({"op": "-", "args": [CLOSE, {"src": "open"}]}, ">",
              {"op": "*", "args": [0.5, {"ind": "atr"}]}),
        _when({"op": "abs", "args": [{"ind": "roc"}]}, "<",
              {"op": "max", "args": [1, 2, 3]}),
        _when({"op": "/", "args": [{"src": "volume"}, 1000000]}, ">",
              {"op": "min", "args": [{"src": "high"}, {"op": "+", "args": [1, 2]}]}),
    ])
    assert validate_spec(spec) == []
    assert _long_text(spec) == (
        "(the close minus the open) is above (0.5 times the 14-day average "
        "true range), the absolute value of the 20-day rate of change is "
        "below the higher of 1, 2 and 3 and (volume divided by 1,000,000) is "
        "above the lower of the high and (1 plus 2)")


def test_symbol_and_foreign_timeframe_scope():
    spec = _spec([
        _when({"src": "close", "sym": "SPY", "tf": "1d"}, ">",
              {"ind": "sma", "n": 50, "sym": "SPY", "tf": "1d"}),
        _when({"src": "volume", "sym": "QQQ"}, ">",
              {"ind": "sma", "of": {"src": "volume"}, "tf": "4h"}),
        _when({"ind": "vwap", "tf": "1h"}, "<", CLOSE),
    ], timeframe="15m")
    assert validate_spec(spec) == []
    assert _long_text(spec) == (
        "SPY's daily close is above SPY's 50-day simple moving average, "
        "QQQ's volume is above the 20-bar simple moving average of volume on "
        "4-hour bars and the session VWAP on hourly bars is below the close")


def test_bars_since_reads_its_condition():
    spec = _spec([_when({"prim": "bars_since",
                         "cond": _when(CLOSE, ">", {"ind": "bb", "field": "upper"})},
                        "<", 5)])
    assert validate_spec(spec) == []
    assert _long_text(spec) == (
        'the days since "the close is above the upper Bollinger band '
        '(20-day, 2 standard deviations)" was last true is below 5')


def test_windows_read_from_their_rows():
    spec = _spec([
        _when(CLOSE, ">", {"ind": "highest", "of": {"src": "high"},
                           "window": "prior_day"}),
        _when(CLOSE, ">", {"ind": "first", "window": "ny_open_30"}),
    ], timeframe="15m")
    assert validate_spec(spec, WINDOWS) == []
    assert _long_text(spec, WINDOWS) == (
        "the close is above the highest value of the high during the prior "
        "session and the close is above the first value of the close between "
        "09:30 and 10:00 New York time")
    assert LOW_IEX_DISCLOSURE not in rule_sentence(spec, WINDOWS)


def test_a_low_iex_window_adds_the_disclosure_once():
    spec = _spec([
        _when(CLOSE, ">", {"ind": "highest", "window": "london"}),
        _when(CLOSE, ">", {"ind": "lowest", "window": "london"}),
    ], timeframe="15m")
    assert validate_spec(spec, WINDOWS) == []
    text = rule_sentence(spec, WINDOWS)
    assert "between 08:00 and 16:30 London time" in text
    assert text.endswith(f"Target 2x the risk. {LOW_IEX_DISCLOSURE}")
    assert text.count(LOW_IEX_DISCLOSURE) == 1


def test_exits_and_percent_risk_for_a_short():
    spec = _spec(short=[_when(CLOSE, "<", {"ind": "sma"})],
                 exits={"exit": {"any": [_when({"ind": "rsi"}, "<", 20)]},
                        "trailing": {"kind": "atr", "mult": 3, "n": 10},
                        "time_stop": {"bars": 26},
                        "breakeven_at": {"rr": 1.5}},
                 risk={"stop": {"kind": "percent", "pct": 1.5},
                       "target": {"kind": "percent", "pct": 4}})
    assert validate_spec(spec) == []
    assert rule_sentence(spec) == (
        "Short on the daily chart when the close is below the 20-day simple "
        "moving average. Exit early when the 14-day RSI is below 20. Trail "
        "the stop 3x the 10-day average true range behind the best price. "
        "Close the trade after 26 15-minute bars. Move the stop to entry once "
        "the trade is up 1.5x the risk. Stop 1.5% above entry. Target 4% "
        "below entry.")


def test_both_sides_and_a_percent_trail():
    spec = _spec([_when(CLOSE, ">", {"ind": "sma"})],
                 short=[_when(CLOSE, "<", {"ind": "sma"})],
                 exits={"trailing": {"kind": "percent", "pct": 2}},
                 risk={"stop": {"kind": "percent", "pct": 1},
                       "target": {"kind": "percent", "pct": 3}})
    assert validate_spec(spec) == []
    text = rule_sentence(spec)
    assert "Long on the daily chart when" in text
    assert "Short on the daily chart when" in text
    assert text.endswith("Trail the stop 2% behind the best price. "
                         "Stop 1% from entry. Target 3% from entry.")


def test_every_core_term_has_its_words():
    vocabulary = core_vocabulary()
    assert [t.name for t in vocabulary.all_terms() if t.plain is None] == []


def test_an_injected_term_without_words_reads_through_the_fallback():
    term = Term("chaikin_osc", "bar", {"fast_n": (2, 50), "slow_n": (3, 100)},
                {"fast_n": 3, "slow_n": 10}, lambda b, a: b["close"])
    vocabulary = core_vocabulary().with_terms(term)
    spec = _spec([_when({"ind": "chaikin_osc", "slow_n": 20}, ">", 0)])
    assert _long_text(spec, vocabulary) == (
        "chaikin osc (fast n 3, slow n 20) is above 0")


def test_an_injected_phrase_is_used_and_cannot_raise_out():
    good = Term("news_count", "primitive", {}, {}, lambda *a, **k: None,
                plain=lambda a, unit, of: "the story count")
    bad = Term("news_bad", "primitive", {}, {}, lambda *a, **k: None,
               plain=lambda a, unit, of: a["missing"])
    vocabulary = core_vocabulary().with_terms(good, bad)
    spec = _spec([_when({"prim": "news_count"}, ">", 2),
                  _when({"prim": "news_bad"}, ">", 0)])
    assert _long_text(spec, vocabulary) == (
        "the story count is above 2 and news bad is above 0")


def test_a_non_callable_phrase_is_refused_where_the_term_is_built():
    with pytest.raises(TypeError, match="plain"):
        Term("x", "bar", {}, {}, lambda b, a: b, plain="words")


@pytest.mark.parametrize("bad", [
    None, [], "spec", 3,
    {"long": 5},
    {"long": {"all": "x"}},
    {"long": {"all": [{"lhs": {"ind": ["x"]}, "op": ["y"], "rhs": {"op": "abs"}}]}},
    {"long": {"all": [{"lhs": {"prim": "nope"}, "op": ">", "rhs": {"src": 3}}]}},
    {"long": {"all": [{"lhs": {"ind": "sma", "n": "x"}, "op": ">", "rhs": True}]}},
    {"long": {"all": [{"lhs": {"ind": "macd", "field": ["x"]}, "op": ">", "rhs": 0}]}},
    {"timeframe": ["1d"], "long": {"all": []}},
    {"risk": 3, "exits": {"trailing": 1, "time_stop": "x"}},
    {"risk": {"stop": [], "target": {"kind": "rr", "rr": "x"}}},
])
def test_garbage_never_raises(bad):
    assert isinstance(rule_sentence(bad), str)
