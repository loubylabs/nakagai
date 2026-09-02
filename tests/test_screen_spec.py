"""ScreenSpec v1: the conditions-only IR reusing RuleSpec v2's grammar."""

from datetime import time

import pytest
import numpy as np

from nakagai.screen.prompt import render_screen_prompt
from nakagai.strategies.rules.spec import group_text, validate_condition_group
from nakagai.strategies.rules.vocabulary import (
    Term, Vocabulary, core_vocabulary,
)
from nakagai.strategies.rules.windows import PRIOR_DAY, WindowSpec

RSI_LT_30 = {"lhs": {"ind": "rsi", "n": 14}, "op": "<", "rhs": 30}
LOW_IEX_DISCLOSURE = "US-equity extended-hours IEX data can be sparse."
LONDON = WindowSpec(
    "london", "Europe/London", time(8), time(16, 30), "weekday", "low_iex")
NY_AM = WindowSpec(
    "ny_am", "America/New_York", time(9, 30), time(12),
    "xnys_session", "standard")
NY_OPEN_15 = WindowSpec(
    "ny_open_15", "America/New_York", time(9, 30), time(9, 45),
    "xnys_session", "standard")
NY_OPEN_30 = WindowSpec(
    "ny_open_30", "America/New_York", time(9, 30), time(10),
    "xnys_session", "standard")
WINDOW_VOCABULARY = core_vocabulary().with_windows(
    LONDON, NY_AM, NY_OPEN_15, NY_OPEN_30, PRIOR_DAY)


def _screen_with(expr: dict, tf: str = "15m") -> dict:
    return {
        "version": 1,
        "tf": tf,
        "conditions": {"all": [
            {"lhs": {"src": "close"}, "op": ">", "rhs": expr},
        ]},
    }


@pytest.mark.parametrize("expr,tf", [
    ({"ind": "highest", "of": {"src": "high"}, "window": "london"},
     "15m"),
    ({"ind": "last", "of": {"src": "close"}, "window": "prior_day"},
     "15m"),
    ({"ind": "highest", "of": {"src": "high"}, "tf": "15m",
      "window": "ny_open_15"}, "1h"),
    ({"ind": "highest", "of": {"src": "high"},
      "window": "ny_open_15"}, "15m"),
    ({"ind": "highest", "of": {"src": "high"},
      "window": "prior_day"}, "1d"),
], ids=["current", "required", "own-tf", "equal-width", "daily-prior"])
def test_screen_spec_accepts_every_window_contract(expr, tf):
    assert validate_screen_spec(
        _screen_with(expr, tf), vocabulary=WINDOW_VOCABULARY) == []


@pytest.mark.parametrize("expr,tf,expected", [
    ({"ind": "highest", "of": {"src": "high"}, "window": "unknown"},
     "15m", "unknown window 'unknown'"),
    ({"src": "high", "window": "london"},
     "15m", "window is only valid on an aggregate indicator"),
    ({"op": "max", "args": [{"src": "high"}, 1], "window": "london"},
     "15m", "window is only valid on an aggregate indicator"),
    ({"prim": "gap_pct", "window": "london"},
     "15m", "window is only valid on an aggregate indicator"),
    ({"ind": "rsi", "window": "london"},
     "15m", "rsi does not support window aggregation"),
    ({"ind": "first", "of": {"src": "open"}},
     "15m", "first requires window"),
    ({"ind": "highest", "of": {"src": "high"}, "n": 20,
      "window": "london"},
     "15m", "highest cannot combine n with window"),
    ({"ind": "highest", "of": {"src": "high"},
      "window": "ny_open_30"},
     "1h", "window 'ny_open_30' spans 30 minutes, narrower than '1h' bars "
           "(60 minutes)"),
    ({"ind": "highest", "of": {"src": "high"}, "window": "ny_am"},
     "1d", "window 'ny_am' is intraday and cannot be resolved from "
           "session-aligned '1d' bars"),
], ids=[
    "unknown", "source", "math", "primitive", "non-aggregate",
    "required", "scope-conflict", "wide-fixed-frame", "daily-current",
])
def test_screen_spec_refuses_every_invalid_window_shape(expr, tf, expected):
    errors = validate_screen_spec(
        _screen_with(expr, tf), vocabulary=WINDOW_VOCABULARY)
    assert errors == [f"conditions.all[0].rhs: {expected}"]


def test_validate_condition_group_accepts_a_bare_group():
    assert validate_condition_group({"all": [RSI_LT_30]}) == []


def test_validate_condition_group_rejects_a_non_group():
    errs = validate_condition_group([RSI_LT_30])
    assert errs and "expected" in errs[0]


def test_validate_condition_group_reports_paths_from_the_given_root():
    bad = {"all": [{"lhs": {"ind": "nope"}, "op": "<", "rhs": 1}]}
    errs = validate_condition_group(bad, "conditions")
    assert errs and errs[0].startswith("conditions.all[0]")


def test_group_text_renders_a_bare_group():
    text = group_text({"all": [RSI_LT_30]})
    assert "ALL of:" in text and "rsi(14) is below 30" in text


from nakagai.screen.spec import (
    describe_screen, is_intraday, max_lookback, referenced_timeframes,
    screen_reference_pairs, validate_screen_spec,
)

GOOD = {"version": 1, "tf": "1d", "conditions": {"all": [RSI_LT_30]}}

DISCOVERY_FACT_NAMES = (
    "float_shares",
    "shares_outstanding",
    "market_cap",
    "price",
    "change_pct",
    "gap_pct",
    "session_volume",
)

RELATIVE_SCOPE_SCREEN = {
    "version": 1,
    "tf": "15m",
    "conditions": {
        "all": [
            {
                "lhs": {"src": "close"},
                "op": ">",
                "rhs": {"src": "close", "sym": "SPY"},
            },
            {
                "lhs": {
                    "ind": "sma", "n": 20, "of": {"src": "close"},
                    "sym": "SPY", "tf": "15m",
                },
                "op": ">",
                "rhs": {
                    "ind": "sma", "n": 20,
                    "of": {"src": "close", "sym": "QQQ", "tf": "1d"},
                    "sym": "SPY", "tf": "15m",
                },
            },
        ],
    },
}


def test_validate_screen_spec_accepts_a_good_spec():
    assert validate_screen_spec(GOOD) == []


@pytest.mark.parametrize("fact", DISCOVERY_FACT_NAMES)
def test_screen_accepts_the_closed_discovery_fact_vocabulary(fact):
    spec = {"version": 1, "tf": "1d", "conditions": {"all": [
        {"lhs": {"fact": fact}, "op": "<", "rhs": 20_000_000},
    ]}}
    assert validate_screen_spec(spec) == []
    assert fact.replace("_", " ") in describe_screen(spec)


@pytest.mark.parametrize("extra", [
    {"tf": "1h"},
    {"sym": "SPY"},
    {"window": "prior_day"},
    {"provider": "somewhere"},
])
def test_fact_nodes_accept_no_scope_or_provider_keys(extra):
    spec = {"version": 1, "tf": "1d", "conditions": {"all": [
        {"lhs": {"fact": "price", **extra}, "op": ">", "rhs": 10},
    ]}}
    assert validate_screen_spec(spec)


def test_fact_nodes_refuse_cross_semantics():
    spec = {"version": 1, "tf": "1d", "conditions": {"all": [
        {"lhs": {"fact": "price"}, "op": "crosses_above", "rhs": 10},
    ]}}
    errors = validate_screen_spec(spec)
    assert any("level" in error for error in errors)


@pytest.mark.parametrize("lhs", [
    {"op": "+", "args": [{"fact": "price"}, 1]},
    {"op": "abs", "args": [{"fact": "price"}]},
    {"op": "+", "args": [1, 2]},
], ids=["fact-plus-number", "fact-absolute", "numbers-only"])
def test_cross_left_math_without_a_series_is_a_level(lhs):
    spec = {"version": 1, "tf": "1d", "conditions": {"all": [
        {"lhs": lhs, "op": "crosses_above", "rhs": 10},
    ]}}
    errors = validate_screen_spec(spec)
    assert any("level" in error for error in errors)


def test_cross_left_math_with_a_technical_series_is_accepted():
    spec = {"version": 1, "tf": "1d", "conditions": {"all": [
        {"lhs": {"op": "+", "args": [
            {"src": "close"}, {"fact": "price"},
        ]}, "op": "crosses_above", "rhs": 10},
    ]}}
    assert validate_screen_spec(spec) == []


def test_cross_left_math_with_a_series_is_not_misclassified_as_a_level():
    spec = {"version": 1, "tf": "1d", "conditions": {"all": [
        {"lhs": {"op": "+", "args": [
            {"src": "close"},
            {"prim": "fvg_nearest", "direction": "long", "field": "top"},
        ]}, "op": "crosses_above", "rhs": 10},
    ]}}
    errors = validate_screen_spec(spec)
    assert any("anchored to the end" in error for error in errors)
    assert not any("must contain a technical series" in error for error in errors)


def test_screen_refuses_a_fact_outside_the_closed_vocabulary():
    spec = {"version": 1, "tf": "1d", "conditions": {"all": [
        {"lhs": {"fact": "beta"}, "op": ">", "rhs": 1},
    ]}}
    errors = validate_screen_spec(spec)
    assert any("unknown fact" in error for error in errors)


def test_validate_screen_spec_defaults_tf_to_1d():
    assert validate_screen_spec({"version": 1, "conditions": {"all": [RSI_LT_30]}}) == []


def test_validate_screen_spec_rejects_wrong_version():
    errs = validate_screen_spec({**GOOD, "version": 2})
    assert any("version must be 1" in e for e in errs)


def test_validate_screen_spec_rejects_bad_tf():
    errs = validate_screen_spec({**GOOD, "tf": "5m"})
    assert any("tf must be one of" in e for e in errs)


def test_validate_screen_spec_requires_conditions():
    errs = validate_screen_spec({"version": 1})
    assert any("conditions" in e for e in errs)


def test_validate_screen_spec_rejects_unknown_keys():
    errs = validate_screen_spec({**GOOD, "risk": {}})
    assert any("unknown keys" in e and "risk" in e for e in errs)


def test_validate_screen_spec_walks_the_grammar():
    bad = {"version": 1, "conditions": {"all": [{"lhs": {"ind": "nope"}, "op": "<", "rhs": 1}]}}
    errs = validate_screen_spec(bad)
    assert any(e.startswith("conditions.all[0]") for e in errs)


def test_a_daily_screen_may_not_reference_an_intraday_timeframe():
    """The default screen tf is 1d, so "daily screen plus one intraday
    reference" is the first shape a user reaches for and the one the evaluator
    cannot carry: a daily label has no close time to compare against. Refused
    per symbol at evaluation time it wrote an error note on every row and read
    as a screen that matched nothing."""
    spec = {"version": 1, "tf": "1d", "conditions": {"all": [
        {"lhs": {"src": "close"}, "op": ">",
         "rhs": {"ind": "sma", "n": 20, "tf": "1h"}}]}}
    errs = validate_screen_spec(spec)
    assert any("session-aligned" in e for e in errs), errs


def test_an_intraday_screen_may_reference_a_higher_timeframe():
    spec = {"version": 1, "tf": "15m", "conditions": {"all": [
        {"lhs": {"src": "close"}, "op": ">",
         "rhs": {"ind": "sma", "n": 20, "tf": "1d"}}]}}
    assert validate_screen_spec(spec) == []


@pytest.mark.parametrize("node", [
    {"prim": "custom_numeric", "n": 10 ** 400},
    {"prim": "custom_numeric"},
], ids=["explicit", "default"])
def test_screen_spec_rejects_numeric_injected_values_without_a_canonical_form(node):
    base = core_vocabulary()
    replacement = Term(
        "custom_numeric", "primitive",
        {"n": (-(10 ** 401), 10 ** 401)},
        {"n": 10 ** 400}, lambda *_args: None,
    )
    vocabulary = base.with_terms(replacement)
    spec = {"version": 1, "tf": "15m", "conditions": {"all": [
        {"lhs": {"src": "close"}, "op": ">", "rhs": node}]}}
    errs = validate_screen_spec(spec, vocabulary=vocabulary)
    assert len(errs) == 1, errs
    assert "custom_numeric.n has invalid argument rule" in errs[0]


@pytest.mark.parametrize("value", [9007199254740993, -9007199254740993],
                         ids=["positive", "negative"])
def test_screen_spec_rejects_integers_that_round_during_canonicalization(value):
    spec = {"version": 1, "tf": "15m", "conditions": {"all": [
        {"lhs": {"src": "close"}, "op": ">", "rhs": value}]}}
    errs = validate_screen_spec(spec)
    assert len(errs) == 1 and "number is out of range" in errs[0], errs


def test_screen_spec_refuses_a_required_injected_numeric_argument():
    vocabulary = core_vocabulary().with_terms(
        Term("required_numeric", "primitive", {"n": (1, 10)}, {},
             lambda *_args: None))
    spec = {"version": 1, "tf": "15m", "conditions": {"all": [
        {"lhs": {"src": "close"}, "op": ">",
         "rhs": {"prim": "required_numeric"}}]}}
    errs = validate_screen_spec(spec, vocabulary=vocabulary)
    assert len(errs) == 1 and "required_numeric needs n" in errs[0], errs


def test_screen_spec_refuses_an_invalid_injected_choice_default():
    vocabulary = core_vocabulary().with_terms(
        Term("invalid_choice", "primitive", {"side": ("long", "short")},
             {"side": "sideways"}, lambda *_args: None))
    spec = {"version": 1, "tf": "15m", "conditions": {"all": [
        {"lhs": {"src": "close"}, "op": ">",
         "rhs": {"prim": "invalid_choice"}}]}}
    errs = validate_screen_spec(spec, vocabulary=vocabulary)
    assert len(errs) == 1 and "invalid_choice.side default" in errs[0], errs


@pytest.mark.parametrize("bounds", [
    (-(10 ** 400), 10 ** 400),
    (float("-inf"), float("inf")),
], ids=["huge", "infinite"])
def test_screen_spec_rejects_invalid_injected_numeric_bounds(bounds):
    base = core_vocabulary()
    replacement = Term(
        "bad_bounds", "primitive", {"n": bounds}, {"n": 30},
        lambda *_args: None,
    )
    vocabulary = base.with_terms(replacement)
    spec = {"version": 1, "tf": "15m", "conditions": {"all": [
        {"lhs": {"src": "close"}, "op": ">",
         "rhs": {"prim": "bad_bounds"}}]}}
    errs = validate_screen_spec(spec, vocabulary=vocabulary)
    assert len(errs) == 1 and "invalid argument rule" in errs[0], errs


def test_validate_screen_spec_rejects_non_dict():
    assert validate_screen_spec([1]) == ["spec must be a JSON object"]


def test_describe_screen_renders_the_readback():
    text = describe_screen(GOOD)
    assert text.startswith("Screen on 1d bars")
    assert "rsi(14) is below 30" in text


def test_relative_scope_screen_description_matches_the_frozen_readback():
    assert validate_screen_spec(RELATIVE_SCOPE_SCREEN) == []
    assert describe_screen(RELATIVE_SCOPE_SCREEN) == (
        "Screen on 15m bars, matching symbols where ALL of:\n"
        "  - close is above SPY:close\n"
        "  - SPY:sma(20)[15m] is above "
        "SPY:sma(20, of=QQQ:close[1d])[15m]"
    )


def test_screen_reference_pairs_are_exact_instead_of_a_cross_product():
    assert screen_reference_pairs(RELATIVE_SCOPE_SCREEN) == (
        ("QQQ", "1d"), ("SPY", "15m"),
    )
    assert ("SPY", "1d") not in screen_reference_pairs(RELATIVE_SCOPE_SCREEN)
    assert ("QQQ", "15m") not in screen_reference_pairs(RELATIVE_SCOPE_SCREEN)


def test_windowed_screen_readback_discloses_only_low_iex_rows():
    london = _screen_with(
        {"ind": "highest", "of": {"src": "high"}, "window": "london"})
    standard = _screen_with(
        {"ind": "highest", "of": {"src": "high"}, "window": "ny_am"})

    london_text = describe_screen(london, vocabulary=WINDOW_VOCABULARY)
    assert "highest(of=high) over london" in london_text
    assert LOW_IEX_DISCLOSURE in london_text

    standard_text = describe_screen(standard, vocabulary=WINDOW_VOCABULARY)
    assert "highest(of=high) over ny_am" in standard_text
    assert LOW_IEX_DISCLOSURE not in standard_text


def test_the_screen_surface_reads_one_vocabulary_end_to_end():
    """Prompt, validator, and readback must agree on the term list.

    Threading only the prompt would advertise a term to the model that the
    validator then refuses, spending every retry on an error the caller caused.
    """
    vocab = core_vocabulary().with_terms(
        Term("double_close", "series", {}, {}, lambda s, _a: s * 2,
             doc="twice the input series"))
    spec = {"version": 1, "tf": "1d", "conditions": {"all": [
        {"lhs": {"ind": "double_close"}, "op": ">", "rhs": 0}]}}
    assert "- double_close(no args)" in render_screen_prompt(vocab)
    assert validate_screen_spec(spec, vocabulary=vocab) == []
    assert "double_close is above 0" in describe_screen(spec, vocabulary=vocab)
    # The default surface refuses it by name rather than accepting it, so a
    # missed injection is findable instead of silent.
    errs = validate_screen_spec(spec)
    assert errs and "unknown indicator 'double_close'" in errs[0]


def test_referenced_timeframes_collects_base_and_node_tfs():
    spec = {"version": 1, "tf": "1h", "conditions": {"all": [
        {"lhs": {"src": "close", "tf": "1d"}, "op": ">", "rhs": {"ind": "sma", "n": 50}}]}}
    assert referenced_timeframes(spec) == {"1h", "1d"}


def test_is_intraday():
    assert not is_intraday(GOOD)
    assert is_intraday({**GOOD, "tf": "15m"})


def test_max_lookback_finds_the_longest_window():
    spec = {"version": 1, "conditions": {"all": [
        {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 200}},
        {"lhs": {"ind": "macd", "fast": 12, "slow": 26, "signal": 9, "field": "macd"},
         "op": ">", "rhs": 0}]}}
    assert max_lookback(spec) == 200
    assert max_lookback(GOOD) == 20  # floor


def test_screen_spec_inherits_not_in_validation_and_readback():
    spec = {"version": 1, "tf": "1d",
            "conditions": {"not": {"any": [RSI_LT_30]}}}
    assert validate_screen_spec(spec) == []
    assert describe_screen(spec) == (
        "Screen on 1d bars, matching symbols where NOT ANY of:\n"
        "  - rsi(14) is below 30")


def _section(prompt: str, header: str) -> str:
    """One "#" section of the rendered screen prompt, its header line included.

    Same reason as its twin in test_rules_vocabulary.py: an unscoped substring
    search over a prompt this large passes for the wrong reason routinely.
    """
    return prompt.split(header, 1)[1].split("\n#", 1)[0]


def test_screen_prompt_renders_a_condition_typed_arg_readably():
    prompt = render_screen_prompt()
    assert "- bars_since(cond={lhs,op,rhs})" in prompt.splitlines()
    assert "cond=condition" not in prompt


def test_screen_prompt_renders_window_rows_from_the_supplied_vocabulary():
    prompt = render_screen_prompt(WINDOW_VOCABULARY)
    lines = prompt.splitlines()
    london = next(line for line in lines if line.startswith("- london:"))
    assert london == (
        "- london: timezone=Europe/London; span=[08:00, 16:30); "
        "recurrence=weekday; confidence=low_iex. " + LOW_IEX_DISCLOSURE)
    ny_am = next(line for line in lines if line.startswith("- ny_am:"))
    assert ny_am == (
        "- ny_am: timezone=America/New_York; span=[09:30, 12:00); "
        "recurrence=xnys_session; confidence=standard")
    assert LOW_IEX_DISCLOSURE not in ny_am
    assert '"window"?: <registered window>' in prompt
    assert "- first(no args) [takes of=<expr>] [window required; reducer=first]" in lines


def test_screen_prompt_describes_a_second_condition_taking_term_generically(
        count_where_vocab):
    prompt = render_screen_prompt(count_where_vocab)
    assert "- count_where(when={lhs,op,rhs}, n=(1, 50))" in prompt.splitlines()


def test_the_screen_primitives_header_states_the_prohibition_generically():
    header = _section(render_screen_prompt(), "# Primitives").split("\n- ", 1)[0]
    assert "bars_since" not in header
    assert "cross" in header


def test_screen_prompt_documents_not_in_the_grammar_paragraph():
    schema = _section(render_screen_prompt(), "# Schema")
    assert '{"not":' in schema
    assert '{"not": {"all":' in schema
