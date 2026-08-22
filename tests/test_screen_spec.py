"""ScreenSpec v1: the conditions-only IR reusing RuleSpec v2's grammar."""

import pytest
import numpy as np

from nakagai.screen.prompt import render_screen_prompt
from nakagai.strategies.rules.spec import group_text, validate_condition_group
from nakagai.strategies.rules.vocabulary import (
    Term, Vocabulary, core_vocabulary,
)

RSI_LT_30 = {"lhs": {"ind": "rsi", "n": 14}, "op": "<", "rhs": 30}


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
    validate_screen_spec,
)

GOOD = {"version": 1, "tf": "1d", "conditions": {"all": [RSI_LT_30]}}


def test_validate_screen_spec_accepts_a_good_spec():
    assert validate_screen_spec(GOOD) == []


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


def test_screen_spec_refuses_an_opening_range_shorter_than_its_bar():
    spec = {"version": 1, "tf": "1h", "conditions": {"all": [
        {"lhs": {"src": "close"}, "op": ">",
         "rhs": {"prim": "opening_range_high", "minutes": 30}}]}}
    errs = validate_screen_spec(spec)
    assert any("opening_range_high" in error and "30-minute" in error
               and "'1h'" in error and "60 minutes" in error
               for error in errs), errs


@pytest.mark.parametrize("minutes", [
    1, 121, float("nan"), float("inf"), float("-inf"), True, "30", None,
    10 ** 400, -(10 ** 400),
])
def test_screen_spec_reports_invalid_opening_range_minutes_only_once(minutes):
    spec = {"version": 1, "tf": "1h", "conditions": {"all": [
        {"lhs": {"src": "close"}, "op": ">",
         "rhs": {"prim": "opening_range_high", "minutes": minutes}}]}}
    errs = validate_screen_spec(spec)
    assert len(errs) == 1, errs
    assert "opening_range_high.minutes must be a number in [5, 120]" in errs[0]


def test_screen_spec_rejects_a_huge_injected_opening_range_default():
    base = core_vocabulary()
    huge = -(10 ** 400)
    replacement = Term(
        "opening_range_high", "primitive",
        {"minutes": (-(10 ** 401), 10 ** 401)},
        {"minutes": huge}, lambda *_args: None,
    )
    vocabulary = Vocabulary(
        base.indicators,
        {**base.primitives, "opening_range_high": replacement},
    )
    spec = {"version": 1, "tf": "1h", "conditions": {"all": [
        {"lhs": {"src": "close"}, "op": ">",
         "rhs": {"prim": "opening_range_high"}}]}}
    errs = validate_screen_spec(spec, vocabulary=vocabulary)
    assert len(errs) == 1, errs
    assert "opening_range_high.minutes has invalid argument rule" in errs[0]


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


def test_screen_spec_refuses_a_wide_hourly_bar_with_numpy_opening_bounds():
    base = core_vocabulary()
    replacement = Term(
        "opening_range_high", "primitive",
        {"minutes": (np.int64(5), np.int64(120))}, {"minutes": 30},
        lambda *_args: None,
    )
    vocabulary = Vocabulary(
        base.indicators,
        {**base.primitives, "opening_range_high": replacement},
    )
    spec = {"version": 1, "tf": "1h", "conditions": {"all": [
        {"lhs": {"src": "close"}, "op": ">",
         "rhs": {"prim": "opening_range_high"}}]}}
    errs = validate_screen_spec(spec, vocabulary=vocabulary)
    assert any("opening_range_high" in error and "30-minute" in error
               and "60 minutes" in error for error in errs), errs


@pytest.mark.parametrize("node", [
    {"prim": "opening_range_high", "minutes": np.int64(30)},
    {"prim": "opening_range_high"},
], ids=["explicit", "default"])
def test_screen_spec_rejects_numpy_opening_range_values(node):
    base = core_vocabulary()
    replacement = Term(
        "opening_range_high", "primitive", {"minutes": (5, 120)},
        {"minutes": np.int64(30)}, lambda *_args: None,
    )
    vocabulary = Vocabulary(
        base.indicators,
        {**base.primitives, "opening_range_high": replacement},
    )
    spec = {"version": 1, "tf": "15m", "conditions": {"all": [
        {"lhs": {"src": "close"}, "op": ">", "rhs": node}]}}
    errs = validate_screen_spec(spec, vocabulary=vocabulary)
    assert len(errs) == 1, errs
    assert "opening_range_high.minutes must be a number" in errs[0]


@pytest.mark.parametrize("node", [
    {"prim": "opening_range_high", "minutes": np.float64(30.0)},
    {"prim": "opening_range_high"},
], ids=["explicit-float", "default-float"])
def test_screen_spec_rejects_numpy_float_opening_range_values(node):
    base = core_vocabulary()
    replacement = Term(
        "opening_range_high", "primitive", {"minutes": (5, 120)},
        {"minutes": np.float64(30.0)}, lambda *_args: None,
    )
    vocabulary = Vocabulary(
        base.indicators,
        {**base.primitives, "opening_range_high": replacement},
    )
    spec = {"version": 1, "tf": "15m", "conditions": {"all": [
        {"lhs": {"src": "close"}, "op": ">", "rhs": node}]}}
    errs = validate_screen_spec(spec, vocabulary=vocabulary)
    assert len(errs) == 1, errs
    assert "opening_range_high.minutes must be a number" in errs[0]


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
