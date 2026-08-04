"""ScreenSpec v1: the conditions-only IR reusing RuleSpec v2's grammar."""

from nakagai.screen.prompt import render_screen_prompt
from nakagai.strategies.rules.spec import group_text, validate_condition_group
from nakagai.strategies.rules.vocabulary import Term, core_vocabulary

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
