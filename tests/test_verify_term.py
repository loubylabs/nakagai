# tests/test_verify_term.py
"""The per-term causality gate, and its honesty about what it cannot test."""

import numpy as np
import pandas as pd
import pytest

import nakagai.strategies.rules.verify as verify_module
from nakagai.strategies.rules.vocabulary import Term, core_vocabulary
from nakagai.strategies.rules.verify import (
    CONDITION_ARG, MAX_ARG_SETS, TermVerdict, arg_sets, evaluate_term,
    exemption_reason, field_mismatch,
)


def test_an_end_anchored_term_is_exempt_and_says_why():
    v = core_vocabulary()
    reason = exemption_reason(v.primitives["fvg_nearest"])
    assert reason is not None
    assert "end_anchored" in reason


def test_a_condition_taking_term_is_exempt_and_says_why():
    v = core_vocabulary()
    reason = exemption_reason(v.primitives["bars_since"])
    assert reason is not None
    assert "condition" in reason


def test_an_ordinary_term_is_not_exempt():
    v = core_vocabulary()
    assert exemption_reason(v.indicators["sma"]) is None
    assert exemption_reason(v.indicators["macd"]) is None
    assert exemption_reason(v.primitives["gap_pct"]) is None


def test_exactly_these_core_terms_are_exempt():
    """The exempt set is a declared constant, not whatever the code happens to skip.

    A term becoming silently exempt is how a real look-ahead bug would hide, so
    the set is asserted rather than counted.
    """
    exempt = {t.name for t in core_vocabulary().all_terms()
              if exemption_reason(t) is not None}
    assert exempt == {"fvg_nearest", "order_block", "bars_since"}


SESSIONS = 160
BARS_PER_SESSION = 26
EXCHANGE_TZ = "America/New_York"


@pytest.fixture(scope="module")
def bars():
    """Multi-session RTH-shaped 15m bars: 26 a day, 160 weekdays, no weekends.

    160 sessions rather than a round 40 because the vocabulary's widest range rule
    is rvol's `sessions: (5, 60)`, and a mandated argument set that is NaN at every
    probe row proves nothing about the term. Each bar opens at the previous close
    so bodies take both signs, which keeps order_block and any close-against-open
    condition from being constant. Seeded, so the gate's own result is reproducible.

    ANCHORED IN EXCHANGE-LOCAL TIME, not at a fixed UTC hour. A frame pinned to
    14:30 UTC is the 09:30 bell only until daylight saving moves, and 160 sessions
    from January crosses that boundary in March. Measured: the UTC-pinned version
    leaves opening_range_high and opening_range_low NaN at every probe row, because
    the bars no longer start at the open; this version checks all 34 non-exempt
    terms.
    """
    rng = np.random.default_rng(19)
    days = pd.bdate_range("2026-01-05", periods=SESSIONS, tz=EXCHANGE_TZ)
    stamps = [d + pd.Timedelta(hours=9, minutes=30) + i * pd.Timedelta(minutes=15)
              for d in days for i in range(BARS_PER_SESSION)]
    idx = pd.DatetimeIndex(stamps).tz_convert("UTC")
    idx.name = "ts"
    n = len(idx)
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    open_ = np.concatenate([[close[0] - 0.05], close[:-1]])
    return pd.DataFrame(
        {"open": open_,
         "high": np.maximum(open_, close) + np.abs(rng.normal(0, 0.15, n)),
         "low": np.minimum(open_, close) - np.abs(rng.normal(0, 0.15, n)),
         "close": close,
         "volume": 1000.0 + rng.integers(0, 500, n)},
        index=idx)


def test_the_fixture_is_session_shaped_with_two_sided_bodies(bars):
    """A flat fixture empties several terms of content and reads as coverage.

    The row count and the two-sided bodies are not enough on their own: a
    continuous 24h index of the same length passes both while giving the five
    session-scoped primitives a session shape no production frame has. So this
    reads the index.
    """
    assert len(bars) == SESSIONS * BARS_PER_SESSION
    bodies = bars["close"] - bars["open"]
    assert (bodies > 0).any() and (bodies < 0).any()

    local = bars.index.tz_convert(EXCHANGE_TZ)
    assert len(set(local.date)) == SESSIONS, "index is not one block per session"
    minutes = local.hour * 60 + local.minute
    assert minutes.min() == 9 * 60 + 30, "sessions do not start at the bell"
    assert minutes.max() < 16 * 60, "bars fall outside regular hours"


def test_the_fixture_survives_the_daylight_saving_boundary(bars):
    """The bug this fixture shape exists to avoid, asserted directly.

    160 weekdays from 2026-01-05 crosses the March transition, so a fixed UTC
    offset would put the later sessions an hour off the bell. Every session must
    open at 09:30 local on both sides of it.
    """
    local = bars.index.tz_convert(EXCHANGE_TZ)
    opens = {d: None for d in sorted(set(local.date))}
    for ts in local:
        if opens[ts.date()] is None:
            opens[ts.date()] = ts
    assert {(t.hour, t.minute) for t in opens.values()} == {(9, 30)}
    assert len({t.utcoffset() for t in opens.values()}) == 2, (
        "the fixture never crosses a DST boundary, so it cannot prove this")


def test_the_fixture_outlasts_the_widest_range_rule_in_the_vocabulary(bars):
    """The widest session-denominated bound in the vocabulary, found by search.

    Searched by ARG NAME, not by term name. Filtering on term.name == "rvol"
    would read rvol's own bound back out and pass forever, including for a term
    added later that declares a wider sessions range; this reddens instead, and
    the fixture then has to grow. Today the search finds exactly one arg.

    Bars-denominated bounds are excluded deliberately rather than overlooked.
    SESSIONS counts days and highest's n: (2, 500) counts bars, so folding them
    together would demand a 1000-session fixture to clear a 500-bar lookback
    that 160 sessions of 26 bars already covers four times over.
    """
    sessions_bounds = {(term.name, name): rule[1]
                       for term in core_vocabulary().all_terms()
                       for name, rule in term.args.items()
                       if name == "sessions" and isinstance(rule, tuple)
                       and len(rule) == 2
                       and all(isinstance(x, (int, float))
                               and not isinstance(x, bool) for x in rule)}
    assert sessions_bounds, "no session-denominated range rule found to size against"
    widest = max(sessions_bounds.values())
    assert widest == 60, f"the widest sessions bound moved to {widest}: {sessions_bounds}"
    assert SESSIONS > widest * 2, "probes must sit well past the widest session window"


def test_evaluate_term_handles_a_series_term(bars):
    out = evaluate_term(core_vocabulary().indicators["sma"], bars, {"n": 20})
    assert isinstance(out, pd.Series) and len(out) == len(bars)
    assert not pd.isna(out.iloc[-1])


def test_evaluate_term_selects_the_field_of_a_frame_term(bars):
    term = core_vocabulary().indicators["macd"]
    args = {"fast": 12, "slow": 26, "signal": 9, "field": "hist"}
    hist = evaluate_term(term, bars, args)
    signal = evaluate_term(term, bars, {**args, "field": "signal"})
    assert isinstance(hist, pd.Series)
    assert not hist.equals(signal), "field selection is not happening"


def test_evaluate_term_selects_the_field_of_a_bar_term(bars):
    """bar-kind terms return DataFrames too, so field selection is not frame-only."""
    term = core_vocabulary().indicators["donchian"]
    args = {"n": 20, "field": "upper"}
    upper = evaluate_term(term, bars, args)
    lower = evaluate_term(term, bars, {**args, "field": "lower"})
    assert (upper.dropna() >= lower.dropna()).all()


def test_evaluate_term_handles_a_bar_term_returning_one_series(bars):
    out = evaluate_term(core_vocabulary().indicators["atr"], bars, {"n": 14})
    assert isinstance(out, pd.Series) and (out.dropna() > 0).all()


def test_evaluate_term_handles_a_primitive(bars):
    out = evaluate_term(core_vocabulary().primitives["gap_pct"], bars, {})
    assert isinstance(out, pd.Series) and len(out) == len(bars)


def test_every_core_multi_output_term_declares_the_columns_it_produces(bars):
    """Schema against reality, for the terms core wrote. All seven agree today.

    Nine terms declare a `field`; seven of them return a DataFrame. The other
    two, fvg_nearest and order_block, select their field inside the function
    and return a float, so field_mismatch is a no-op on them. They stay in the
    loop rather than being filtered out, because narrowing this to indicators
    would let a future DataFrame-returning primitive skip the check entirely.
    """
    for term in core_vocabulary().all_terms():
        if not term.args.get("field"):
            continue
        # The raw callable, without evaluate_term's field narrowing, because the
        # question is which columns exist before one of them is selected. The
        # three-way split is evaluate_term's: a primitive called down the series
        # branch binds the defaults dict to `bars` and crashes inside find_fvgs.
        if term.kind == "primitive":
            raw = term.fn(None, bars, **dict(term.defaults))
        elif term.kind == "bar":
            raw = term.fn(bars, dict(term.defaults))
        else:
            raw = term.fn(bars["close"], dict(term.defaults))
        assert field_mismatch(term, raw) is None, term.name


def test_a_term_that_under_declares_its_fields_is_caught(bars):
    """The guard above passes on arrival, so prove it can fail.

    A generated `ta` signature that declares three fields and returns four leaves
    the fourth never evaluated, which is invisible from the schema.
    """
    term = Term("under_declared", "frame", {"field": ("a",)}, {"field": "a"},
                lambda s, a: pd.DataFrame({"a": s, "b": s.shift(-1)}))
    raw = term.fn(bars["close"], {"field": "a"})
    reason = field_mismatch(term, raw)
    assert reason is not None and "b" in reason


def test_arg_sets_crosses_every_choice_combination():
    """A peek reachable only at one combination of choices must be reachable here."""
    term = Term("two_choices", "series",
                {"mode": ("safe", "fast"), "offset": (0.0, 1.0),
                 "level": ("a", "b")},
                {"mode": "safe", "offset": 0.0, "level": "a"},
                lambda s, a: s)
    combos = {(s["mode"], s["level"]) for s in arg_sets(term)}
    assert combos == {("safe", "a"), ("safe", "b"), ("fast", "a"), ("fast", "b")}


def test_arg_sets_varies_range_endpoints_against_every_choice_combination():
    term = core_vocabulary().indicators["macd"]
    sets = arg_sets(term)
    for field in ("macd", "signal", "hist"):
        fasts = {s["fast"] for s in sets if s["field"] == field}
        assert {2, 100} <= fasts, f"{field}: range endpoints not varied under it"


def test_arg_sets_covers_defaults_every_choice_and_both_range_ends():
    term = core_vocabulary().indicators["macd"]
    sets = arg_sets(term)
    assert dict(term.defaults) in sets
    assert {s["field"] for s in sets} == {"macd", "signal", "hist"}


def test_arg_sets_covers_every_field_of_every_multi_output_term():
    for term in core_vocabulary().all_terms():
        want = set(term.args.get("field", ()))
        if not want:
            continue
        got = {s["field"] for s in arg_sets(term)}
        assert got >= want, f"{term.name}: fields {sorted(want - got)} not covered"


def test_arg_sets_is_deduplicated_and_stable():
    term = core_vocabulary().indicators["sma"]
    sets = arg_sets(term)
    assert len(sets) == len({tuple(sorted(s.items())) for s in sets})
    assert arg_sets(term) == sets


def test_arg_sets_skips_a_condition_arg():
    term = core_vocabulary().primitives["bars_since"]
    assert all("cond" not in s for s in arg_sets(term))


def test_arg_sets_of_a_term_with_no_args_is_one_empty_set():
    assert arg_sets(core_vocabulary().primitives["gap_pct"]) == ({},)


def test_no_core_term_exceeds_the_argument_set_cap():
    worst = max(core_vocabulary().all_terms(), key=lambda t: len(arg_sets(t)))
    assert len(arg_sets(worst)) <= MAX_ARG_SETS, worst.name


def test_a_term_over_the_cap_is_refused_loudly_not_sampled():
    """No silent caps: a bounded result that reads as complete is the failure."""
    term = Term("too_wide", "series",
                {f"c{i}": ("x", "y", "z") for i in range(6)},
                {f"c{i}": "x" for i in range(6)},
                lambda s, a: s)
    with pytest.raises(ValueError, match="argument sets"):
        arg_sets(term)


def test_the_cap_counts_what_the_term_produces_not_the_formula(monkeypatch):
    """A default sitting on its own range endpoint must not cost a refusal.

    The combinatorial formula says 2 choices times (1 + 2 ranges) = 6, but
    offset's default IS its own lower endpoint, so two of those six collapse and
    the term really produces 4. Counting the formula would refuse this term at a
    cap of 4 or 5 for sets it never generates. Node 02 generates terms from
    another library's signatures, where a default on a bound is ordinary.
    """
    term = Term("default_on_a_bound", "series",
                {"mode": ("safe", "fast"), "offset": (0.0, 1.0)},
                {"mode": "safe", "offset": 0.0},
                lambda s, a: s)
    assert len(arg_sets(term)) == 4

    monkeypatch.setattr(verify_module, "MAX_ARG_SETS", 4)
    assert len(arg_sets(term)) == 4, "the real count fits the cap and must pass"

    monkeypatch.setattr(verify_module, "MAX_ARG_SETS", 3)
    with pytest.raises(ValueError, match="more than 3"):
        arg_sets(term)
