# tests/test_verify_term.py
"""The per-term causality gate, and its honesty about what it cannot test."""

import numpy as np
import pandas as pd
import pytest

import nakagai.strategies.rules.verify as verify_module
from nakagai.strategies.rules.vocabulary import Term, core_vocabulary
from nakagai.strategies.rules.verify import (
    CHECKED, CONDITION_ARG, EXEMPT, FAILED, MAX_ARG_SETS, PROBE_COUNT,
    TermVerdict, VACUOUS, arg_sets, evaluate_term, exemption_reason,
    field_mismatch, probe_rows, verify_term, verify_vocabulary,
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


def test_arg_sets_refuses_a_bare_string_rule_rather_than_indexing_its_letters():
    """Measured: {"period": "lookback"} enumerated {'period': 'l'} and {'period': 'o'}.

    The partition asked "is this a choice rule" and treated everything else as a
    range, then read rule[0] and rule[1] with no shape check, so a bare string
    became a pair of bounds one letter each.
    """
    term = Term("stringy_rule", "series", {"period": "lookback"}, {},
                lambda s, a: s)
    with pytest.raises(ValueError, match="period"):
        arg_sets(term)


def test_arg_sets_refuses_a_list_of_choices_rather_than_testing_two_of_three(bars):
    """The dangerous shape: three branches declared, two called, verdict CHECKED.

    Measured: {"mode": ["x", "y", "z"]} enumerated {'mode': 'x'} and
    {'mode': 'y'} as if the list were a pair of bounds, so 'z' was never called
    while arg_sets_checked read 2. That is the bounded result reading as
    complete that the cap refuses, arriving through a different door. ArgRule is
    documented as a tuple, but node 02's terms come from a generator, and
    catching a generator that got the schema shape wrong is what this gate is
    for.
    """
    term = Term("listy_rule", "series", {"mode": ["x", "y", "z"]},
                {"mode": "x"}, lambda s, a: s)
    with pytest.raises(ValueError, match="mode"):
        arg_sets(term)

    verdict = verify_term(term, bars)
    assert verdict.status == FAILED, "an unusable schema is a rejection, not a crash"
    assert verdict.cause == "unenumerable"


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


def _peeking_series(s, a):
    """Reads one row into the future. This is the defect the gate exists for."""
    return s.shift(-1)


def test_probe_rows_sit_past_the_warmup_and_are_bounded_in_count(bars):
    rows = probe_rows(len(bars))
    assert len(rows) == PROBE_COUNT
    assert min(rows) >= len(bars) // 2
    assert max(rows) < len(bars)


def test_a_peeking_term_fails_the_gate(bars):
    """The gate's own falsification test. Watch this fail before trusting a pass."""
    verdict = verify_term(Term("peeker", "series", {}, {}, _peeking_series), bars)
    assert verdict.status == FAILED
    assert verdict.reason, "a failure must say which row disagreed"


def test_an_honest_term_is_checked(bars):
    verdict = verify_term(core_vocabulary().indicators["sma"], bars)
    assert verdict.status == CHECKED
    assert verdict.arg_sets_checked == len(arg_sets(core_vocabulary().indicators["sma"]))


def test_every_mandated_argument_set_must_be_non_vacuous(bars):
    """One all-NaN argument set makes the TERM vacuous, not just that set.

    Both plan-gate lenses caught the opposite: rvol's mandated sessions=60 was NaN
    at every probe on a short fixture while the term reported CHECKED, so the
    boundary went untested behind a green run.
    """
    short = bars.iloc[:40 * 26]
    verdict = verify_term(core_vocabulary().primitives["rvol"], short)
    assert verdict.status == VACUOUS
    assert "sessions" in verdict.reason and "60" in verdict.reason


def test_rvol_is_checked_on_the_full_fixture(bars):
    """The same term on the properly sized fixture. This is why it is 160 sessions."""
    verdict = verify_term(core_vocabulary().primitives["rvol"], bars)
    assert verdict.status == CHECKED, verdict.reason


def test_an_all_nan_term_is_vacuous_not_checked(bars):
    term = Term("always_nan", "series", {}, {},
                lambda s, a: pd.Series(float("nan"), index=s.index))
    assert verify_term(term, bars).status == VACUOUS


def test_a_term_that_raises_is_failed_not_swallowed(bars):
    term = Term("explodes", "series", {}, {},
                lambda s, a: (_ for _ in ()).throw(ValueError("boom")))
    verdict = verify_term(term, bars)
    assert verdict.status == FAILED and "boom" in verdict.reason


def test_a_term_the_gate_cannot_enumerate_is_a_verdict_not_a_crash(bars):
    """Over the cap is a rejection, not an exception out of verify_term.

    arg_sets raises past MAX_ARG_SETS on purpose. If that propagates, node 02's
    batch of 100-plus terms dies on the first wide schema instead of reporting
    one refused term and carrying on, and the gate stops being able to say
    anything about the other 99. The spec is explicit: a term whose signature
    the gate cannot enumerate is a rejection, not a pass, and a crash is
    neither.
    """
    term = Term("too_wide_to_enumerate", "series",
                {f"c{i}": ("x", "y", "z") for i in range(6)},
                {f"c{i}": "x" for i in range(6)},
                lambda s, a: s)
    verdict = verify_term(term, bars)
    assert verdict.status == FAILED
    assert "enumerate" in verdict.reason
    assert str(MAX_ARG_SETS) in verdict.reason, "the reason must name the cap"


def test_a_term_whose_schema_generates_no_argument_sets_is_failed(bars):
    """CHECKED after zero calls is the one verdict that must never be reachable.

    `is_choice_rule(())` is True, because all() over an empty tuple is True, so
    an enum arg that resolves to nothing partitions as a choice, the cross
    product is empty, and the probe loop never runs. The body here reads one row
    into the future, so CHECKED would admit a peeking term the gate never called.
    Node 02 generates schemas from another library's signatures, where an enum
    arg resolving empty is ordinary rather than exotic.
    """
    term = Term("empty_choice", "series", {"mode": ()}, {},
                lambda s, a: s.shift(-1))
    assert arg_sets(term) == (), "the shape this test is about has changed"

    verdict = verify_term(term, bars)
    assert verdict.status == FAILED
    assert verdict.reason, "a refusal must say why nothing was called"
    assert verdict.arg_sets_checked == 0


# The four term shapes that escaped verify_term as tracebacks, one test each.
# Crash safety was applied to the whole-frame call and the prefix call and to
# nothing between them, so the narrowing, the field lookup and the value
# extraction on the whole-frame side were unguarded while the identical
# extraction on the prefix side was guarded. A traceback out of verify_term
# takes down a whole node 02 batch of 100-plus terms over one malformed one,
# which is the outcome the module's own words at the enumeration step refuse.


def test_a_term_returning_a_frame_with_no_field_declared_is_a_verdict(bars):
    """Escaped as KeyError 'field' from the narrowing step.

    Declaring no `field` while returning a DataFrame passes field_mismatch,
    because nothing declared equals nothing produced, and then there is no
    field to narrow by.
    """
    term = Term("empty_frame", "series", {}, {}, lambda s, a: pd.DataFrame())
    verdict = verify_term(term, bars)
    assert verdict.status == FAILED
    assert verdict.cause == "gate_error"
    assert "KeyError" in verdict.reason


def test_a_term_returning_a_non_numeric_series_is_a_verdict(bars):
    """Escaped as ValueError: could not convert string to float."""
    term = Term("stringly", "series", {}, {},
                lambda s, a: pd.Series("x", index=s.index))
    verdict = verify_term(term, bars)
    assert verdict.status == FAILED
    assert verdict.cause == "gate_error"
    assert "ValueError" in verdict.reason


def test_a_term_returning_fewer_rows_than_the_frame_is_a_shape_failure(bars):
    """Escaped as IndexError: single positional indexer is out-of-bounds.

    This one is not reported as gate_error. A returned Series that does not line
    up with the frame cannot be probed by position at all, and if the lengths
    happened to line up far enough to index, the mismatch would read as "row i
    read a row after itself" and blame a term for a peek it never made. So the
    length is checked before probing and reported as the shape problem it is.
    """
    term = Term("truncated", "series", {}, {}, lambda s, a: s.iloc[:10])
    verdict = verify_term(term, bars)
    assert verdict.status == FAILED
    assert verdict.cause == "schema"
    assert "10" in verdict.reason and str(len(bars)) in verdict.reason
    assert "read a row after itself" not in verdict.reason


def test_a_term_returning_duplicate_column_names_is_a_verdict(bars):
    """Escaped as TypeError: float() argument must be ... not 'DataFrame'.

    Two columns named "a" satisfy field_mismatch, which compares sets, and then
    narrowing by "a" returns a DataFrame rather than a Series.
    """
    term = Term("duplicated", "frame", {"field": ("a",)}, {"field": "a"},
                lambda s, a: pd.concat([s.rename("a"), s.rename("a")], axis=1))
    verdict = verify_term(term, bars)
    assert verdict.status == FAILED
    assert verdict.cause == "gate_error"
    assert "TypeError" in verdict.reason


def test_a_term_whose_schema_disagrees_with_its_columns_is_failed(bars):
    term = Term("under_declared", "frame", {"field": ("a",)}, {"field": "a"},
                lambda s, a: pd.DataFrame({"a": s, "b": s.shift(-1)}))
    verdict = verify_term(term, bars)
    assert verdict.status == FAILED
    assert "b" in verdict.reason


def _peek_under_late(s, a):
    return s.shift(-1) if a["mode"] == "late" else s


def _nan_under_late(s, a):
    return pd.Series(float("nan"), index=s.index) if a["mode"] == "late" else s


def test_a_failed_verdict_carries_a_machine_readable_cause(bars):
    """FAILED is one bucket for five conditions, and only one of them is a peek.

    This is the module's own thesis one level down: if a bare boolean cannot
    tell proved-causal from could-not-test, a bare FAILED cannot tell
    this-term-peeks from our-gate-broke. Node 02 lists rejected terms in CI
    output and has to classify them, and string-matching prose written for a
    human is not a classification. Four causes are reachable from outside the
    gate; "gate_error" is asserted by the crash-safety tests below.
    """
    peeker = Term("peeker", "series", {}, {}, _peeking_series)
    assert verify_term(peeker, bars).cause == "lookahead"

    explodes = Term("explodes", "series", {}, {},
                    lambda s, a: (_ for _ in ()).throw(ValueError("boom")))
    assert verify_term(explodes, bars).cause == "uncallable"

    under_declared = Term("under_declared", "frame", {"field": ("a",)},
                          {"field": "a"},
                          lambda s, a: pd.DataFrame({"a": s, "b": s.shift(-1)}))
    assert verify_term(under_declared, bars).cause == "schema"

    too_wide = Term("too_wide", "series",
                    {f"c{i}": ("x", "y", "z") for i in range(6)},
                    {f"c{i}": "x" for i in range(6)}, lambda s, a: s)
    assert verify_term(too_wide, bars).cause == "unenumerable"

    empty_choice = Term("empty_choice", "series", {"mode": ()}, {},
                        _peeking_series)
    assert verify_term(empty_choice, bars).cause == "unenumerable"


def test_a_verdict_that_is_not_a_failure_carries_no_cause(bars):
    """An empty cause is what "this is not a rejection" reads as downstream."""
    checked = verify_term(core_vocabulary().indicators["sma"], bars)
    assert checked.status == CHECKED and checked.cause == ""

    exempt = verify_term(core_vocabulary().primitives["bars_since"], bars)
    assert exempt.status == EXEMPT and exempt.cause == ""

    vacuous = verify_term(Term("always_nan", "series", {}, {},
                               lambda s, a: pd.Series(float("nan"), index=s.index)),
                          bars)
    assert vacuous.status == VACUOUS and vacuous.cause == ""


def test_a_rejection_reports_how_many_argument_sets_already_passed(bars):
    """arg_sets_checked is a factual claim, and 0 is false when N-1 sets passed.

    Both terms here are honest under `mode="early"` and broken under
    `mode="late"`, which is the second set enumerated. "failed on set 1 of 2" is
    what node 02 needs to print; a bare 0 says the gate got nowhere, which is
    not what happened.
    """
    peeks_late = Term("peeks_late", "series", {"mode": ("early", "late")},
                      {"mode": "early"}, _peek_under_late)
    verdict = verify_term(peeks_late, bars)
    assert verdict.status == FAILED and verdict.cause == "lookahead"
    assert verdict.arg_sets_checked == 1

    empty_late = Term("empty_late", "series", {"mode": ("early", "late")},
                      {"mode": "early"}, _nan_under_late)
    verdict = verify_term(empty_late, bars)
    assert verdict.status == VACUOUS
    assert verdict.arg_sets_checked == 1


@pytest.mark.parametrize(
    "name", ["sma", "ema", "rsi", "macd", "bb", "atr", "donchian", "vwap",
             "ichimoku", "stoch", "supertrend", "keltner", "obv", "adx"])
def test_every_shipped_indicator_shape_is_checked_and_passes(name, bars):
    verdict = verify_term(core_vocabulary().indicators[name], bars)
    assert verdict.status == CHECKED, f"{name}: {verdict.reason}"


@pytest.mark.parametrize("name", ["fvg_nearest", "order_block"])
def test_an_honest_end_anchored_term_would_fail_without_the_exemption(name, bars, monkeypatch):
    """The exemption is load-bearing: prove what happens when it is not there.

    These two terms are honest. The gate reports them FAILED with the exemption
    removed, because term.fn returns a float for a frame and the comparison is
    therefore not a causality test at all. That is the false failure the exemption
    exists to prevent, and asserting it is what makes the exemption more than a
    comment.
    """
    term = core_vocabulary().primitives[name]
    assert verify_term(term, bars).status == EXEMPT

    monkeypatch.setattr(verify_module, "exemption_reason", lambda _term: None)
    assert verify_term(term, bars).status == FAILED


def test_an_end_anchored_term_returns_a_scalar_not_a_series(bars):
    """The structural fact underneath the exemption, asserted rather than assumed."""
    term = core_vocabulary().primitives["fvg_nearest"]
    out = term.fn(None, bars, **dict(term.defaults))
    assert isinstance(out, float)
    assert not isinstance(out, pd.Series)


def test_an_exempt_term_is_never_reported_as_checked_or_failed(bars):
    """One assertion that can fail on its own, not a restatement of EXEMPT."""
    statuses = {name: verify_term(core_vocabulary().primitives[name], bars).status
                for name in ("fvg_nearest", "order_block", "bars_since")}
    assert set(statuses.values()) == {EXEMPT}, statuses


def test_an_exempt_verdict_names_the_term_and_gives_a_reason(bars):
    """arg_sets_checked is the dataclass default, so assert what the code sets."""
    verdict = verify_term(core_vocabulary().primitives["bars_since"], bars)
    assert verdict.name == "bars_since"
    assert "condition" in verdict.reason and "evaluator" in verdict.reason


def test_a_condition_taking_term_is_exempt_rather_than_raising(bars):
    """bars_since cannot be called without eval_fn, so the gate refuses it early."""
    assert verify_term(core_vocabulary().primitives["bars_since"], bars).status == EXEMPT


# The exempt set is declared, not counted. A term going silently exempt is how a
# real look-ahead bug would hide behind a green run.
EXPECTED_EXEMPT = {
    "fvg_nearest": "end_anchored",
    "order_block": "end_anchored",
    "bars_since": "condition",
}


def test_every_core_term_is_checked_or_declared_exempt(bars):
    verdicts = verify_vocabulary(core_vocabulary(), bars)
    assert len(verdicts) == len(core_vocabulary().all_terms()) == 37

    failed = [v for v in verdicts if v.status == FAILED]
    assert not failed, "\n".join(f"{v.name}: {v.reason}" for v in failed)

    vacuous = [v for v in verdicts if v.status == VACUOUS]
    assert not vacuous, (
        "a mandated argument set was NaN at every probe row, so the fixture is "
        "too short for it: " + "\n".join(f"{v.name}: {v.reason}" for v in vacuous))


def test_every_checked_term_checked_all_of_its_argument_sets(bars):
    """CHECKED means every mandated set was exercised, not merely one of them."""
    for verdict in verify_vocabulary(core_vocabulary(), bars):
        if verdict.status != CHECKED:
            continue
        term = (core_vocabulary().indicators.get(verdict.name)
                or core_vocabulary().primitives[verdict.name])
        assert verdict.arg_sets_checked == len(arg_sets(term)), verdict.name


def test_the_exempt_set_is_exactly_the_declared_one(bars):
    verdicts = verify_vocabulary(core_vocabulary(), bars)
    exempt = {v.name for v in verdicts if v.status == EXEMPT}
    assert exempt == set(EXPECTED_EXEMPT)
    for v in verdicts:
        if v.status == EXEMPT:
            assert EXPECTED_EXEMPT[v.name] in v.reason


def test_the_gate_covers_every_name_in_the_vocabulary(bars):
    """Enumeration comes from the vocabulary, so there is no manifest to forget."""
    v = core_vocabulary()
    names = {verdict.name for verdict in verify_vocabulary(v, bars)}
    assert names == set(v.indicators) | set(v.primitives)


def test_the_gate_reads_the_vocabulary_it_is_handed(bars):
    """The property every other test in this file is blind to.

    Each of the tests above compares verify_vocabulary(core_vocabulary(), bars)
    against core_vocabulary(), so an implementation that ignored its argument and
    enumerated core_vocabulary() internally would pass all of them. That mutant is
    exactly the disaster node 02 walks into: it composes
    core_vocabulary().with_terms(*ta_terms()), gets 37 green verdicts for core's own
    terms, sees no FAILED, and registers 100-plus unverified terms while CI reads
    green and hard rule 1 is broken.

    So compose a vocabulary this function was not built from and require the
    injected term to come back FAILED.
    """
    injected = Term("house_peeker", "series", {}, {}, lambda s, a: s.shift(-1))
    composed = core_vocabulary().with_terms(injected)
    verdicts = {v.name: v for v in verify_vocabulary(composed, bars)}

    assert "house_peeker" in verdicts, (
        "verify_vocabulary did not read the vocabulary it was handed")
    assert verdicts["house_peeker"].status == FAILED
    assert len(verdicts) == len(core_vocabulary().all_terms()) + 1
