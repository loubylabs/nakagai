# tests/test_verify_term.py
"""The per-term causality gate, and its honesty about what it cannot test."""

import numpy as np
import pandas as pd
import pytest

import nakagai.strategies.rules.verify as verify_module
from nakagai.strategies import indicators as ind
from nakagai.strategies.rules.vocabulary import (
    CONDITION_ARG, Term, core_vocabulary)
from nakagai.strategies.rules.verify import (
    CAUSES, CHECKED, EXEMPT, FAILED, MAX_ARG_SETS, PROBE_COUNT, VACUOUS,
    WARMUP_PROBE_COUNT, _raw_call, arg_sets, evaluate_term, exemption_reason,
    field_mismatch, probe_rows, reference_bars, verify_term, verify_vocabulary,
)


def test_an_end_anchored_term_is_exempt_and_says_why():
    v = core_vocabulary()
    reason = exemption_reason(v.primitives["fvg_nearest"])
    assert reason is not None
    assert "end_anchored" in reason


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
    assert exempt == {"fvg_nearest", "order_block"}
    # N3-D12 retired the condition exemption, so the declared constant has to
    # lose bars_since in the same change the computed set does. Asserted here
    # rather than in a test of its own, which would be a third statement of the
    # same two-member set.
    assert "bars_since" not in EXPECTED_EXEMPT


# Expectations of the shipped frame, held here as literals rather than imported
# from verify.py, so these tests read its properties off the frame instead of
# agreeing with it by construction.
SESSIONS = 160
BARS_PER_SESSION = 26
EXCHANGE_TZ = "America/New_York"


@pytest.fixture(scope="module")
def bars():
    """The reference frame the gate ships, which is what node 02 will run it on.

    The body lives in verify.py rather than here because the wheel ships
    `nakagai` and not `tests/`, and node 02 is a platform node consuming core
    through the rev-pinned git dependency: it gets verify_vocabulary and cannot
    get this fixture. Both load-bearing properties of the frame were discovered
    by running the gate rather than by reading it, so a hand-rebuilt frame over
    there would rediscover them as two silent holes. The tests below assert
    those properties against the shipped function.
    """
    return reference_bars(SESSIONS)


@pytest.fixture(scope="module")
def verdicts(bars):
    """One pass of the gate over core, read by every whole-vocabulary test below.

    The gate is not cheap: each term is called once per mandated argument set
    over the whole frame, and again over 20 prefixes of it. Five readers running
    their own pass cost about 30 of this file's 38 seconds and learned nothing
    the first pass had not already established. This file is the template node
    02 copies onto a frame three times larger with 100-plus terms, so the shape
    of it matters more than the seconds do here.

    The one test that cannot use this is the one that hands the gate a composed
    vocabulary, which is the whole point of that test.
    """
    return verify_vocabulary(core_vocabulary(), bars)


def test_the_shipped_default_is_the_frame_these_tests_assert_against(bars):
    """Node 02 calls reference_bars() with no argument, so the default is the contract.

    Every test in this file passes SESSIONS explicitly. A default that drifted
    smaller would leave them all green while node 02 got a frame too short to
    clear rvol's 60-session bound and read VACUOUS for a causal term.
    """
    assert reference_bars().equals(bars)


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
        # The gate's own dispatch, without evaluate_term's field narrowing,
        # because the question is which columns exist before one is selected.
        # Calling by kind matters: a primitive called down the series branch
        # binds the defaults dict to `bars` and crashes inside find_fvgs.
        raw = _raw_call(term, bars, dict(term.defaults))
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


def test_arg_sets_supplies_a_condition_arg():
    """N3-D12's inverse of node 01's skip: supplied, not skipped.

    A skipped condition arg made the term uncallable, which is the only reason
    it was exempt. Every mandated set now carries the same synthetic mask, so
    the argument space does not grow and both evaluation paths see the arg.
    """
    term = core_vocabulary().primitives["bars_since"]
    sets = arg_sets(term)
    assert sets
    assert all(s.get("cond") == verify_module.SYNTHETIC_CONDITION for s in sets), sets


def test_the_gates_mask_follows_the_condition_it_is_handed(bars, monkeypatch):
    """SYNTHETIC_CONDITION is live, not decorative.

    The callback used to restate the constant as `b["close"] > b["open"]` and
    ignore the `cond` it was passed. Both statements agreed, so nothing was
    wrong today and nothing could ever go wrong loudly: editing the constant to
    widen the gate's mask changed no mask, no verdict, and no test, so a future
    change intended to tighten causality checking would have been a no-op with
    a green suite behind it.

    Flipping the operand is the smallest edit that tells a live constant from an
    inert one. The two masks are near-complements on this frame, so a callback
    reading `cond` cannot return the same count for both, and one restating the
    constant cannot return anything else.
    """
    def echo_mask(ctx, frame, cond, eval_fn=None):
        """Its whole output is the mask's size, so the mask is observable."""
        return pd.Series(float(eval_fn(cond, frame).sum()), index=frame.index)

    term = Term("echo_mask", "primitive", {"cond": CONDITION_ARG}, {}, echo_mask)

    def count_under(condition):
        # The whole chain, not one link of it: the constant is what arg_sets
        # puts on every candidate, and _raw_call is what injects the callback.
        monkeypatch.setattr(verify_module, "SYNTHETIC_CONDITION", condition)
        (args,) = arg_sets(term)
        return _raw_call(term, bars, args).iloc[0]

    rising = count_under({"lhs": {"src": "close"}, "op": ">",
                          "rhs": {"src": "open"}})
    falling = count_under({"lhs": {"src": "close"}, "op": "<",
                           "rhs": {"src": "open"}})
    assert rising and falling, (rising, falling)
    assert rising != falling, (
        f"the mask counts {rising} under both conditions, so the callback is "
        f"restating SYNTHETIC_CONDITION rather than reading it")


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


def test_a_range_rule_with_numpy_bounds_is_read_as_a_range_whichever_shape(bars):
    """The shared range predicate accepts NumPy real scalars consistently.

    Node 02 generates schemas from another library's signatures, which is exactly
    where NumPy scalars arrive.
    """
    ints = Term("numpy_int_bounds", "series",
                {"n": (np.int64(2), np.int64(100))}, {"n": np.int64(20)},
                lambda s, a: s)
    assert {int(s["n"]) for s in arg_sets(ints)} == {2, 20, 100}
    assert verify_term(ints, bars).status == CHECKED

    floats = Term("numpy_float_bounds", "series",
                  {"f": (np.float64(0.0), np.float64(1.0))},
                  {"f": np.float64(0.5)}, lambda s, a: s)
    assert {float(s["f"]) for s in arg_sets(floats)} == {0.0, 0.5, 1.0}


def test_a_pair_of_booleans_is_still_refused_rather_than_read_as_1_to_0():
    """The exclusion that has to survive the widening, in both bool shapes.

    bool subclasses int, so (True, False) would read as the range 1 to 0 without
    it. np.bool_ subclasses neither bool nor numbers.Real, so it is refused by
    the same predicate for a different reason; asserted here so a later widening
    that admits it reddens.
    """
    for pair in ((True, False), (np.bool_(True), np.bool_(False))):
        term = Term("flagged", "series", {"flag": pair}, {}, lambda s, a: s)
        with pytest.raises(ValueError, match="flag"):
            arg_sets(term)


def test_a_declared_argument_with_no_default_is_supplied_not_left_out(bars):
    """The gate must not reject a term over an argument set its schema never sanctioned.

    `vocabulary.py:96-101` checks only that every default names a declared arg,
    never the reverse, so a term may declare `n` and default nothing. The
    baseline set was `{**term.defaults, **combo}`, which then omits `n` entirely.
    Measured: arg_sets returned ({}, {'n': 2}, {'n': 500}) and verify_term
    answered FAILED with cause "uncallable" and reason "args {}: whole-frame call
    raised 'n'", which is a fault the gate created.

    Filling from the low bound costs nothing: that value is a mandated endpoint
    already, so the baseline set dedupes against it rather than adding a set.
    """
    term = Term("no_default", "series", {"n": (2, 500)}, {},
                lambda s, a: s.rolling(a["n"]).mean())

    sets = arg_sets(term)
    assert all("n" in s for s in sets), sets
    assert {s["n"] for s in sets} == {2, 500}
    assert verify_term(term, bars).status == CHECKED


def test_no_core_term_exceeds_the_argument_set_cap():
    """Headroom, asserted explicitly, because arg_sets polices the cap itself.

    The previous form asked for the worst term by len(arg_sets(t)), which calls
    arg_sets on every term, and arg_sets raises past the cap. So the assertion
    after it could only ever be reached when it was already true. This one goes
    red on its own as core grows: a term landing exactly on the cap enumerates
    without raising and fails here.
    """
    widths = {t.name: len(arg_sets(t)) for t in core_vocabulary().all_terms()}
    worst = max(widths, key=widths.get)
    assert widths[worst] < MAX_ARG_SETS, (
        f"{worst} generates {widths[worst]} argument sets against a cap of "
        f"{MAX_ARG_SETS}; widen the cap deliberately or narrow the schema")


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


def test_probe_rows_cover_the_warmup_as_well_as_the_settled_frame(bars):
    """Both regions, and never the last row.

    The settled half is where terms are past their warm-up and the comparison is
    against real numbers. The warm-up half is where a term that fills its leading
    NaNs from future rows shows itself, and it was not probed at all: rows 0 to
    2079 of a 4160-row frame, half the frame, never compared once.

    The last row is excluded because `bars.iloc[:n]` IS `bars`, so that probe
    compares a value with itself and cannot disagree for any term.
    """
    n = len(bars)
    rows = probe_rows(n)
    assert len(rows) == len(set(rows))
    assert len([i for i in rows if i >= n // 2]) == PROBE_COUNT
    assert len([i for i in rows if i < n // 2]) == WARMUP_PROBE_COUNT
    assert min(rows) == 0, "row 0 is where a backfilled NaN shows itself"
    assert max(rows) == n - 2
    assert n - 1 not in rows, "that probe's prefix IS the frame it is compared to"


def test_probe_rows_refuses_a_frame_with_no_strict_prefix_to_probe():
    """Under two rows there is no row whose prefix is shorter than the frame."""
    assert probe_rows(0) == []
    assert probe_rows(1) == []
    assert probe_rows(2) == [0]


def _bfill_series(s, a):
    return ind.sma(s, a["n"]).bfill()


def _bfill_bar(frame, a):
    return ind.sma(frame["close"], a["n"]).bfill()


def _bfill_primitive(_eval_fn, frame, **args):
    return ind.sma(frame["close"], args["n"]).bfill()


@pytest.mark.parametrize("kind, fn", [("series", _bfill_series),
                                      ("bar", _bfill_bar),
                                      ("primitive", _bfill_primitive)])
def test_a_term_that_fills_its_warmup_from_the_future_is_failed(kind, fn, bars):
    """The hole the half-frame probe window left open, on all three kinds.

    `.bfill()` on an indicator is the commonest convenience idiom in the library
    node 02 generates its 100-plus terms from, and it is look-ahead: row 0 is
    handed a number first computable at row n-1 of the warm-up.

    Measured with probes starting at n // 2: CHECKED, on all three kinds, while
    19 rows of the whole-frame output disagreed with their prefix. Every one of
    those rows sat below the first probe. Core's widest warm-up is rvol at 60
    sessions, 1560 rows, still under 2080, so no term's warm-up was ever reached.
    """
    term = Term(f"bfill_{kind}", kind, {"n": (2, 500)}, {"n": 20}, fn)

    whole = ind.sma(bars["close"], 20).bfill()
    assert not pd.isna(whole.iloc[0]), "the counterexample no longer backfills"
    assert pd.isna(ind.sma(bars["close"].iloc[:1], 20).bfill().iloc[-1]), (
        "the prefix must still be NaN there, or there is nothing to disagree with")

    verdict = verify_term(term, bars)
    assert verdict.status == FAILED, f"a warm-up peek is still a peek: {verdict}"
    assert verdict.cause == "lookahead"


def test_a_term_whose_only_evidence_is_the_frame_against_itself_is_vacuous(bars):
    """A probe at row n-1 compares `bars.iloc[:n]` with `bars`, which is itself.

    Measured: a window as wide as the frame is NaN at every probe but the last,
    and that one comparison set `saw_a_number`, so the term came back CHECKED on
    evidence that could not have come out any other way. That is the shape the
    end_anchored exemption exists to refuse, reintroduced for one probe in twenty
    on every term.
    """
    term = Term("whole_frame_window", "series", {}, {},
                lambda s, a: s.rolling(len(bars)).mean())
    out = term.fn(bars["close"], {})
    assert out.notna().sum() == 1 and not pd.isna(out.iloc[-1]), (
        "this term must be non-NaN at exactly the last row")

    verdict = verify_term(term, bars)
    assert verdict.status == VACUOUS, (
        f"the last row's prefix IS the frame, so it is no evidence: {verdict}")


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


def _shifted_index_peeker(s, a):
    """Honest values, relabelled one row early, which is a peek BY LABEL.

    Positionally this agrees with the prefix at every probe row, because the
    values are an ordinary causal rolling mean. The index is the defect: position
    i carries the label of row i-1, so the value published for row i-1 is the one
    computed over rows through i.

    `frame_eval.py:215` returns a term's own Series unchanged when the term's
    timeframe is the driving one, so this index is what production reads by.
    """
    honest = s.rolling(3).mean()
    step = s.index[1] - s.index[0]
    return pd.Series(honest.to_numpy(),
                     index=s.index[:-1].insert(0, s.index[0] - step))


def test_a_term_returning_a_shifted_index_is_a_shape_failure_not_a_pass(bars):
    """Measured on the unguarded gate: CHECKED, with the label reading row i+1.

    Length alone said nothing here: this term returns exactly as many rows as it
    was given, and the gate read them by position, so the shift the index carries
    was invisible to it.
    """
    term = Term("shifted_index_peeker", "series", {}, {}, _shifted_index_peeker)

    out = _shifted_index_peeker(bars["close"], {})
    honest = bars["close"].rolling(3).mean()
    assert not out.index.equals(bars.index), "this term is meant to relabel"
    assert out.loc[bars.index[100]] == honest.iloc[101], (
        "the counterexample no longer reads the future by label")

    verdict = verify_term(term, bars)
    assert verdict.status == FAILED, "a relabelled return is not a pass"
    assert verdict.cause == "schema"
    assert "index" in verdict.reason
    assert "read a row after itself" not in verdict.reason


def test_a_prefix_result_that_does_not_carry_the_prefix_index_is_a_failure(bars):
    """Lining up with the whole frame does not mean lining up with a prefix.

    This term always labels its output with the LAST rows of the full frame, so
    the whole-frame call is indistinguishable from an honest one and only the
    prefix calls are mislabelled. Checking the whole-frame index alone would pass
    it, and its values are honest, so the value comparison passes too.
    """
    full = bars.index

    def relabels_only_a_prefix(s, a):
        return pd.Series(s.to_numpy(), index=full[len(full) - len(s):])

    term = Term("relabels_a_prefix", "series", {}, {}, relabels_only_a_prefix)
    assert relabels_only_a_prefix(bars["close"], {}).index.equals(bars.index), (
        "the whole-frame call must look honest for this test to mean anything")

    verdict = verify_term(term, bars)
    assert verdict.status == FAILED
    assert verdict.cause == "schema"
    assert "index" in verdict.reason


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


def _mutating_bar_peeker(frame, a):
    """Writes the next bar's close into `open`, then answers out of `open`.

    The write is made once and marked, so every prefix the gate afterwards slices
    out of the frame it handed over already carries the peek and agrees with it.
    A term like this is not caught by comparing values, because both sides of the
    comparison are reading the same poisoned column.
    """
    if "_poisoned" not in frame.columns:
        frame["open"] = frame["close"].shift(-1)
        frame["_poisoned"] = 1.0
    return frame["open"]


def test_a_term_that_writes_into_the_frame_it_was_given_is_failed(bars):
    """Measured on the unguarded gate: CHECKED, with `open` holding row i+1's close.

    The gate hands the term the original frame, exactly as frame_eval.py:251
    does, and then slices every prefix out of that same object. A term that
    scribbles on it is comparing against its own scribble.

    The frame is copied here because the guard DETECTS the mutation rather than
    defending against it: the write still lands, and the shared fixture must not
    carry it into the tests that follow.
    """
    scratch = bars.copy()
    term = Term("mutating_bar_peeker", "bar", {}, {}, _mutating_bar_peeker)

    verdict = verify_term(term, scratch)
    assert verdict.status == FAILED, "a term that poisons its input is not a pass"
    assert verdict.cause == "mutation"
    assert "wrote into the frame" in verdict.reason

    assert "_poisoned" in scratch.columns, "the mutation is detected, not prevented"
    assert scratch["open"].iloc[100] == bars["close"].iloc[101], (
        "the counterexample no longer writes the future into the frame")


def test_a_term_that_writes_into_the_frame_after_the_first_call_is_caught(bars):
    """The fingerprint is re-checked after EVERY call, not only the first.

    A term that behaves on the whole-frame call and scribbles from inside the
    probe loop would walk past a check made once. This one keeps the frame it was
    first handed and poisons it on the third call.
    """
    scratch = bars.copy()
    seen = []

    def late_mutator(frame, a):
        seen.append(frame)
        if len(seen) == 3:
            seen[0]["open"] = seen[0]["close"].shift(-1)
        return frame["close"]

    verdict = verify_term(Term("late_mutator", "bar", {}, {}, late_mutator),
                          scratch)
    assert verdict.status == FAILED
    assert verdict.cause == "mutation"
    assert len(seen) >= 3, "the mutation must happen after the whole-frame call"


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


def test_every_failure_mode_names_a_declared_cause_and_every_cause_has_one(bars):
    """CAUSES is the vocabulary node 02 classifies rejects on, and nothing read it.

    This branch's discipline elsewhere is that the exempt set is asserted against
    a declared constant rather than counted, because a set the code merely
    happens to produce is not a contract. CAUSES had the comment and not the
    assertion, so a typo'd or drifted cause string could not be caught.

    Both halves matter. Keying the table on the expected cause stops a verdict
    carrying a cause outside the vocabulary, or a failure mode quietly changing
    which one it reports. Comparing the keys against CAUSES stops CAUSES growing
    a literal that nothing emits, which is the other way a declared constant
    stops describing the code.

    Each term gets its own frame because one of them writes into the frame it is
    given, which is the whole point of that one.
    """
    broken = {
        "lookahead": Term("peeker", "series", {}, {}, _peeking_series),
        "uncallable": Term("explodes", "series", {}, {},
                           lambda s, a: (_ for _ in ()).throw(ValueError("boom"))),
        "schema": Term("under_declared", "frame", {"field": ("a",)},
                       {"field": "a"},
                       lambda s, a: pd.DataFrame({"a": s, "b": s.shift(-1)})),
        "unenumerable": Term("too_wide", "series",
                             {f"c{i}": ("x", "y", "z") for i in range(6)},
                             {f"c{i}": "x" for i in range(6)}, lambda s, a: s),
        "mutation": Term("mutating_bar_peeker", "bar", {}, {},
                         _mutating_bar_peeker),
        "gate_error": Term("empty_frame", "series", {}, {},
                           lambda s, a: pd.DataFrame()),
    }

    assert set(broken) == set(CAUSES), (
        f"declared but never produced here: {sorted(set(CAUSES) - set(broken))}; "
        f"produced here but not declared: {sorted(set(broken) - set(CAUSES))}")

    for want, term in broken.items():
        verdict = verify_term(term, bars.copy())
        assert verdict.status == FAILED, f"{term.name}: {verdict}"
        assert verdict.cause in CAUSES, f"{verdict.cause!r} is not a declared cause"
        assert verdict.cause == want, f"{term.name}: {verdict.reason}"


def test_a_verdict_that_is_not_a_failure_carries_no_cause(bars):
    """An empty cause is what "this is not a rejection" reads as downstream."""
    checked = verify_term(core_vocabulary().indicators["sma"], bars)
    assert checked.status == CHECKED and checked.cause == ""

    exempt = verify_term(core_vocabulary().primitives["fvg_nearest"], bars)
    assert exempt.status == EXEMPT and exempt.cause == ""

    vacuous = verify_term(Term("always_nan", "series", {}, {},
                               lambda s, a: pd.Series(float("nan"), index=s.index)),
                          bars)
    assert vacuous.status == VACUOUS and vacuous.cause == ""


def _vacuous_at_2_peeking_at_500(s, a):
    if a["n"] == 2:
        return pd.Series(float("nan"), index=s.index)
    if a["n"] == 500:
        return s.shift(-1)
    return s


def test_an_earlier_vacuous_argument_set_does_not_hide_a_later_peeking_one(bars):
    """Every mandated set is evaluated, and FAILED outranks VACUOUS.

    Measured: with the sets enumerated as ({'n': 20}, {'n': 2}, {'n': 500}), the
    return on the first vacuous set exited the whole loop, so the look-ahead in
    the n=500 set was never reached. The verdict read VACUOUS naming n=2, and
    that reason points the operator at the fixture, which is the one place the
    answer is not.
    """
    term = Term("vacuous_then_peeking", "series", {"n": (2, 500)}, {"n": 20},
                _vacuous_at_2_peeking_at_500)
    assert [s["n"] for s in arg_sets(term)] == [20, 2, 500], (
        "the ordering this test is about has changed")

    verdict = verify_term(term, bars)
    assert verdict.status == FAILED, f"the peek outranks the vacuity: {verdict}"
    assert verdict.cause == "lookahead"
    assert "500" in verdict.reason


def test_a_term_that_raises_only_over_a_prefix_is_uncallable_not_a_gate_error(bars):
    """cause "gate_error" means THIS module broke, and node 02 classifies on it.

    `evaluate_term`'s only work beyond `_raw_call` is a column selection, so an
    exception on that path nearly always comes out of `term.fn`. Labelling it
    gate_error tells node 02 its gate is broken when what happened is that a
    batch of generated terms needs more history than a prefix supplies, which is
    what "uncallable" already means and what the same exception on the whole
    frame is labelled.

    Measured: cause was "gate_error".
    """
    def needs_the_whole_frame(s, a):
        if len(s) < 4000:
            raise ValueError("needs 4000 bars")
        return s

    verdict = verify_term(
        Term("short_raiser", "series", {}, {}, needs_the_whole_frame), bars)
    assert verdict.status == FAILED
    assert verdict.cause == "uncallable", (
        f"the term raised, not the gate: {verdict.reason}")
    assert "needs 4000 bars" in verdict.reason


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
def test_every_shipped_indicator_shape_is_checked_and_passes(name, verdicts):
    """Read out of the shared pass: verifying these again proves nothing new.

    The named shapes are what the parametrisation is for, so a shape that stops
    being CHECKED still names itself here rather than hiding in a list.
    """
    by_name = {v.name: v for v in verdicts}
    assert name in by_name, f"{name} is no longer a term in the vocabulary"
    assert by_name[name].status == CHECKED, f"{name}: {by_name[name].reason}"


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


def _end_anchored_peeker(_eval_fn, frame, **args):
    """An `end_anchored` term whose body genuinely reads a future row."""
    return frame["close"].shift(-1)


def test_a_broken_end_anchored_term_is_reported_exempt_not_passing(bars, monkeypatch):
    """The frozen acceptance criterion, which honest terms cannot prove.

    "A deliberately broken term that is `end_anchored` is reported as exempt,
    not as passing." Running the two honest core terms through the exemption
    shows the exemption FIRES. It does not show what the gate does with an
    end-anchored term that peeks, and that is the case the criterion is about:
    the gate has to be honest about what it did not test rather than blessing it.

    Both halves of this term matter. It carries the flag, so EXEMPT is the only
    admissible verdict. And it really peeks, so with the exemption removed the
    gate finds the peek, which is what makes EXEMPT a refusal to bless rather
    than an accident of an honest body.
    """
    term = Term("end_anchored_peeker", "primitive", {}, {},
                _end_anchored_peeker, end_anchored=True)

    verdict = verify_term(term, bars)
    assert verdict.status == EXEMPT, (
        f"an end_anchored term that peeks came back {verdict.status}: "
        f"{verdict.reason}")
    assert verdict.status not in (CHECKED, FAILED)
    assert "end_anchored" in verdict.reason
    assert verdict.cause == "", "EXEMPT is not a rejection and carries no cause"

    monkeypatch.setattr(verify_module, "exemption_reason", lambda _term: None)
    unexempt = verify_term(term, bars)
    assert unexempt.status == FAILED, (
        "the term must really be broken, or EXEMPT proves nothing")
    assert unexempt.cause == "lookahead"


def test_an_end_anchored_term_returns_a_scalar_not_a_series(bars):
    """The structural fact underneath the exemption, asserted rather than assumed."""
    term = core_vocabulary().primitives["fvg_nearest"]
    out = term.fn(None, bars, **dict(term.defaults))
    assert isinstance(out, float)
    assert not isinstance(out, pd.Series)


def test_an_exempt_term_is_never_reported_as_checked_or_failed(bars):
    """One assertion that can fail on its own, not a restatement of EXEMPT."""
    statuses = {name: verify_term(core_vocabulary().primitives[name], bars).status
                for name in ("fvg_nearest", "order_block")}
    assert set(statuses.values()) == {EXEMPT}, statuses


def test_an_exempt_verdict_names_the_term_and_gives_a_reason(bars):
    """arg_sets_checked is the dataclass default, so assert what the code sets."""
    verdict = verify_term(core_vocabulary().primitives["fvg_nearest"], bars)
    assert verdict.name == "fvg_nearest"
    assert "end_anchored" in verdict.reason and "scalar" in verdict.reason


def test_bars_since_is_checked_not_exempt(bars):
    """N3-D12: the condition exemption is retired.

    bars_since is a normal CHECKED term now, verified against the same
    whole-frame-against-prefix proof every other term gets, via a synthetic
    causal mask rather than the real grammar evaluator.
    """
    v = core_vocabulary()
    verdict = verify_term(v.primitives["bars_since"], bars)
    assert verdict.status == CHECKED, verdict.reason


def test_exemption_reason_no_longer_mentions_condition():
    v = core_vocabulary()
    assert exemption_reason(v.primitives["bars_since"]) is None


def test_both_evaluation_paths_receive_the_condition_argument(bars, monkeypatch):
    """The behavioral replacement for the source-text guard node 01 made moot.

    A guard counting `term.fn(` occurrences in verify_term's source reads zero
    already, because node 01 extracted _raw_call, so it could never go red.
    This one can: it records what every call actually received, and a path that
    stopped supplying `cond` shows up as a call with `cond` missing rather than
    as an unchanged count.

    The mutation this catches has to be at a CALL SITE, not inside _raw_call.
    `seen` is appended as _raw_call receives its arguments, so a drop performed
    inside _raw_call's own body happens after the recording and is invisible
    here. The divergence being guarded is between the two call sites.

    Measured with the whole-frame call site mutated to drop "cond": bars_since
    is called without its required positional, raises TypeError, and the gate
    turns that into FAILED/uncallable, so this test and
    test_bars_since_is_checked_not_exempt both redden on their CHECKED
    assertion. The `seen` assertions are the backstop for a divergence that
    does NOT raise (a term with a defaulted condition arg, which N3-D13 refuses
    today but a generated vocabulary could reintroduce).
    """
    seen = []
    real = verify_module._raw_call

    def recording(term, b, args):
        seen.append((len(b), dict(args)))
        return real(term, b, args)

    monkeypatch.setattr(verify_module, "_raw_call", recording)
    verdict = verify_term(core_vocabulary().primitives["bars_since"], bars)

    assert verdict.status == CHECKED, verdict.reason
    assert len(seen) > 1, "expected a whole-frame call and at least one prefix call"
    assert all("cond" in args for _, args in seen), (
        f"a call reached the term without its condition: {seen}")
    assert len({length for length, _ in seen}) > 1, (
        "every call saw the same frame length, so no prefix pass ran")


def test_a_future_reading_condition_term_is_caught_by_the_synthetic_mask(bars):
    """The falsification test N3-D12 names.

    A condition-taking term that reads one row into the future against the
    synthetic mask must come back FAILED, proving the gate is capable of
    catching such a term now that it is exercised at all.
    """
    def peeking_bars_since(ctx, b, cond, eval_fn=None):
        mask = eval_fn(cond, b).astype(bool)
        return mask.shift(-1).astype(float)     # reads one row into the future

    peeker = Term("peeking_bars_since", "primitive", {"cond": CONDITION_ARG},
                  {}, peeking_bars_since)
    verdict = verify_term(peeker, bars)
    assert verdict.status == FAILED, verdict.reason
    # FAILED alone is too weak to carry this. verify.py returns FAILED for six
    # causes, and five of them mean NO ROW WAS EVER COMPARED. Measured against a
    # half-implemented N3-D12 (exemption retired, arg_sets' skip left in
    # place), this peeker comes back cause="uncallable", because it is called
    # without cond and raises TypeError; a perfectly causal term returns the
    # identical verdict, so status alone cannot tell a caught peek from a term
    # the gate could not call. Only "lookahead" means a whole-frame row was
    # compared against a prefix row and disagreed.
    assert verdict.cause == "lookahead", (
        f"expected a detected peek, got cause={verdict.cause!r}: {verdict.reason}")


def test_an_honest_condition_term_is_checked_not_merely_not_failed(bars):
    """The other half of the falsification pair, and what makes it a pair.

    Without it, the peeker above is satisfied by a gate that fails EVERY
    condition-taking term for any reason at all.
    """
    def honest_bars_since(ctx, b, cond, eval_fn=None):
        mask = eval_fn(cond, b).astype(bool)
        return mask.astype(float)               # reads row i only

    honest = Term("honest_bars_since", "primitive", {"cond": CONDITION_ARG},
                  {}, honest_bars_since)
    verdict = verify_term(honest, bars)
    assert verdict.status == CHECKED, f"{verdict.cause}: {verdict.reason}"
    assert verdict.arg_sets_checked >= 1, "checked nothing and said CHECKED"


# The exempt set is declared, not counted. A term going silently exempt is how a
# real look-ahead bug would hide behind a green run.
EXPECTED_EXEMPT = {
    "fvg_nearest": "end_anchored",
    "order_block": "end_anchored",
}


def test_every_core_term_is_checked_or_declared_exempt(verdicts):
    assert len(verdicts) == len(core_vocabulary().all_terms()) == 37

    failed = [v for v in verdicts if v.status == FAILED]
    assert not failed, "\n".join(f"{v.name}: {v.reason}" for v in failed)

    vacuous = [v for v in verdicts if v.status == VACUOUS]
    assert not vacuous, (
        "a mandated argument set was NaN at every probe row, so the fixture is "
        "too short for it: " + "\n".join(f"{v.name}: {v.reason}" for v in vacuous))


def test_every_checked_term_checked_all_of_its_argument_sets(verdicts):
    """CHECKED means every mandated set was exercised, not merely one of them."""
    for verdict in verdicts:
        if verdict.status != CHECKED:
            continue
        term = (core_vocabulary().indicators.get(verdict.name)
                or core_vocabulary().primitives[verdict.name])
        assert verdict.arg_sets_checked == len(arg_sets(term)), verdict.name


def test_the_exempt_set_is_exactly_the_declared_one(verdicts):
    exempt = {v.name for v in verdicts if v.status == EXEMPT}
    assert exempt == set(EXPECTED_EXEMPT)
    for v in verdicts:
        if v.status == EXEMPT:
            assert EXPECTED_EXEMPT[v.name] in v.reason


def test_the_gate_covers_every_name_in_the_vocabulary(verdicts):
    """Enumeration comes from the vocabulary, so there is no manifest to forget."""
    v = core_vocabulary()
    names = {verdict.name for verdict in verdicts}
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
