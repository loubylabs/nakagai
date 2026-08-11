# nakagai/strategies/rules/verify.py
"""The per-term causality gate: does this term read only rows <= i?

`frame_eval.py` states the property the whole DSL rests on: every node in the
grammar is causal, so a node computed once over the whole frame and indexed at
row i gives the same number the prefix computation gave. `tests/
test_whole_frame_equivalence.py` pins that for the grammar, node by node, from
a hand-maintained list. This module pins it for ONE TERM, from the term's own
schema, so a term nobody wrote by hand can be admitted or refused.

The two files answer different questions and neither replaces the other. The
equivalence test covers math ops, `tf` cutting, nested `of`, end-anchored spans
and the `bars_since` evaluator injection, which this module does not. This
module covers terms that do not appear in any list, which that test cannot.

WHAT THIS CANNOT TEST, and why it says so out loud rather than passing:

- `end_anchored` terms. `primitives.end_anchored_series` computes row i as
  `term.fn(bars[:i+1])`, so the whole-frame path IS the per-prefix loop and
  comparing them compares a value to itself. There is no whole-frame broadcast
  to compare against, because the function returns a scalar for a frame.
- Terms taking a condition, today only `bars_since`, which needs the evaluator
  handed back to it and cannot be called without one.

A boolean return would make those two indistinguishable from a genuine pass,
which is the one failure this gate cannot afford, so every answer is a
TermVerdict carrying a status and a reason.
"""

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nakagai.strategies.rules.vocabulary import Term, Vocabulary, is_choice_rule

# The cross-repo contract, which is a different and much smaller thing than the
# module's public surface. Node 02 lives in another repository and reaches this
# module through the rev-pinned git dependency; these are the names it may hold
# on to. Everything else here is internal to the gate and free to change.
__all__ = ["CHECKED", "EXEMPT", "FAILED", "VACUOUS",
           "TermVerdict", "reference_bars", "verify_vocabulary"]

# The arg rule that marks a condition-taking term. A bare string rather than a
# tuple, so `is_choice_rule` is False for it. Node 03 promotes this to an
# official arg type; keying on the string means this module needs no edit then.
CONDITION_ARG = "condition"

# A term whose schema generates more than this is refused rather than sampled.
# Measured on core: ichimoku is the worst non-exempt term at 36, fvg_nearest is
# 60 and exempt anyway. A bounded result that reads as complete is worse than a
# loud partial one, so this raises rather than truncating.
MAX_ARG_SETS = 128

CHECKED = "checked"
FAILED = "failed"
EXEMPT = "exempt"
VACUOUS = "vacuous"


# Why a FAILED verdict is a FAILED verdict, in one machine-readable word.
# FAILED is one bucket for five conditions and only "lookahead" is the term
# reading rows after i; "gate_error" is this module's own code failing, which
# the reason strings already said in prose. This is the module's thesis one
# level down: if a bare boolean cannot tell proved-causal from could-not-test,
# a bare FAILED cannot tell this-term-peeks from our-gate-broke. Node 02 lists
# rejected terms in CI output and classifies them on this field, rather than
# string-matching a reason written for a human to read.
CAUSES = ("lookahead", "uncallable", "schema", "unenumerable", "gate_error")


@dataclass(frozen=True)
class TermVerdict:
    """One term's answer. `status` is CHECKED, FAILED, EXEMPT or VACUOUS.

    `arg_sets_checked` counts the mandated argument sets fully verified before
    this verdict was returned. On a rejection that is how many passed first, so
    "failed on set 12 of 21" is readable rather than a bare 0 claiming the gate
    got nowhere.

    `cause` is one of CAUSES on FAILED, and empty on every other status. It is
    appended after `arg_sets_checked` deliberately: the documented field order
    is used positionally at the call sites, and inserting a field mid-order
    would silently rebind them.
    """

    name: str
    status: str
    reason: str = ""
    arg_sets_checked: int = 0
    cause: str = ""


def exemption_reason(term: Term) -> str | None:
    """Why this term cannot be checked by whole-frame-against-prefix, or None."""
    if term.end_anchored:
        return ("end_anchored: the whole-frame path is already the per-prefix "
                "loop (primitives.end_anchored_series), so this check would "
                "compare a value to itself")
    for name, rule in term.args.items():
        if rule == CONDITION_ARG:
            return (f"takes a condition in {name!r}, which needs an evaluator "
                    f"injected before the term can be called at all")
    return None


def field_mismatch(term: Term, out) -> str | None:
    """Do the term's declared `field` choices match the columns it really returns?

    Declared and produced are different questions, and only the second one says
    which columns actually get evaluated. A term declaring three fields while
    returning four leaves the fourth unevaluated and unmentioned, which is what a
    generated signature gets wrong and what a schema-only reading cannot see.

    THE HOLE THIS LEAVES OPEN, named rather than left to be rediscovered: a term
    that declares `field` choices and returns something other than a DataFrame
    is not checked here at all. The gate then verifies it once per declared
    field against one and the same computation and reports CHECKED with
    arg_sets_checked equal to the number of fields, a count that overstates what
    actually varied. Today that is exactly the two end-anchored primitives,
    which select their field inside the function and return a float, and both
    are exempt anyway. Closing it would change what happens to them, which the
    exemption tests pin, so it is a separate decision and not this node's.
    """
    if not isinstance(out, pd.DataFrame):
        return None
    declared = set(term.args.get("field", ()))
    produced = set(out.columns)
    if declared == produced:
        return None
    missing = sorted(produced - declared)
    absent = sorted(declared - produced)
    parts = []
    if missing:
        parts.append(f"produces undeclared {missing}, which would never be evaluated")
    if absent:
        parts.append(f"declares {absent}, which it does not produce")
    return "; ".join(parts)


def _raw_call(term: Term, bars: pd.DataFrame, args: dict):
    """One term's output over `bars`, before any narrowing.

    Four kinds, three calling conventions: series and frame terms take the close
    Series, bar terms take the whole frame, and a primitive takes the evaluator
    slot and the frame with its args spread. They are frame_eval.py:238-278's
    conventions, mirrored rather than imported, because FrameEval takes a whole
    grammar node and this takes one term.

    One function rather than a copy at each of the three call sites. A gate that
    reached a term by a different convention than the evaluator does would be
    verifying something the DSL never runs.
    """
    if term.kind == "primitive":
        return term.fn(None, bars, **args)
    if term.kind == "bar":
        return term.fn(bars, args)
    return term.fn(bars["close"], args)     # series, frame


def evaluate_term(term: Term, bars: pd.DataFrame, args: dict):
    """One term's output over `bars`, narrowed to one column if it returns a frame.

    A DataFrame return is narrowed by `field` for BOTH frame-kind and bar-kind
    terms: donchian, ichimoku, keltner, stoch and supertrend are bar-kind and
    multi-output, so narrowing only frame-kind would raise on five terms.
    """
    out = _raw_call(term, bars, args)
    if isinstance(out, pd.DataFrame):
        out = out[args["field"]]
    return out


def _is_range_rule(rule) -> bool:
    """The other half of the ArgRule union: a low and a high, both numbers.

    Booleans are excluded even though bool subclasses int, so a rule of
    (True, False) is refused rather than read as the range 1 to 0. The same
    exclusion guards the vocabulary's own search for the widest sessions bound.
    """
    return (isinstance(rule, tuple) and len(rule) == 2
            and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in rule))


def arg_sets(term: Term) -> tuple[dict, ...]:
    """Every argument set D11 mandates for this term.

    The full cross product over choice rules, because that is where a callable
    branches and a peek can be reachable only under one combination. Range
    endpoints are varied one at a time against each choice combination rather
    than crossed with each other, which would be exponential in the range-argument
    count and buys much less: a numeric bound rarely selects a code path.

    A condition arg is skipped rather than sampled: a term declaring one is exempt
    anyway, and there is no value to sample.

    A rule of any other shape is refused here rather than guessed at. Reading
    "everything that is not a choice rule is a range" and then indexing rule[0]
    and rule[1] turns a bare string into two one-letter bounds and a list of
    three choices into two of them, and the second one is the dangerous one: the
    term reports CHECKED with a count that looks complete while a declared
    branch was never called. ArgRule is documented as a tuple, but node 02's
    terms come from a generator, and a generator getting the schema shape wrong
    is exactly what this gate is supposed to catch.
    """
    choices, ranges = {}, {}
    for name, rule in term.args.items():
        if rule == CONDITION_ARG:
            continue
        if is_choice_rule(rule):
            choices[name] = rule
        elif _is_range_rule(rule):
            ranges[name] = rule
        else:
            raise ValueError(
                f"term {term.name!r} declares arg {name!r} as {rule!r}, which "
                f"is neither a tuple of string choices nor a 2-tuple of "
                f"numeric bounds; the gate will not guess which one was meant")

    out, seen = [], set()

    def add(candidate: dict) -> None:
        """Keep a new candidate, and refuse once the REAL count passes the cap.

        Counted after deduplication, not from the combinatorial formula. A
        default that coincides with one of its own range endpoints collapses two
        candidates into one, so the formula overstates what the term produces and
        would refuse a term that fits. That direction is safe but still wrong,
        and it stops being hypothetical at node 02, where terms are generated
        from another library's signatures and a default sitting on a bound is
        ordinary rather than exotic.

        Raising from inside the loop also bounds the work: the walk stops at the
        cap instead of materializing a pathological schema's whole cross product
        first.
        """
        key = tuple(sorted(candidate.items()))
        if key in seen:
            return
        if len(seen) >= MAX_ARG_SETS:
            raise ValueError(
                f"term {term.name!r} generates more than {MAX_ARG_SETS} "
                f"argument sets, over the cap; widen the cap deliberately or "
                f"narrow the schema, but do not sample it silently")
        seen.add(key)
        out.append(candidate)

    combos = ((dict(zip(choices, values))
               for values in itertools.product(*choices.values()))
              if choices else iter([{}]))
    for combo in combos:
        base = {**term.defaults, **combo}
        add(base)
        for name, rule in ranges.items():
            for value in (rule[0], rule[1]):
                add({**base, name: value})
    return tuple(out)


# A fixed COUNT, not a fixed stride: widening the fixture must not multiply the
# cost, because node 02 runs this over 100+ terms in CI. Probes start at half the
# frame so every term is past its warm-up, including rvol at its 60-session
# maximum. The residual risk is that a term peeking only between probes passes;
# raising PROBE_COUNT is the lever if measured CI time allows it.
PROBE_COUNT = 20


def probe_rows(n: int) -> list[int]:
    """PROBE_COUNT rows evenly spaced across the second half of the frame."""
    lo, hi = n // 2, n - 1
    if hi <= lo:
        return []
    step = (hi - lo) / max(PROBE_COUNT - 1, 1)
    return sorted({int(lo + round(k * step)) for k in range(PROBE_COUNT)})


def _value_at(out, i: int) -> float:
    return float(out.iloc[i]) if isinstance(out, pd.Series) else float(out)


def _agrees(whole: float, prefix: float) -> bool:
    """Exact equality, deliberately, with no tolerance.

    The same arithmetic over a prefix of the same frame should produce the same
    float, and core's terms are built on pandas rolling and ewm, which accumulate
    sequentially rather than batching by array length. A term that broke that,
    an FFT convolution or a numba-engine reduction, could differ in the last bit
    and be reported FAILED for a reason that is not look-ahead.

    A tolerance would trade that false failure for a false pass, and a false pass
    is the one outcome this gate cannot afford: a term peeking by a small amount
    is still peeking. The false failure is loud, names the row, and a human reads
    it. So the tolerance stays out, and this docstring is the record of the
    choice rather than an oversight for node 02 to rediscover.
    """
    return (pd.isna(whole) and pd.isna(prefix)) or whole == prefix


def verify_term(term: Term, bars: pd.DataFrame) -> TermVerdict:
    """Is this term causal: does row i depend only on rows <= i?

    Computes the term over the whole frame, recomputes it over the prefix ending
    at each probe row, and compares. A disagreement means the whole-frame value at
    row i used a row after i, which is the look-ahead this gate refuses.

    Returns a verdict rather than a boolean so EXEMPT and VACUOUS never read as a
    pass. Vacuity is judged PER ARGUMENT SET: one mandated set that is NaN at every
    probe makes the term vacuous, because the set proves nothing and the schema
    said it had to be tested.

    Every way out of the work below is a verdict, including the ways that are
    this module failing rather than the term. Enumerating the schema, calling the
    term, reading what it returned and probing it are each guarded, because node
    02 runs 100+ terms in one batch and one traceback would end the batch and
    leave the other 99 unreported. The verdict's `cause` is what tells a reject
    that peeked apart from one the gate could not read.
    """
    reason = exemption_reason(term)
    if reason is not None:
        return TermVerdict(term.name, EXEMPT, reason)

    rows = probe_rows(len(bars))
    if not rows:
        return TermVerdict(term.name, VACUOUS,
                           f"frame of {len(bars)} rows is too short to probe")

    # Enumeration is a rejection, not a crash. arg_sets raises over the cap, and
    # letting that propagate would take down a whole node 02 batch of 100+ terms
    # over one wide schema, instead of reporting one refused term among many.
    try:
        every_arg_set = arg_sets(term)
    except Exception as exc:
        return TermVerdict(term.name, FAILED,
                           f"cannot enumerate this term's arguments: {exc}",
                           cause="unenumerable")

    # Nothing to call is a refusal, not a pass. is_choice_rule(()) is True,
    # because all() over an empty tuple is True, so an enum arg that resolves to
    # nothing partitions as a choice, the cross product is empty and the loop
    # below never runs. Falling through would return CHECKED for a term the gate
    # never called, which is precisely what a status vocabulary exists to
    # prevent. FAILED rather than VACUOUS: VACUOUS means the term was called and
    # taught us nothing, and here it was not called at all.
    if not every_arg_set:
        return TermVerdict(term.name, FAILED,
                           "schema generates no argument sets, so nothing was "
                           "called",
                           cause="unenumerable")

    checked = 0
    for args in every_arg_set:
        try:
            raw = _raw_call(term, bars, args)
        except Exception as exc:
            return TermVerdict(term.name, FAILED,
                               f"args {args}: whole-frame call raised {exc}",
                               checked, "uncallable")
        # Everything from here to the end of the argument set is inside one
        # guard, for the same reason the enumeration above is: a term the gate
        # cannot make sense of is a rejection, not an exception out of
        # verify_term that takes down a node 02 batch of 100+ terms. Reading
        # the whole-frame result is four steps, not one, and each of them has
        # been measured escaping as a traceback on a malformed return: the
        # field check, the narrowing, the field lookup and the extraction.
        try:
            mismatch = field_mismatch(term, raw)
            if mismatch is not None:
                return TermVerdict(term.name, FAILED, f"schema: {mismatch}",
                                   checked, "schema")

            # Reuse the call already made rather than calling term.fn a second
            # time on the whole frame. evaluate_term's only step beyond the raw
            # dispatch is this same narrowing, and node 02 multiplies the
            # saving by 100+.
            whole = raw[args["field"]] if isinstance(raw, pd.DataFrame) else raw

            # A return that does not line up with the frame is a shape problem,
            # reported as one. Probing it by position either raises, which says
            # nothing about causality, or, if the lengths line up far enough to
            # index, compares row i against some other row and reports the term
            # for a peek it never made. Cause "schema" rather than "gate_error"
            # because the disagreement is the term's, not this module's.
            if isinstance(whole, pd.Series) and len(whole) != len(bars):
                return TermVerdict(
                    term.name, FAILED,
                    f"args {args}: returned {len(whole)} rows for a frame of "
                    f"{len(bars)}, so its values cannot be lined up with the "
                    f"rows they would describe",
                    checked, "schema")

            saw_a_number = False
            for i in rows:
                want = _value_at(whole, i)
                try:
                    prefix = _value_at(
                        evaluate_term(term, bars.iloc[:i + 1], args), -1)
                except Exception as exc:
                    # "evaluating raised", not "the term raised": this call goes
                    # through evaluate_term, which is the gate's own code, so a
                    # bug in the gate must not read as the term's causality
                    # failure. The row and the argument set are worth naming
                    # here, which is why this sits inside the outer guard rather
                    # than being folded into it.
                    return TermVerdict(
                        term.name, FAILED,
                        f"args {args} row {i}: evaluating over the prefix raised "
                        f"{type(exc).__name__}: {exc}",
                        checked, "gate_error")
                if not _agrees(want, prefix):
                    return TermVerdict(
                        term.name, FAILED,
                        f"args {args} row {i}: whole-frame {want!r} != prefix "
                        f"{prefix!r}, so row {i} read a row after itself",
                        checked, "lookahead")
                saw_a_number = saw_a_number or not pd.isna(want)
        except Exception as exc:
            return TermVerdict(
                term.name, FAILED,
                f"args {args}: reading the whole-frame result raised "
                f"{type(exc).__name__}: {exc}",
                checked, "gate_error")

        if not saw_a_number:
            return TermVerdict(
                term.name, VACUOUS,
                f"args {args} are NaN at all {len(rows)} probe rows, so agreement "
                f"proves nothing about this mandated argument set",
                checked)
        checked += 1

    return TermVerdict(term.name, CHECKED, "", checked)


def verify_vocabulary(vocabulary: Vocabulary,
                      bars: pd.DataFrame) -> tuple[TermVerdict, ...]:
    """Every term in `vocabulary`, verified, in all_terms() order.

    Enumerates from the vocabulary itself rather than from a list a human keeps,
    which is the whole point: node 02 registers terms generated from another
    project's library, and a manifest someone forgets to update is a hole that
    reads as coverage.
    """
    return tuple(verify_term(term, bars) for term in vocabulary.all_terms())


# The frame the gate is meant to be run on, shipped in the wheel rather than
# left in tests/, because `[tool.hatch.build.targets.wheel] packages` is
# ["nakagai"] and node 02 lives in another repository: it gets
# verify_vocabulary through the rev-pinned git dependency and cannot get the
# test fixture. Both of this frame's load-bearing properties were discovered by
# running the gate rather than by reading it, so a hand-rebuilt frame would
# rediscover them as two silent holes rather than as errors.
BARS_PER_SESSION = 26
EXCHANGE_TZ = "America/New_York"


def reference_bars(sessions: int = 160) -> pd.DataFrame:
    """Multi-session RTH-shaped 15m bars: 26 a day, `sessions` weekdays, no weekends.

    160 sessions by default rather than a round 40 because core's widest range
    rule is rvol's `sessions: (5, 60)`, and a mandated argument set that is NaN
    at every probe row proves nothing about the term. A vocabulary that adds a
    wider session-denominated bound has to widen this. Each bar opens at the
    previous close so bodies take both signs, which keeps order_block and any
    close-against-open condition from being constant. Seeded, so the gate's own
    result is reproducible.

    ANCHORED IN EXCHANGE-LOCAL TIME, not at a fixed UTC hour. A frame pinned to
    14:30 UTC is the 09:30 bell only until daylight saving moves, and 160
    sessions from January crosses that boundary in March. Measured: the
    UTC-pinned version leaves opening_range_high and opening_range_low NaN at
    every probe row, because the bars no longer start at the open, and the gate
    reports VACUOUS for terms that are perfectly causal.
    """
    rng = np.random.default_rng(19)
    days = pd.bdate_range("2026-01-05", periods=sessions, tz=EXCHANGE_TZ)
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
