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

- `window_required` terms. Their callable has no unwindowed meaning. The
  window evaluator supplies the occurrence and reducer behavior, so invoking
  `term.fn` here would test an operation the language refuses.

- `end_anchored` terms. `term.fn` returns a SCALAR for a frame, not a series, so
  there is no whole-frame value at row i to compare a prefix against. What the
  gate would compare is that one scalar, read off the whole frame, against the
  scalar the same function returns for each prefix, and those are two different
  numbers. Measured with the exemption neutralized: `fvg_nearest` comes back
  FAILED, "whole-frame 88.44637561140192 != prefix 93.37990812403403", and the
  cause reads "lookahead". So the hazard is a FALSE FAILURE on an honest term,
  not a comparison of a value to itself that could never fail.
  `primitives.end_anchored_series` is what the evaluator really runs for these,
  computing row i as `term.fn(bars[:i+1])`, which is causal by construction.

A term taking a condition used to be exempt because it needs an evaluator
handed back to it and cannot be called without one. N3-D12 supplies
SYNTHETIC_CONDITION and the callback that reads it, so such a term is checked
like any other.

A boolean return would make an exemption indistinguishable from a genuine pass,
which is the one failure this gate cannot afford, so every answer is a
TermVerdict carrying a status and a reason.
"""

import itertools
import json
import operator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nakagai.strategies.rules.vocabulary import (
    Term, Vocabulary, is_choice_rule, is_condition_rule, is_range_rule)

# The cross-repo contract, which is a different and much smaller thing than the
# module's public surface. Node 02 lives in another repository and reaches this
# module through the rev-pinned git dependency; these are the names it may hold
# on to. Everything else here is internal to the gate and free to change.
__all__ = ["CAUSES", "CHECKED", "EXEMPT", "FAILED", "VACUOUS",
           "TermVerdict", "reference_bars", "verify_vocabulary"]

# The value handed to every condition-typed arg, so the term is callable here.
#
# Deterministic and ROW-LOCAL: `close > open` reads only the current row, so it
# needs no grammar evaluator and cannot itself introduce a peek. The question
# this module asks is whether THE TERM reads only rows <= i, and handing it a
# known-causal condition isolates that question. The condition's own causality
# is N3-D5's four guards' job, and test_whole_frame_equivalence.py's.
#
# It replaces an exemption rather than sitting beside one. Node 01 exempted a
# condition-taking term because there was no evaluator here to inject, so the
# term could not be called at all; it was never uncheckable in principle. With
# N3-D9 a platform-registered condition-taking term needs no core PR, so an
# exemption left standing would see such a term registered, validated,
# evaluated and reported EXEMPT, which is to say never causality checked, in
# direct breach of hard rule 1: no term enters the vocabulary without passing
# verify_term.
SYNTHETIC_CONDITION = {"lhs": {"src": "close"}, "op": ">", "rhs": {"src": "open"}}

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
# string-matching a reason written for a human to read. Exported for the same
# reason the four status literals are: a consumer that classifies on this field
# and cannot read the vocabulary would hardcode the strings and drift silently.
CAUSES = ("lookahead", "uncallable", "schema", "unenumerable", "mutation",
          "gate_error")


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
    """Why this term cannot be checked by whole-frame-against-prefix, or None.

    A window-required term has no unwindowed callable to compare. N3-D12
    retired the condition exemption: a condition-typed arg now gets
    SYNTHETIC_CONDITION and the evaluator callback that reads it, so the term is
    callable here and is checked like any other. end_anchored is unchanged;
    term.fn returns a scalar for a frame there, so the whole-frame path IS the
    per-prefix loop and the comparison would fail an honest term.
    """
    if term.window_required:
        return ("window_required: term.fn has no unwindowed meaning; the "
                "window evaluator applies the declared reducer to each "
                "occurrence")
    if term.end_anchored:
        # "causal by construction" is load-bearing and it is not free. It holds
        # because end_anchored_series calls term.fn on bars[:i+1] AND because
        # every value the term reads is cut to that prefix, including the mask a
        # condition-typed arg's injected callback returns. This gate exempts the
        # whole class, so nothing here would notice if that second half stopped
        # being true; frame_eval.py's injection carries the guard for it, and
        # test_an_end_anchored_primitive_gets_the_evaluator_injected_too is what
        # reddens when it lapses.
        return ("end_anchored: term.fn returns a scalar for a frame, not a "
                "series, so there is no whole-frame value at row i to compare a "
                "prefix against and the comparison would fail an honest term; "
                "the evaluator runs these through "
                "primitives.end_anchored_series, which is causal by "
                "construction so long as everything it hands the term is cut to "
                "the prefix, the injected condition mask included")
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


def _synthetic_mask(cond: dict, bars: pd.DataFrame) -> pd.Series:
    """`cond` evaluated row-locally, with no grammar evaluator involved.

    Deliberately the smallest reader that can answer SYNTHETIC_CONDITION's
    shape, rather than a call into FrameEval: this module verifies causality,
    so a gate that leaned on the evaluator would be asking the subject of the
    experiment to certify itself. Two column reads and one comparison cannot
    shift, roll, or reach past their own row whatever the constant says.

    Anything outside that shape raises rather than degrading, so a constant
    edited into something this cannot read fails here and names itself instead
    of quietly verifying every term against a mask nobody chose.
    """
    ops = {">": operator.gt, "<": operator.lt,
           ">=": operator.ge, "<=": operator.le}
    return ops[cond["op"]](bars[cond["lhs"]["src"]], bars[cond["rhs"]["src"]])


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

    A condition-typed arg additionally needs an evaluator callback, which is
    injected HERE rather than at either call site, so verify_term's whole-frame
    pass and evaluate_term's per-prefix pass cannot receive different ones. It
    reads the condition it is HANDED off the frame it is handed, which is
    row-local, so the callback cannot introduce a peek of its own. Keyed on the
    arg's declared type, exactly as `frame_eval.py:281` keys the real injection,
    so a term registered outside core reaches it the same way.

    It reads `cond` rather than restating it. An earlier draft hard-coded
    `b["close"] > b["open"]` here while SYNTHETIC_CONDITION said the same thing
    twenty lines up, which made the constant inert: editing it to widen or
    tighten the gate's mask changed nothing, and no test could go red, because
    every candidate still carried a mask the callback had already decided.
    """
    if any(is_condition_rule(rule) for rule in term.args.values()):
        args = {**args, "eval_fn": _synthetic_mask}
    if term.kind == "primitive":
        return term.fn(None, bars, **args)
    if term.kind == "bar":
        return term.fn(bars, args)
    return term.fn(bars["close"], args)     # series, frame


def _narrow(out, args: dict):
    """One term's output, narrowed to one column if it returned a frame.

    A DataFrame return is narrowed by `field` for BOTH frame-kind and bar-kind
    terms: donchian, ichimoku, keltner, stoch and supertrend are bar-kind and
    multi-output, so narrowing only frame-kind would raise on five terms.

    One function rather than a copy here and another inline in verify_term, for
    the same reason _raw_call is one function: the whole-frame side reuses the
    call it already made and narrows it itself, while the prefix side goes
    through evaluate_term. Narrowing differently on the two sides would compare a
    value from one column against a value from another and call the difference
    look-ahead.
    """
    return out[args["field"]] if isinstance(out, pd.DataFrame) else out


def evaluate_term(term: Term, bars: pd.DataFrame, args: dict):
    """One term's output over `bars`, narrowed to one column if it returns a frame."""
    return _narrow(_raw_call(term, bars, args), args)


def _frame_fingerprint(bars: pd.DataFrame) -> tuple:
    """A cheap summary of the frame that changes when a term writes into it.

    The gate hands a term the caller's own frame, exactly as `frame_eval.py:251`
    does, and then slices every prefix out of that same object. A term that
    scribbles on it is compared against its own scribble, so both sides agree and
    the verdict reads CHECKED. Measured: a bar term writing the next bar's close
    into `open` came back CHECKED with the frame poisoned.

    Detected rather than defended against. Copying the frame per argument set
    would hide the defect, and a term that mutates its input is broken whatever
    else it does; copying would also cost node 02 real time, at 184 argument sets
    on core alone.

    The columns catch a term that adds, drops or renames one, the shape catches a
    term that drops rows, and the per-column sums with the NaN count catch a value
    overwritten in place. One numpy reduction rather than a digest of every byte,
    because this is recomputed after all 21 calls of every argument set and node
    02 multiplies that by 100-plus terms: measured at about 19us on the reference
    frame, so about 0.07s over a whole pass of core.

    Deliberately not a cryptographic digest. Two writes whose sums cancel would
    slip past. This is a tripwire for a term that scribbles on its input, not a
    defence against one built to evade the tripwire, and the second is a different
    problem from the one a causality gate is for.
    """
    values = bars.to_numpy()
    if values.dtype.kind in "fiub":
        summary = (tuple(np.nansum(values, axis=0).tolist()),
                   int(np.isnan(values).sum()))
    else:
        # A frame carrying a non-numeric column reduces to no meaningful sum, so
        # hash it instead. Total rather than skipped: a column this function
        # cannot summarize is a column a term could rewrite unseen.
        summary = (int(pd.util.hash_pandas_object(bars, index=False).sum()),)
    return (tuple(bars.columns), values.shape, values.dtype.str, summary)


def arg_sets(term: Term) -> tuple[dict, ...]:
    """Every argument set D11 mandates for this term.

    The full cross product over choice rules, because that is where a callable
    branches and a peek can be reachable only under one combination. Range
    endpoints are varied one at a time against each choice combination rather
    than crossed with each other, which would be exponential in the range-argument
    count and buys much less: a numeric bound rarely selects a code path.

    A condition arg is SUPPLIED rather than skipped, per N3-D12: every candidate
    carries the same SYNTHETIC_CONDITION, so the argument space does not grow
    and both evaluation paths see the arg. Skipping it left the term uncallable,
    which was the whole of its exemption.

    A rule of any other shape is refused here rather than guessed at. Reading
    "everything that is not a choice rule is a range" and then indexing rule[0]
    and rule[1] turns a bare string into two one-letter bounds and a list of
    three choices into two of them, and the second one is the dangerous one: the
    term reports CHECKED with a count that looks complete while a declared
    branch was never called. ArgRule is documented as a tuple, but node 02's
    terms come from a generator, and a generator getting the schema shape wrong
    is exactly what this gate is supposed to catch.
    """
    choices, ranges, conditions = {}, {}, {}
    for name, rule in term.args.items():
        if is_condition_rule(rule):
            conditions[name] = SYNTHETIC_CONDITION
            continue
        if is_choice_rule(rule):
            choices[name] = rule
        elif is_range_rule(rule):
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

        The key is a canonical encoding rather than a tuple of items, because a
        condition-typed arg's value is a dict and hashing one raises. Measured
        with SYNTHETIC_CONDITION supplied and the old key left in place:
        verify_term(bars_since) came back status=failed cause=unenumerable,
        "cannot enumerate this term's arguments: unhashable type: 'dict'", which
        is every condition-taking term rejected rather than checked.
        `sort_keys=True` keeps two orderings of the same candidate equal, which
        is what `sorted(...)` bought; `default=repr` keeps the key total over
        values json cannot encode, so a future non-JSON arg value degrades to a
        coarser key rather than raising. Canonicalized rather than special-cased
        on the condition, so the next nested value needs no third branch.
        """
        key = json.dumps(candidate, sort_keys=True, default=repr)
        if key in seen:
            return
        if len(seen) >= MAX_ARG_SETS:
            raise ValueError(
                f"term {term.name!r} generates more than {MAX_ARG_SETS} "
                f"argument sets, over the cap; widen the cap deliberately or "
                f"narrow the schema, but do not sample it silently")
        seen.add(key)
        out.append(candidate)

    # A declared range arg with no default, filled from its low bound.
    # `vocabulary.py:96-101` checks only that every default names a declared arg,
    # never the reverse, so a term may declare `n` and default nothing. The
    # baseline set would then omit it, term.fn would raise KeyError, and the gate
    # would reject a perfectly causal term as "uncallable" over an argument set
    # its own schema never sanctioned. Node 02 generates terms from another
    # library's signatures, where a required parameter with no library default is
    # ordinary. Choice args need no such filling: the cross product supplies
    # every one of them. The low bound costs nothing, because it is a mandated
    # endpoint already and the baseline dedupes against it.
    supplied = {name: rule[0] for name, rule in ranges.items()
                if name not in term.defaults}

    combos = ((dict(zip(choices, values))
               for values in itertools.product(*choices.values()))
              if choices else iter([{}]))
    for combo in combos:
        base = {**supplied, **term.defaults, **combo, **conditions}
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

# The same again, across the warm-up. IN ADDITION to the settled probes, never
# instead of them: the floor exists so every term has real numbers to compare,
# and trading that away to reach the warm-up would buy one hole with another.
WARMUP_PROBE_COUNT = 20


def _spread(lo: int, hi: int, count: int) -> set[int]:
    """`count` rows evenly spaced from lo to hi inclusive, deduplicated."""
    if hi < lo:
        return set()
    step = (hi - lo) / max(count - 1, 1)
    return {int(lo + round(k * step)) for k in range(count)}


def probe_rows(n: int) -> list[int]:
    """Rows to compare, across the warm-up AND the settled part of the frame.

    Probing only the settled half left the other half never compared once, and
    look-ahead confined to a term's warm-up was therefore admitted as CHECKED.
    Measured: `sma(n).bfill()` came back CHECKED on all three term kinds while 19
    of its rows held a value first computable at a strictly later row, every one
    of them below the first probe. `.bfill()` is the commonest convenience idiom
    in the library node 02 generates its terms from, so this is not an exotic
    shape. The warm-up rows cost nothing in false failures: an honest term is NaN
    on both sides there and `_agrees` reads NaN against NaN as agreement.

    THE LAST ROW IS NEVER PROBED. `bars.iloc[:n]` IS `bars`, so a probe at n-1
    compares a value with itself, cannot disagree for any term, and yet counts as
    evidence: measured, a window as wide as the frame was NaN at every other
    probe and came back CHECKED on that one self-comparison. A probe that cannot
    fail is not a probe, and excluding it also keeps it out of the vacuity guard.

    A fixed COUNT in each region, not a fixed stride: widening the fixture must
    not multiply the cost, because node 02 runs this over 100-plus terms in CI.
    The residual risk is a term peeking only between probes; raising either count
    is the lever if measured CI time allows it.
    """
    hi = n - 2
    if hi < 0:
        return []
    settled = _spread(n // 2, hi, PROBE_COUNT)
    warmup = _spread(0, min(n // 2 - 1, hi), WARMUP_PROBE_COUNT)
    return sorted(settled | warmup)


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

    # The frame as it was handed over, to compare every call's aftermath against.
    # Taken once, before anything is called, because a term that has already
    # written into it would otherwise set the baseline itself.
    try:
        untouched = _frame_fingerprint(bars)
    except Exception as exc:
        return TermVerdict(
            term.name, FAILED,
            f"cannot fingerprint the frame this term was to be checked on: "
            f"{type(exc).__name__}: {exc}",
            cause="gate_error")

    checked, vacuous = 0, None
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
            # Asked before anything else is read, because every later comparison
            # in this argument set is against the frame the term just changed.
            if _frame_fingerprint(bars) != untouched:
                return TermVerdict(
                    term.name, FAILED,
                    f"args {args}: the whole-frame call wrote into the frame it "
                    f"was given, so every comparison after it is against a frame "
                    f"the term itself changed",
                    checked, "mutation")

            mismatch = field_mismatch(term, raw)
            if mismatch is not None:
                return TermVerdict(term.name, FAILED, f"schema: {mismatch}",
                                   checked, "schema")

            # Reuse the call already made rather than calling term.fn a second
            # time on the whole frame. evaluate_term's only step beyond the raw
            # dispatch is this same narrowing, and node 02 multiplies the
            # saving by 100+.
            whole = _narrow(raw, args)

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

            # The same length is not the same rows. This gate reads the return
            # by POSITION, and `frame_eval.py:215` publishes the term's own
            # Series unchanged on its native timeframe, so a return that keeps
            # the row count while carrying a different index is read one way
            # here and another way in production. Measured: a term returning an
            # honest rolling mean relabelled one row early passed as CHECKED
            # while the value published at row i was computed over rows through
            # i+1. That is the one outcome this gate cannot afford, so the index
            # is compared and not merely counted.
            if isinstance(whole, pd.Series) and not whole.index.equals(bars.index):
                return TermVerdict(
                    term.name, FAILED,
                    f"args {args}: returned an index that is not the frame's, "
                    f"so its values would be published against rows other than "
                    f"the ones they were computed from",
                    checked, "schema")

            saw_a_number = False
            for i in rows:
                want = _value_at(whole, i)
                window = bars.iloc[:i + 1]
                try:
                    got = evaluate_term(term, window, args)
                    # The prefix answer is read by position too, and lining up
                    # with the whole frame does not mean lining up with a prefix
                    # of it: a term labelling its output with the frame's LAST
                    # rows is indistinguishable from an honest one on the whole
                    # frame and wrong on every prefix.
                    if (isinstance(got, pd.Series)
                            and not got.index.equals(window.index)):
                        return TermVerdict(
                            term.name, FAILED,
                            f"args {args} row {i}: over the prefix it returned "
                            f"an index that is not the prefix's, so the two "
                            f"sides of this comparison describe different rows",
                            checked, "schema")
                    prefix = _value_at(got, -1)
                except Exception as exc:
                    # "uncallable", the same cause the whole-frame call raising
                    # gets, and NOT "gate_error". evaluate_term's only work
                    # beyond _raw_call is a column selection, so an exception on
                    # this path nearly always comes out of term.fn: a term
                    # needing more history than a prefix supplies is the ordinary
                    # case. gate_error means this module broke, and node 02
                    # classifies rejects on this field, so labelling a term fault
                    # as a gate fault sends it looking in the wrong repository.
                    # The row and the argument set are worth naming here, which
                    # is why this sits inside the outer guard rather than being
                    # folded into it.
                    return TermVerdict(
                        term.name, FAILED,
                        f"args {args} row {i}: the prefix call raised "
                        f"{type(exc).__name__}: {exc}",
                        checked, "uncallable")
                # After every call, not only after the first: a term that behaves
                # on the whole frame and scribbles from inside the probe loop
                # would walk past a check made once.
                if _frame_fingerprint(bars) != untouched:
                    return TermVerdict(
                        term.name, FAILED,
                        f"args {args} row {i}: the prefix call wrote into the "
                        f"frame the gate is checking against, so the rows the "
                        f"remaining probes read are the term's own work",
                        checked, "mutation")
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
            # Recorded and carried on from, not returned. Returning here exited
            # the whole loop, so a LATER mandated set that reads the future was
            # never evaluated: measured, a term vacuous at n=2 and peeking at
            # n=500 came back VACUOUS naming n=2, and that reason points the
            # operator at the fixture, which is the one place the answer is not.
            # FAILED outranks VACUOUS, so the peek has to be looked for first.
            if vacuous is None:
                vacuous = args
            continue
        checked += 1

    if vacuous is not None:
        return TermVerdict(
            term.name, VACUOUS,
            f"args {vacuous} are NaN at all {len(rows)} probe rows, so agreement "
            f"proves nothing about this mandated argument set",
            checked)
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
    UTC-pinned version moves the final 15-minute bars outside regular hours,
    making session-shaped terms exercise a different frame than intended.
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
