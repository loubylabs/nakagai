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

import pandas as pd

from nakagai.strategies.rules.vocabulary import Term, Vocabulary, is_choice_rule

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


@dataclass(frozen=True)
class TermVerdict:
    """One term's answer. `status` is CHECKED, FAILED, EXEMPT or VACUOUS."""

    name: str
    status: str
    reason: str = ""
    arg_sets_checked: int = 0


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


def evaluate_term(term: Term, bars: pd.DataFrame, args: dict):
    """One term's output over `bars`, by its kind's calling convention.

    The four conventions are frame_eval.py:238-278's, mirrored rather than
    imported, because FrameEval takes a whole grammar node and this takes one
    term. A DataFrame return is narrowed by `field` for BOTH frame-kind and
    bar-kind terms: donchian, ichimoku, keltner, stoch and supertrend are bar-kind
    and multi-output, so narrowing only frame-kind would raise on five terms.
    """
    if term.kind == "primitive":
        out = term.fn(None, bars, **args)
    elif term.kind == "bar":
        out = term.fn(bars, args)
    else:                                   # series, frame
        out = term.fn(bars["close"], args)
    if isinstance(out, pd.DataFrame):
        out = out[args["field"]]
    return out


def arg_sets(term: Term) -> tuple[dict, ...]:
    """Every argument set D11 mandates for this term.

    The full cross product over choice rules, because that is where a callable
    branches and a peek can be reachable only under one combination. Range
    endpoints are varied one at a time against each choice combination rather
    than crossed with each other, which would be exponential in the range-argument
    count and buys much less: a numeric bound rarely selects a code path.

    A condition arg is skipped rather than sampled: a term declaring one is exempt
    anyway, and there is no value to sample.
    """
    choices, ranges = {}, {}
    for name, rule in term.args.items():
        if rule == CONDITION_ARG:
            continue
        (choices if is_choice_rule(rule) else ranges)[name] = rule

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
    except Exception as exc:                           # noqa: BLE001
        return TermVerdict(term.name, FAILED,
                           f"cannot enumerate this term's arguments: {exc}")

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
                           "called")

    checked = 0
    for args in every_arg_set:
        try:
            raw = (term.fn(None, bars, **args) if term.kind == "primitive"
                   else term.fn(bars, args) if term.kind == "bar"
                   else term.fn(bars["close"], args))
        except Exception as exc:                       # noqa: BLE001
            return TermVerdict(term.name, FAILED,
                               f"args {args}: whole-frame call raised {exc}")
        mismatch = field_mismatch(term, raw)
        if mismatch is not None:
            return TermVerdict(term.name, FAILED, f"schema: {mismatch}")

        # Reuse the call already made rather than calling term.fn a second time
        # on the whole frame. evaluate_term's only step beyond the raw dispatch
        # is this same narrowing, and node 02 multiplies the saving by 100+.
        whole = raw[args["field"]] if isinstance(raw, pd.DataFrame) else raw
        saw_a_number = False
        for i in rows:
            want = _value_at(whole, i)
            try:
                prefix = _value_at(evaluate_term(term, bars.iloc[:i + 1], args), -1)
            except Exception as exc:                   # noqa: BLE001
                # "evaluating raised", not "the term raised": this call goes
                # through evaluate_term, which is the gate's own code, so a bug
                # in the gate must not read as the term's causality failure.
                return TermVerdict(
                    term.name, FAILED,
                    f"args {args} row {i}: evaluating over the prefix raised "
                    f"{type(exc).__name__}: {exc}")
            if not _agrees(want, prefix):
                return TermVerdict(
                    term.name, FAILED,
                    f"args {args} row {i}: whole-frame {want!r} != prefix "
                    f"{prefix!r}, so row {i} read a row after itself")
            saw_a_number = saw_a_number or not pd.isna(want)
        if not saw_a_number:
            return TermVerdict(
                term.name, VACUOUS,
                f"args {args} are NaN at all {len(rows)} probe rows, so agreement "
                f"proves nothing about this mandated argument set")
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
