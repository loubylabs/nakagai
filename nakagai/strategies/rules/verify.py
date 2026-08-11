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

from dataclasses import dataclass

from nakagai.strategies.rules.vocabulary import Term, is_choice_rule

# The arg rule that marks a condition-taking term. A bare string rather than a
# tuple, so `is_choice_rule` is False for it. Node 03 promotes this to an
# official arg type; keying on the string means this module needs no edit then.
CONDITION_ARG = "condition"

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
