"""Every name the grammar admits has a Pine form, and every helper it leans on.

The completeness assertion is the whole point of the file. A term declared
without a lowering is not a compile error anywhere: it is a spec the compiler
refuses at generation time, which a caller only discovers by writing that spec
and hitting the refusal. Asserting the vocabulary is covered turns that into a
failure here, next to the declaration that caused it.

The rest of the file guards the helper graph, which nothing else can: a helper
whose source calls another helper it never declared as a dependency emits Pine
that references an undefined function, and TradingView reports that from the
chart rather than from the compiler.
"""

import json
import os
import re
import subprocess
import sys

from nakagai.strategies.rules import lower_pine
from nakagai.strategies.rules.pine.lowerings import HELPERS
from nakagai.strategies.rules.vocabulary import core_vocabulary

# One condition per primitive, on an intraday chart so the intraday-only ones
# are admitted. Every helper in the program's graph is reachable from here.
EVERY_PRIMITIVE = {
    "version": 2, "name": "every_primitive", "timeframe": "15m",
    "long": {"all": [
        {"lhs": {"prim": "opening_range_high"}, "op": ">", "rhs": 0},
        {"lhs": {"prim": "opening_range_low"}, "op": ">", "rhs": 0},
        {"lhs": {"prim": "prev_session_high"}, "op": ">", "rhs": 0},
        {"lhs": {"prim": "prev_session_low"}, "op": ">", "rhs": 0},
        {"lhs": {"prim": "prev_session_close"}, "op": ">", "rhs": 0},
        {"lhs": {"prim": "gap_pct"}, "op": ">", "rhs": 0},
        {"lhs": {"prim": "rvol"}, "op": ">", "rhs": 1},
        {"lhs": {"prim": "minutes_into_session"}, "op": ">", "rhs": 30},
        {"lhs": {"prim": "day_of_week"}, "op": "<", "rhs": 4},
        {"lhs": {"prim": "swing_high"}, "op": ">", "rhs": 0},
        {"lhs": {"prim": "swing_low"}, "op": ">", "rhs": 0},
        {"lhs": {"prim": "leg_retrace"}, "op": ">", "rhs": 0.62},
        {"lhs": {"prim": "bars_since",
                 "cond": {"lhs": {"src": "close"}, "op": ">",
                          "rhs": {"src": "open"}}}, "op": "<", "rhs": 5},
        {"lhs": {"prim": "fvg_nearest"}, "op": ">", "rhs": 0},
        {"lhs": {"prim": "order_block"}, "op": ">", "rhs": 0},
    ]},
}


def test_every_current_term_has_a_pine_lowering():
    missing = [term.name for term in core_vocabulary().all_terms()
               if term.pine is None]
    assert missing == []


def test_every_helper_a_term_names_is_registered():
    for term in core_vocabulary().all_terms():
        for helper_id in term.pine.helpers:
            assert helper_id in HELPERS, \
                f"{term.name} names the unregistered helper {helper_id!r}"


def test_every_helper_dependency_is_registered():
    for helper in HELPERS.values():
        for dependency in helper.dependencies:
            assert dependency in HELPERS, \
                f"{helper.id} depends on the unregistered {dependency!r}"


def test_a_helper_declares_exactly_the_helpers_its_source_calls():
    """The one thing a reader cannot check by eye, and the compiler cannot see.

    A helper's dependencies decide what gets emitted before it. A source
    calling `nk_new_session()` without declaring it compiles here and fails on
    the chart with an undefined identifier, in a program that may not even use
    the primitive that would have pulled the dependency in by another route.
    Equality rather than containment, so a dependency nothing calls is caught
    too: it would emit a Pine function no line of the script reads.
    """
    for helper in HELPERS.values():
        called = set(re.findall(r"\bnk_[a-z0-9_]+", helper.source)) - {helper.id}
        assert called == set(helper.dependencies), \
            f"{helper.id} calls {sorted(called)} and declares " \
            f"{sorted(helper.dependencies)}"


def test_a_helper_is_named_by_the_function_its_source_defines():
    # The id is what a lowering asks for and what the dependency graph orders;
    # the source's own header is what Pine binds. An id that named a different
    # function would emit both a function nothing calls and a call to a
    # function nothing defines.
    for helper in HELPERS.values():
        assert helper.source.startswith(f"{helper.id}("), helper.id


def test_every_registered_helper_is_reachable_from_some_term():
    """No dead Pine in the registry: every helper is called by a lowering, or
    by a helper that one calls."""
    reachable: set[str] = set()
    frontier = [helper_id for term in core_vocabulary().all_terms()
                for helper_id in term.pine.helpers]
    while frontier:
        helper_id = frontier.pop()
        if helper_id in reachable:
            continue
        reachable.add(helper_id)
        frontier.extend(HELPERS[helper_id].dependencies)
    assert set(HELPERS) == reachable


def test_a_program_emits_every_helper_after_the_ones_it_calls():
    program = lower_pine(EVERY_PRIMITIVE)
    emitted: list[str] = []
    for helper in program.helpers:
        assert set(helper.dependencies) <= set(emitted), \
            f"{helper.id} is emitted before {sorted(set(helper.dependencies) - set(emitted))}"
        emitted.append(helper.id)
    assert len(emitted) == len(set(emitted))


def test_a_program_emits_the_helper_graph_it_actually_reaches():
    """Exactly the closure, with nothing missing and nothing extra.

    Missing is Pine that calls a function nothing defines. Extra is a function
    the script never reads, which on a chart is a compile-time cost and a
    reader's dead end.
    """
    program = lower_pine(EVERY_PRIMITIVE)
    closure: set[str] = set()
    frontier = [helper_id for term in core_vocabulary().primitives.values()
                for helper_id in term.pine.helpers]
    while frontier:
        helper_id = frontier.pop()
        if helper_id in closure:
            continue
        closure.add(helper_id)
        frontier.extend(HELPERS[helper_id].dependencies)
    assert {helper.id for helper in program.helpers} == closure
    # The spec divides nothing, so the one helper no primitive reaches stays out.
    assert closure == set(HELPERS) - {"nk_div"}


# Compiled in a fresh interpreter so the comparison spans two hash seeds. The
# helper graph is walked out of a set, which is exactly the shape an in-process
# check cannot see: set iteration is stable within one process whatever
# PYTHONHASHSEED says.
_PROBE = """
import json, sys
from nakagai.strategies.rules import lower_pine
print(repr(lower_pine(json.loads(sys.argv[1]))))
"""


def test_the_primitive_helper_order_is_identical_across_hash_seeds():
    payload = json.dumps(EVERY_PRIMITIVE)
    outputs = set()
    for seed in ("0", "1", "524287"):
        run = subprocess.run(
            [sys.executable, "-c", _PROBE, payload],
            capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONHASHSEED": seed})
        outputs.add(run.stdout)
    assert len(outputs) == 1
    assert outputs.pop().strip() == repr(lower_pine(EVERY_PRIMITIVE))
