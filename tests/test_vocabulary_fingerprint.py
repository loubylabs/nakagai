"""Hard rule 5: a vocabulary must hash identically in a pool worker and its
parent.

`Vocabulary` is deliberately unhashable (a frozen dataclass over two mappings,
so the generated `__hash__` raises `TypeError` on the dicts it hashes), which
is why the digest below is computed explicitly rather than taken from `hash()`.
That property is documented at `vocabulary.py`'s own `Vocabulary` docstring and
is not this node's to guard; asserting it here would add a test that cannot
fail from anything in this node's diff.

The digest is rebuilt in a FRESH interpreter via a spawn-start pool, never
fork: `core_vocabulary` is `@cache`d, and a forked worker inherits the parent's
already-warmed cache, which would make the test agree with itself without
rebuilding anything.

Two things go into the digest, and both are load-bearing.

The per-term tuple carries each term's args and defaults **in declaration
order**, not sorted. Sorting them would hide the failure mode this rule names:
a schema assembled from a set or any other unordered comprehension rebuilds in
a different order in a fresh interpreter, because string hashing is seeded per
process, and that order is visible to `_bounds` in both prompt renderers and to
`_check_args`'s error messages.

The `spec_hash` half covers **one spec per grammar shape**, not one spec. A
canonicalization that is unstable, or simply wrong, for `any`, for `not`, for a
double negation, for the short side or for an exits group would be invisible to
a single `all`-shaped spec.

The grammar names read below (`GROUP_KEYS`, `validate_spec`,
`is_condition_rule`) are imported inside the test or helper that uses them
rather than at module scope. All three arrive with this node, so a
module-scope import would turn every failing state below into one collection
error against the branch point, and the failure each test is written to see
would never be watched.
"""

import hashlib
import json
import multiprocessing as mp

from nakagai.strategies.rules.canon import spec_hash
from nakagai.strategies.rules.vocabulary import core_vocabulary

_RISK = {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
         "target": {"kind": "rr", "rr": 2.0}}
_LEAF = {"lhs": {"src": "close"}, "op": ">", "rhs": 1}
_CONDITION_LEAF = {"lhs": {"prim": "bars_since",
                           "cond": {"lhs": {"src": "close"}, "op": ">",
                                    "rhs": {"src": "open"}}},
                   "op": "<", "rhs": 5}


def _spec(name, **parts):
    return {"version": 2, "name": name, "timeframe": "1h", "risk": _RISK,
            **parts}


# One per grammar shape the node admits, so a canonicalization that is unstable
# for exactly one of them cannot hide behind the others.
SHAPE_CORPUS = {
    "all": _spec("all", long={"all": [_LEAF]}),
    "any": _spec("any", long={"any": [_LEAF, dict(_LEAF, op="<")]}),
    "nested": _spec("nested", long={"all": [{"any": [_LEAF]}, _LEAF]}),
    "not": _spec("not", long={"not": {"any": [_LEAF]}}),
    "not-not": _spec("not-not", long={"not": {"not": {"all": [_LEAF]}}}),
    "not-nested": _spec("not-nested",
                        long={"all": [{"not": {"any": [_LEAF]}}, _LEAF]}),
    "condition-arg": _spec("condition-arg", long={"all": [_CONDITION_LEAF]}),
    "short-side": _spec("short-side", short={"not": {"all": [_LEAF]}}),
    "exits-group": _spec("exits-group", long={"all": [_LEAF]},
                         exits={"exit": {"not": {"any": [_LEAF]}}}),
}


def _term_tuple(t):
    return (t.name, t.kind, tuple(t.args.items()), tuple(t.defaults.items()),
            t.end_anchored, t.session_scoped, t.driving_frame_intraday)


def vocabulary_fingerprint() -> str:
    """Module-level and picklable: a spawn-context worker imports it by
    qualified name, which a closure or a local function could not be."""
    v = core_vocabulary()
    terms = sorted(_term_tuple(t) for t in v.all_terms())
    hashes = [spec_hash(SHAPE_CORPUS[k], v) for k in sorted(SHAPE_CORPUS)]
    blob = repr(terms) + "|" + "|".join(hashes)
    return hashlib.sha256(blob.encode()).hexdigest()


def _keys_in(spec):
    """The group keys reachable in one spec."""
    from nakagai.strategies.rules.spec import GROUP_KEYS
    used, stack = set(), [spec.get("long"), spec.get("short"),
                          spec.get("exits", {}).get("exit")]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        for key in GROUP_KEYS:
            if key in node:
                used.add(key)
                val = node[key]
                stack.extend(val if isinstance(val, list) else [val])
    return used


def test_every_grammar_shape_is_in_the_fingerprint_corpus():
    """The corpus is the coverage claim, so it is asserted rather than
    trusted: a shape added to the grammar and forgotten here would leave the
    fingerprint silently narrower than the rule requires.

    The distinctness assertion is the one that carries the weight. A union of
    group keys over the whole corpus is an aggregate floor: every key here is
    supplied by more than one entry, so an entry replaced by a copy of
    another entry's shape leaves the union unchanged and the test green. A
    hand-written per-entry key table was tried and is worse, because it is a
    second description of the corpus that drifts from it. Canonical structure
    is derived from the corpus itself and cannot drift.
    """
    from nakagai.strategies.rules.spec import GROUP_KEYS, validate_spec
    for shape, spec in SHAPE_CORPUS.items():
        assert validate_spec(spec) == [], f"{shape}: {validate_spec(spec)}"

    # One spec per grammar shape means nine DIFFERENT shapes.
    structures = {shape: spec_hash(spec, core_vocabulary())
                  for shape, spec in SHAPE_CORPUS.items()}
    dupes = {h for h in structures.values()
             if list(structures.values()).count(h) > 1}
    assert not dupes, (
        "two corpus entries have the same canonical structure, so one of them "
        f"is not the shape it is named for: "
        f"{sorted(k for k, v in structures.items() if v in dupes)}")
    assert len(SHAPE_CORPUS) == 9

    assert set().union(*(_keys_in(s) for s in SHAPE_CORPUS.values())) == set(GROUP_KEYS)


def _condition_taking_terms():
    """The names of every term declaring a condition-typed arg."""
    from nakagai.strategies.rules.vocabulary import is_condition_rule
    return {t.name for t in core_vocabulary().all_terms()
            if any(is_condition_rule(r) for r in t.args.values())}


def _condition_terms_used_by(spec) -> set[str]:
    """Which condition-taking terms `spec`'s RULES call, name field excluded.

    ONE predicate, called by the shape assertion below and by the guard that
    proves the shape assertion can fail. The guard must CALL this rather than
    restate it: the two previous versions of that guard each re-implemented
    the check against a locally built dict, so they agreed with a copy and a
    simplification of the real predicate reddened nothing. This is the third
    version, and calling is the whole of what makes it different.

    Only the three rule groups are searched. The spec's own `name` is the
    substring trap this predicate exists to avoid, since the corpus entry is
    called "condition-arg", and `risk` and `timeframe` cannot name a term.

    The floor lives HERE, in the shared predicate, rather than in either
    caller. `any(...)` over an empty set of terms is vacuously False, so an
    empty vocabulary reads as "no condition in this spec" to a caller and
    silently satisfies the negative half. Measured: with `is_condition_rule`
    mutated to `return False`, the guard below PASSED while the shape
    assertion failed, which is exactly the shape of a test that guards
    nothing.
    """
    terms = _condition_taking_terms()
    assert terms, (
        "no condition-taking term exists, so this predicate has nothing to "
        "look for and would answer 'no condition here' about every spec")
    rules = json.dumps([spec.get("long"), spec.get("short"),
                        spec.get("exits", {}).get("exit")])
    return {name for name in terms if f'"{name}"' in rules}


def test_the_condition_arg_shape_actually_carries_a_condition():
    """acceptance item 10 requires 'one spec per grammar shape INCLUDING a
    condition argument'. Editing that entry's leaf to an ordinary one is a
    plausible copy edit, and no group-key assertion can see it, because a
    condition argument is not a group key.

    Do NOT write this as `"cond" in json.dumps(spec)`. That was tried and is
    vacuous: the spec's own name field is "condition-arg", which contains the
    substring, so the assertion holds with the condition leaf removed.
    Measured, with an ordinary leaf substituted:

        'cond' in json.dumps(bad) -> True   # supplied by spec["name"]

    Assert against the LEAF, by asking the vocabulary which terms declare a
    condition-typed arg and requiring one of them to appear in the group.
    That question is `_condition_terms_used_by`, shared with the guard below
    so the guard reddens on a change to it.
    """
    used = _condition_terms_used_by(SHAPE_CORPUS["condition-arg"])
    assert used, (
        f"the condition-arg corpus entry's rules declare none of "
        f"{sorted(_condition_taking_terms())}, so the fingerprint covers no "
        f"condition argument and acceptance item 10 is unmet")


def test_the_condition_arg_guard_is_not_satisfied_by_the_spec_name():
    """The predicate above, watched refusing, since its two predecessors could
    not: both re-implemented it here instead of calling it, so neither could
    see a change to the real one.

    Takes the corpus entry itself and swaps in an ordinary leaf, so the name
    supplying the "cond" substring is the corpus's own rather than a copy that
    could drift from it, and requires `_condition_terms_used_by` to answer
    empty. Simplify that predicate back to a substring search over the whole
    spec and this goes red on its own; empty the vocabulary of
    condition-taking terms and its floor takes this red too.
    """
    bad = dict(SHAPE_CORPUS["condition-arg"], long={"all": [_LEAF]})
    assert "cond" in json.dumps(bad), (
        "the substring form would have passed this, which is why it is gone")
    assert not _condition_terms_used_by(bad), (
        "a spec whose rules call no condition-taking term reads as carrying "
        "a condition argument, so the assertion above proves nothing")


def test_fingerprint_agrees_across_a_spawn_start_pool():
    ctx = mp.get_context("spawn")
    with ctx.Pool(2) as pool:
        workers = pool.starmap(vocabulary_fingerprint, [() for _ in range(2)])
    assert len({vocabulary_fingerprint(), *workers}) == 1
