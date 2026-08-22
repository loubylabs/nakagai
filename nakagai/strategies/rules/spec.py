"""RuleSpec v2: the declarative strategy IR.

A spec is plain JSON, version 2 only. Operands are expression trees over
series sources, an indicator registry, math ops, and stateful primitives.
Entries are nested all/any/not condition groups per side; exits and risk are
first-class blocks. validate_spec reports precise per-path errors (the NL
compiler's retry loop feeds on them); describe_spec renders the trust-step
readback; canon.py owns identity hashing.
"""

from nakagai.data.schema import DEFAULT_TIMEFRAMES
from nakagai.strategies.rules.vocabulary import (
    Vocabulary, is_choice_rule, is_condition_rule, resolve_vocabulary,
)

VERSION = 2
SESSION_ALIGNED = DEFAULT_TIMEFRAMES.session_aligned
# The cadence the engine replays on. A spec's `timeframe` says which bars its
# CONDITIONS are computed on; it never says which bars the play decides on,
# because Engine.run walks this one whatever the spec asks for.
DRIVING = DEFAULT_TIMEFRAMES.driving
SOURCES = ("open", "high", "low", "close", "volume")
# The grammar's timeframes ARE the engine's axis. This used to be a second
# hardcoded tuple sitting two lines under the import it duplicates, so adding a
# timeframe meant editing both and a spec could name one the engine never loads.
TIMEFRAMES = DEFAULT_TIMEFRAMES.all
OPS = (">", "<", ">=", "<=", "crosses_above", "crosses_below")
CROSS_OPS = ("crosses_above", "crosses_below")
MATH_OPS: dict[str, tuple[int, int]] = {   # op -> (min arity, max arity)
    "+": (2, 8), "-": (2, 2), "*": (2, 8), "/": (2, 2),
    "abs": (1, 1), "min": (2, 8), "max": (2, 8),
}
STOP_KINDS = ("atr", "percent")
TARGET_KINDS = ("rr", "percent")
MAX_DEPTH = 8
# Every group key the grammar admits, in one tuple. Six call sites (the
# validator, the describe renderer, the evaluator, the canonicalizer, the Pine
# lowering and margins) used to spell this inline, and only the validator
# refused a key it did not know.
GROUP_KEYS = ("all", "any", "not")
MAX_CONDITIONS = 30
MAX_NODES = 40           # indicator + primitive nodes per spec
# The risk and exit blocks are the one part of the grammar whose bounds and
# defaults do not come from a Term, so both are named here rather than written
# inline in the checks below. Three readers share every pair: the validator,
# the engine that sizes a live stop from it, and the Pine compiler that puts
# the same numbers on the inputs it emits. A bound that drifted would let a
# chart be set to a value the engine refuses, and a default that drifted would
# start a chart somewhere the engine never runs.
STOP_ATR_MULT_BOUNDS = (0.1, 10.0)
STOP_ATR_MULT_DEFAULT = 2.0
STOP_ATR_N_BOUNDS = (2, 100)
STOP_ATR_N_DEFAULT = 14
STOP_PCT_BOUNDS = (0.05, 50.0)
STOP_PCT_DEFAULT = 2.0
TARGET_RR_BOUNDS = (0.1, 20.0)
TARGET_RR_DEFAULT = 2.0
TARGET_PCT_BOUNDS = (0.05, 100.0)
TARGET_PCT_DEFAULT = 4.0
TRAILING_ATR_MULT_BOUNDS = (0.5, 10.0)
TRAILING_ATR_MULT_DEFAULT = 2.0
TRAILING_ATR_N_BOUNDS = (2, 100)
TRAILING_ATR_N_DEFAULT = 14
TRAILING_PCT_BOUNDS = (0.1, 50.0)
TRAILING_PCT_DEFAULT = 2.0
# time_stop.bars and breakeven_at.rr have no default: a spec that names the
# block must name the number, so there is nothing for a reader to fall back to.
TIME_STOP_BOUNDS = (1, 500)
BREAKEVEN_RR_BOUNDS = (0.1, 10.0)

DEFAULT_RISK = {"stop": {"kind": "atr", "n": STOP_ATR_N_DEFAULT,
                         "mult": STOP_ATR_MULT_DEFAULT},
                "target": {"kind": "rr", "rr": TARGET_RR_DEFAULT}}


def is_group_node(node) -> bool:
    """True for a dict spelling any group key.

    One definition, imported by all SIX walkers over the grammar: the
    validator and the describe renderer here, the evaluator, the
    canonicalizer, the Pine lowering, and margins. Each of them used to spell
    `"all" in node or "any" in node` inline, and only the validator refused a
    key it did not know about, so a group key added in one place and forgotten
    in another is exactly the shape of bug this closes.

    Count the callers, do not trust this list. It said five when `not` landed,
    because `margins.group_margin` was missed and had to be closed a commit
    later; a walk added without importing this is invisible to every reader of
    this docstring.
    """
    return isinstance(node, dict) and any(k in node for k in GROUP_KEYS)


class _Budget:
    def __init__(self):
        self.conditions = 0
        self.nodes = 0


def _check_args(name: str, given: dict, schema: dict, path: str, errs: list[str],
                budget: _Budget, vocabulary: Vocabulary, depth: int,
                skip: tuple[str, ...] = ()) -> None:
    # A condition-typed arg may not declare a default (N3-D13), so its
    # ABSENCE is itself an error, unlike every other arg here: those fall back
    # to term.defaults at evaluation time, so the spec is free to omit them
    # and the loop below only ever walks the keys the spec actually supplied.
    for arg, rule in schema.items():
        if is_condition_rule(rule) and arg not in given:
            errs.append(f"{path}: {name} needs {arg} = {{lhs, op, rhs}}")
    for arg, v in given.items():
        if arg in skip:
            continue
        if arg not in schema:
            errs.append(f"{path}: {name} takes no arg {arg!r} (valid: {sorted(schema)})")
            continue
        rule = schema[arg]
        # The condition branch must run BEFORE the choice/range branches, not
        # as a third one after them: rule is the bare string "condition", and
        # the range branch's `lo, hi = rule` raises ValueError unpacking it.
        if is_condition_rule(rule):
            _check_condition_arg(name, arg, v, given, path, errs, budget,
                                 vocabulary, depth)
        elif is_choice_rule(rule):
            if v not in rule:
                errs.append(f"{path}: {name}.{arg} must be one of {rule}, got {v!r}")
        else:
            lo, hi = rule
            if _not_num(v, lo, hi):
                errs.append(f"{path}: {name}.{arg} must be a number in [{lo}, {hi}], got {v!r}")


def _check_condition_arg(name: str, arg: str, cond, given: dict, path: str,
                         errs: list[str], budget: _Budget,
                         vocabulary: Vocabulary, depth: int) -> None:
    """The four guards N3-D5 requires on every condition-typed arg, generic
    over the term and the arg name rather than written once per bars_since.

    Guard 1, shape. Guard 2, no cross ops (the accumulating primitives this
    guards ffill over an elementwise mask, not a momentary crossing event).
    Guard 3, no end-anchored primitive anywhere inside: those are NaN outside
    the span they are evaluated over, so a reader reaching outside it gets a
    span-dependent answer, the same bar counting differently depending on
    which walk-forward window replayed it. Guard 4, no session-scoped
    primitive inside a `tf`-qualified use: the node's own `tf` reframes this
    subtree onto a foreign frame, which would smuggle a session-scoped
    primitive there. All four apply unconditionally per N3-D5: refusing a
    spec that would have been safe is recoverable, admitting one that reaches
    outside its evaluated span is not.
    """
    # Guard 1's message names the arg as `{name}.{arg}` and shows what was
    # given, where _check_args' absent-arg message says `{name} needs {arg}`.
    # The two used to be word for word identical, which cost this guard its
    # only test: the test written for it passed the arg ABSENT, landed in the
    # other branch, and could not tell them apart, so deleting this
    # errs.append left the whole suite green. It is also the more useful
    # message, since a spec that DID supply the arg is not told it needs one.
    if not isinstance(cond, dict) or not {"lhs", "op", "rhs"} <= set(cond):
        errs.append(f"{path}: {name}.{arg} must be a condition "
                    f"{{lhs, op, rhs}}, got {cond!r}")
        return
    if cond.get("op") in CROSS_OPS:
        errs.append(f"{path}: {name}.{arg} conditions use comparison ops only")
    else:
        _check_condition(cond, f"{path}.{arg}", errs, budget, vocabulary,
                         depth + 1)
    end_anchored = {n for n, t in vocabulary.primitives.items() if t.end_anchored}
    for bad in sorted(_prims_in(cond, end_anchored)):
        errs.append(f"{path}: {bad} is anchored to the end of the frame and "
                    f"cannot sit inside {name}.{arg}")
    if "tf" in given:
        session_scoped = {n for n, t in vocabulary.primitives.items()
                          if t.session_scoped}
        for bad in sorted(_prims_in(cond, session_scoped)):
            errs.append(f"{path}: {bad} is session-scoped and cannot sit "
                        f"inside {name}.{arg} with tf")


def _canonicalizable(value: float) -> bool:
    """Whether a numeric operand survives this grammar's own canonical form.

    `canon.canonical_expr` returns `float(node)` for every numeric scalar, and
    that is load-bearing rather than incidental: it is what makes `20` and
    `20.0` one spec. So a number outside the float range has no canonical form,
    which means no `spec_hash`, which means it can be neither stored nor
    identified. Accepting it was never accepting a usable spec.

    0.6.1 dropped this check, on the argument that refusing the number changed
    the verdict on something the grammar accepts. That argument did not know
    about the canonical form. Downstream it showed up as a spec that compiled
    on attempt one and then took the platform's save path to `OverflowError:
    int too large to convert to float`, so the refusal moved from the one place
    that can explain it to the one place that cannot.

    The readback's own fallback in `_expr_text` stays, because a describer is
    read by surfaces that must not raise whatever reaches them.

    The test is exactly `float()` succeeding, and NOT `math.isfinite`. JSON has
    no infinity literal, but `1e309` parses to one, and `float(inf)` is `inf`,
    which `canonical_expr` returns and `spec_hash` hashes. So an infinity HAS a
    canonical form and is accepted here. Refusing it would be a different rule
    with a different reason (a comparison against infinity is constant), and
    smuggling that in under this one is how a guard comes to refuse more than
    it can explain.
    """
    try:
        float(value)
    except (OverflowError, ValueError):
        return False
    return True


def names(value: object, allowed) -> bool:
    """Whether an untrusted JSON value NAMES a member of `allowed`.

    Every mapping and set in this grammar is keyed by string, and `value` is
    whatever arrived in the caller's JSON. `value in allowed` raises
    `TypeError: unhashable type` on a list or an object, out of a function whose
    entire contract is to return a list of errors, and the NL builder's retry
    loop is the caller that cannot survive it: a malformed model reply became a
    503 rather than the retry the model could have acted on.

    A non-string names nothing, so it is simply not known, and the caller
    reports it with the same message it uses for a name it does not recognise.
    """
    return isinstance(value, str) and value in allowed


def _prims_in(node, names_allowed) -> set[str]:
    """Which of `names` appear as primitives anywhere in an expression tree.
    Iterative walk with a seen set: this runs before the shape checks bound
    depth, so it must not recurse or loop on adversarially deep input."""
    found: set[str] = set()
    stack, seen = [node], set()
    while stack:
        item = stack.pop()
        if id(item) in seen:
            continue
        if isinstance(item, dict):
            seen.add(id(item))
            if names(item.get("prim"), names_allowed):
                found.add(item["prim"])
            stack.extend(item.values())
        elif isinstance(item, list):
            seen.add(id(item))
            stack.extend(item)
    return found


# What a one-bar session does to each primitive that cannot survive it. Said
# per primitive because the NL compiler retries against this text, and "needs
# intraday bars" alone gives it nothing to reason with.
_OPENING_RANGE_PRIMS = frozenset({"opening_range_high", "opening_range_low"})
_ONE_BAR_SESSION = {
    "opening_range_high": "the opening-range window is the first few minutes "
                          "after the 09:30 bell and a whole-session bar cannot "
                          "sit inside it, so the level is NaN on every bar",
    "opening_range_low": "the opening-range window is the first few minutes "
                         "after the 09:30 bell and a whole-session bar cannot "
                         "sit inside it, so the level is NaN on every bar",
    "minutes_into_session": "every bar sits 0 minutes into its own session",
    "rvol": "every bar shares one clock time, so the same-clock-time baseline "
            "becomes the whole series and this reads as a plain "
            "trailing-median volume ratio (a daily relative-volume measure "
            "would be a separate primitive with its own name)",
}


def _one_bar_session(prim: str) -> str:
    """Why this primitive reads wrong on a whole-session bar, in its own words
    where core has them and in general terms where it does not.

    `_ONE_BAR_SESSION` explains CORE's primitives. The flag it explains,
    `Term.driving_frame_intraday`, is settable by any caller injecting a
    vocabulary, and this validator is reached with the caller's terms in it, so
    a subscript here raised KeyError on a term the prompt had just taught the
    model. The fallback is the flag's own meaning, which is true of every term
    that sets it.
    """
    return _ONE_BAR_SESSION.get(
        prim, "it reads a position within the trading session, and a "
              "whole-session bar has only one such position")


def _check_opening_range_window(item: dict, prim: str, src_tf: str,
                                path: str, errs: list[str], term) -> None:
    if prim not in _OPENING_RANGE_PRIMS:
        return
    delta = DEFAULT_TIMEFRAMES.deltas.get(src_tf)
    minutes = item.get("minutes", term.defaults.get("minutes"))
    if delta is None or _not_num(minutes, *term.args["minutes"]):
        return
    bar_minutes = delta.total_seconds() / 60
    if bar_minutes > minutes:
        errs.append(
            f"{path}: {prim} asks for a {minutes:g}-minute opening range "
            f"on {src_tf!r} bars, which are {bar_minutes:g} minutes wide; "
            "use a finer timeframe or widen minutes")


def _check_session_aligned_refs(node, eval_tf: str, path: str,
                                errs: list[str], vocabulary: Vocabulary) -> None:
    """Refuse what a session-aligned frame cannot answer. Two rules live here.

    The first: a cross-timeframe reference evaluated on a session-aligned
    frame. frame_eval carries a series from one timeframe onto another by
    asking when each DESTINATION bar closed, and a session-aligned label
    carries no close time: a daily bar's label holds the session date, not the
    16:00 NY bell. The evaluator refuses to guess, but it does so per symbol at
    evaluation time, inside the screener's per-symbol try/except, where a daily
    screen with one intraday reference writes an error note on every row and
    reads as a screen that simply matched nothing. Say it once here instead,
    where the NL compiler's retry loop can see it; the runtime guard stays as a
    backstop.

    The second: an intraday-only primitive whose EFFECTIVE frame is session
    aligned. That fires whether or not a foreign timeframe is involved, so it
    reads src_tf rather than comparing it to the parent's. A spec declaring
    "timeframe": "1d" and using opening_range_high used to validate clean and
    then read NaN forever; nothing raised, because the only primitive rule was
    the foreign-`tf` one, which such a spec never trips.

    The two rules run over two different sets, deliberately: see
    Term.driving_frame_intraday in vocabulary.py on why day_of_week is refused
    a foreign `tf` and welcome on daily bars.

    `eval_tf` follows the evaluator: a node's own `tf` is the frame its
    children are computed on, which is how a bars_since with a tf, or an
    indicator's `of`, moves the destination frame out from under a subtree.
    That is also what lets the second rule catch a primitive whose parent
    carries the tf, a shape the own-`tf` check in _check_expr cannot see.
    Iterative for the same reason _prims_in is.
    """
    stack, seen = [(node, eval_tf, path)], set()
    while stack:
        item, tf, at = stack.pop()
        key = (id(item), tf)
        if key in seen:
            continue
        if isinstance(item, list):
            seen.add(key)
            stack.extend((v, tf, f"{at}[{i}]") for i, v in enumerate(item))
            continue
        if not isinstance(item, dict):
            continue
        seen.add(key)
        # src_tf is the frame THIS node is evaluated on: its own tf when it
        # carries one, otherwise the frame it inherited.
        src_tf = item.get("tf", tf) if isinstance(item.get("tf", tf), str) else tf
        if src_tf != tf and tf in SESSION_ALIGNED:
            errs.append(f"{at}: {tf} is session-aligned, so a reference to "
                        f"{src_tf!r} has no well-defined visibility cutoff; "
                        "move it to an intraday timeframe")
        prim = item.get("prim")
        # `names` rather than a bare `.get`: an unhashable prim raises out of
        # the lookup, and this walk runs over the caller's whole condition tree.
        term = vocabulary.primitives[prim] if names(prim, vocabulary.primitives) else None
        if term is not None:
            _check_opening_range_window(item, prim, src_tf, at, errs, term)
        if term is not None and term.driving_frame_intraday and src_tf in SESSION_ALIGNED:
            errs.append(f"{at}: {prim} needs intraday bars and this one is "
                        f"evaluated on {src_tf!r}, where "
                        f"{_one_bar_session(prim)}; move it to an intraday "
                        "timeframe")
        stack.extend((v, src_tf, f"{at}.{k}")
                     for k, v in item.items() if k != "tf")


def _check_tf(node: dict, path: str, errs: list[str],
              allowed_extra: tuple[str, ...] = ()) -> None:
    """Rejects a `tf` not in TIMEFRAMES, and (for src leaves) any extra key
    beyond src/tf."""
    if "tf" in node and not names(node["tf"], TIMEFRAMES):
        errs.append(f"{path}: tf must be one of {TIMEFRAMES}, got {node['tf']!r}")
    if allowed_extra:
        unknown = set(node) - set(allowed_extra) - {"tf"}
        if unknown:
            errs.append(f"{path}: unknown keys {sorted(unknown)}")


def _check_expr(node, path: str, errs: list[str], budget: _Budget,
                vocabulary: Vocabulary, depth: int = 0,
                series_required: bool = False) -> None:
    if depth > MAX_DEPTH:
        errs.append(f"{path}: expression depth exceeds {MAX_DEPTH}")
        return
    if isinstance(node, bool):
        errs.append(f"{path}: booleans are not operands")
        return
    if isinstance(node, (int, float)):
        if series_required:
            errs.append(f"{path}: the left side of a cross must be a series, not a number")
        elif not _canonicalizable(node):
            errs.append(f"{path}: number is out of range")
        return
    if not isinstance(node, dict):
        errs.append(f"{path}: operand must be a number or an expression object")
        return
    if "src" in node:
        if not names(node["src"], SOURCES):
            errs.append(f"{path}: unknown source {node['src']!r} (valid: {SOURCES})")
        _check_tf(node, path, errs, allowed_extra=("src",))
        return
    if "op" in node:
        op = node["op"]
        if not names(op, MATH_OPS):
            errs.append(f"{path}: unknown math op {op!r} (valid: {sorted(MATH_OPS)})")
            return
        args = node.get("args")
        lo, hi = MATH_OPS[op]
        if not isinstance(args, list) or not lo <= len(args) <= hi:
            errs.append(f"{path}: {op!r} takes {lo}-{hi} args")
            return
        for i, a in enumerate(args):
            _check_expr(a, f"{path}.args[{i}]", errs, budget, vocabulary,
                        depth + 1)
        if set(node) - {"op", "args"}:
            errs.append(f"{path}: math nodes take only op/args")
        return
    if "ind" in node:
        budget.nodes += 1
        name = node["ind"]
        if not names(name, vocabulary.indicators):
            errs.append(f"{path}: unknown indicator {name!r} "
                        f"(valid: {sorted(vocabulary.indicators)})")
            return
        term = vocabulary.indicators[name]
        if "of" in node:
            if term.kind == "bar":
                errs.append(f"{path}: {name} works on full bars and takes no `of`")
            else:
                _check_expr(node["of"], f"{path}.of", errs, budget,
                            vocabulary, depth + 1)
        _check_args(name, node, term.args, path, errs, budget, vocabulary,
                    depth, skip=("ind", "of", "tf"))
        _check_tf(node, path, errs)
        return
    if "prim" in node:
        budget.nodes += 1
        name = node["prim"]
        if not names(name, vocabulary.primitives):
            errs.append(f"{path}: unknown primitive {name!r} "
                        f"(valid: {sorted(vocabulary.primitives)})")
            return
        term = vocabulary.primitives[name]
        if series_required and term.end_anchored:
            # An end-anchored primitive is one level read from the tail of the
            # frame, not a series, which is exactly what Term.end_anchored
            # means. crossed_above's scalar branch only ever covered the RHS,
            # and the old eval_condition returned False outright for a
            # non-Series LHS, so a spec shaped this way was permanently dead.
            # _cross_prev is symmetric, so it would now fire.
            errs.append(f"{path}: the left side of a cross must be a series; "
                        f"{name} is a level read from the end of the frame")
        _check_args(name, node, term.args, path, errs, budget, vocabulary,
                    depth, skip=("prim", "tf"))
        if "tf" in node and term.session_scoped:
            errs.append(f"{path}: {name} is session-scoped and takes no tf")
        _check_tf(node, path, errs)
        return
    errs.append(f"{path}: expression object needs one of src/ind/op/prim")


def _check_condition(cond, path: str, errs: list[str], budget: _Budget,
                     vocabulary: Vocabulary, depth: int = 0) -> None:
    budget.conditions += 1
    if budget.conditions > MAX_CONDITIONS:
        errs.append(f"{path}: more than {MAX_CONDITIONS} conditions")
        return
    if not isinstance(cond, dict) or not {"lhs", "op", "rhs"} <= set(cond):
        errs.append(f"{path}: condition needs lhs, op, rhs")
        return
    op = cond["op"]
    if not names(op, OPS):
        errs.append(f"{path}: unknown op {op!r} (valid: {OPS})")
    _check_expr(cond["lhs"], f"{path}.lhs", errs, budget, vocabulary, depth,
                series_required=op in CROSS_OPS)
    _check_expr(cond["rhs"], f"{path}.rhs", errs, budget, vocabulary, depth)
    if op in CROSS_OPS:
        # series_required only ever inspects the operand's TOP node, so it saw
        # {"prim": "fvg_nearest"} and not {"op": "*", "args": [that, 1.0]}. The
        # nested form is not the same rule read from one level down, it is a
        # different rule:
        #
        #   _cross_prev broadcasts an end-anchored operand rather than shifting
        #   it, because a level has no honest history to shift (see its
        #   docstring). It matches on the node itself, so nesting it under any
        #   math op hides it, and the whole computed series gets .shift(1) --
        #   the reading this grammar deliberately does NOT use.
        #
        #   Worse, it is span-dependent. End-anchored primitives are NaN outside
        #   the span, so on a real SPY 15m frame the same nested condition gives
        #   1089 crossings under a replay span and 0 under a scan span. Same
        #   bars, same spec, different answer.
        #
        # So the only defined shape is a bare end-anchored primitive, and only
        # on the right (the left stays refused above). Anything deeper is
        # refused here rather than silently evaluated under the other reading.
        for side in ("lhs", "rhs"):
            node = cond[side]
            top = node.get("prim") if isinstance(node, dict) else None
            # `{top}` below builds a SET, so an unhashable prim raises there
            # even though nothing looks it up. A non-string names no primitive,
            # so it stands for "no bare top" exactly as a missing key does.
            top = top if isinstance(top, str) else None
            end_anchored = {n for n, t in vocabulary.primitives.items()
                            if t.end_anchored}
            nested = _prims_in(node, end_anchored) - {top}
            for bad in sorted(nested):
                errs.append(f"{path}.{side}: {bad} is anchored to the end of "
                            f"the frame and cannot be nested inside a cross "
                            f"operand; use it bare on the right of the cross")
    if budget.nodes > MAX_NODES:
        errs.append(f"{path}: more than {MAX_NODES} indicator/primitive nodes")


def _check_group(group, path: str, errs: list[str], budget: _Budget,
                 vocabulary: Vocabulary, depth: int = 0) -> None:
    if depth > MAX_DEPTH:
        errs.append(f"{path}: group depth exceeds {MAX_DEPTH}")
        return
    if (not isinstance(group, dict) or len(group) != 1
            or next(iter(group)) not in GROUP_KEYS):
        errs.append(f"{path}: expected {{\"all\": [...]}}, {{\"any\": [...]}}, "
                    "or {\"not\": {...}}")
        return
    key, val = next(iter(group.items()))
    if key == "not":
        # N3-D6: `not` takes a GROUP, never a bare leaf. One accepted shape
        # rather than two, so {"not": {"all": [<leaf>]}} is how a single
        # condition is negated, and the refusal names that form because an
        # error a user cannot act on is a worse product than a refusal.
        # N3-D7: `not` may contain `not` directly, and it counts against
        # MAX_DEPTH like any other group, which the recursive call enforces
        # the same way it does for all/any.
        if not (isinstance(val, dict) and len(val) == 1
                and next(iter(val)) in GROUP_KEYS):
            errs.append(f"{path}.not: expected a group ({{\"all\": [...]}}, "
                        "{\"any\": [...]}, or {\"not\": {...}}), not a bare "
                        "condition")
            return
        _check_group(val, f"{path}.not", errs, budget, vocabulary, depth + 1)
        return
    if not isinstance(val, list) or not val:
        errs.append(f"{path}.{key}: must be a non-empty list")
        return
    for i, item in enumerate(val):
        p = f"{path}.{key}[{i}]"
        if is_group_node(item):
            _check_group(item, p, errs, budget, vocabulary, depth + 1)
            continue
        _check_condition(item, p, errs, budget, vocabulary)


def _not_num(v, lo, hi) -> bool:
    """True when v is not a plain int/float in [lo, hi]; bools are excluded.
    This never raises, so it is safe to call on any user-supplied value
    (a string, a list, None) before it ever reaches float()/int()."""
    return isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi


def _span(bounds) -> str:
    """A bounds pair as the error messages spell it: [0.1, 10], not [0.1, 10.0]."""
    return f"[{bounds[0]:g}, {bounds[1]:g}]"


def _check_exits(exits, errs: list[str], budget: _Budget,
                 vocabulary: Vocabulary) -> None:
    if not isinstance(exits, dict):
        errs.append("exits must be an object")
        return
    unknown = set(exits) - {"exit", "trailing", "time_stop", "breakeven_at"}
    if unknown:
        errs.append(f"exits: unknown keys {sorted(unknown)}")
    if "exit" in exits:
        _check_group(exits["exit"], "exits.exit", errs, budget, vocabulary)
    if "trailing" in exits:
        t = exits["trailing"]
        if not isinstance(t, dict) or t.get("kind") not in ("atr", "percent"):
            errs.append("exits.trailing.kind must be 'atr' or 'percent'")
        elif t["kind"] == "atr":
            if _not_num(t.get("mult", TRAILING_ATR_MULT_DEFAULT), *TRAILING_ATR_MULT_BOUNDS):
                errs.append(f"exits.trailing.mult must be in {_span(TRAILING_ATR_MULT_BOUNDS)}")
            if _not_num(t.get("n", TRAILING_ATR_N_DEFAULT), *TRAILING_ATR_N_BOUNDS):
                errs.append(f"exits.trailing.n must be in {_span(TRAILING_ATR_N_BOUNDS)}")
            unknown_t = set(t) - {"kind", "mult", "n"}
            if unknown_t:
                errs.append(f"exits.trailing: unknown keys {sorted(unknown_t)}")
        else:
            if _not_num(t.get("pct", TRAILING_PCT_DEFAULT), *TRAILING_PCT_BOUNDS):
                errs.append(f"exits.trailing.pct must be in {_span(TRAILING_PCT_BOUNDS)}")
            unknown_t = set(t) - {"kind", "pct"}
            if unknown_t:
                errs.append(f"exits.trailing: unknown keys {sorted(unknown_t)}")
    if "time_stop" in exits:
        ts = exits["time_stop"]
        b = ts.get("bars") if isinstance(ts, dict) else None
        lo, hi = TIME_STOP_BOUNDS
        if isinstance(b, bool) or not isinstance(b, int) or not lo <= b <= hi:
            errs.append(f"exits.time_stop.bars must be an integer in [{lo}, {hi}]")
    if "breakeven_at" in exits:
        ba = exits["breakeven_at"]
        r = ba.get("rr") if isinstance(ba, dict) else None
        lo, hi = BREAKEVEN_RR_BOUNDS
        if isinstance(r, bool) or not isinstance(r, (int, float)) or not lo <= r <= hi:
            errs.append(f"exits.breakeven_at.rr must be a number in [{lo}, {hi}]")


def validate_risk(risk) -> list[str]:
    """Risk-block validation shared by rule specs and composite specs."""
    errs: list[str] = []
    if not isinstance(risk, dict):
        return ["risk must be an object"]
    stop = risk.get("stop", DEFAULT_RISK["stop"])
    if not isinstance(stop, dict):
        errs.append("risk.stop must be an object")
    elif stop.get("kind") not in STOP_KINDS:
        errs.append(f"risk.stop.kind must be one of {STOP_KINDS}")
    elif stop["kind"] == "atr":
        if _not_num(stop.get("mult", STOP_ATR_MULT_DEFAULT), *STOP_ATR_MULT_BOUNDS):
            errs.append(f"risk.stop.mult must be in {_span(STOP_ATR_MULT_BOUNDS)}")
        if _not_num(stop.get("n", STOP_ATR_N_DEFAULT), *STOP_ATR_N_BOUNDS):
            errs.append(f"risk.stop.n must be in {_span(STOP_ATR_N_BOUNDS)}")
    elif _not_num(stop.get("pct", STOP_PCT_DEFAULT), *STOP_PCT_BOUNDS):
        errs.append(f"risk.stop.pct must be in {_span(STOP_PCT_BOUNDS)}")
    target = risk.get("target", DEFAULT_RISK["target"])
    if not isinstance(target, dict):
        errs.append("risk.target must be an object")
    elif target.get("kind") not in TARGET_KINDS:
        errs.append(f"risk.target.kind must be one of {TARGET_KINDS}")
    elif target["kind"] == "rr":
        if _not_num(target.get("rr", TARGET_RR_DEFAULT), *TARGET_RR_BOUNDS):
            errs.append(f"risk.target.rr must be in {_span(TARGET_RR_BOUNDS)}")
    elif _not_num(target.get("pct", TARGET_PCT_DEFAULT), *TARGET_PCT_BOUNDS):
        errs.append(f"risk.target.pct must be in {_span(TARGET_PCT_BOUNDS)}")
    return errs


def risk_text(risk: dict) -> str:
    """The 'Stop: … Target: …' sentence shared by both describe functions."""
    stop = risk.get("stop", DEFAULT_RISK["stop"])
    target = risk.get("target", DEFAULT_RISK["target"])
    atr_stop = (f"{stop.get('mult', STOP_ATR_MULT_DEFAULT):g}x "
                f"ATR({int(stop.get('n', STOP_ATR_N_DEFAULT))}) from entry")
    stop_text = (atr_stop if stop.get("kind") == "atr"
                 else f"{stop.get('pct', STOP_PCT_DEFAULT):g}% from entry")
    target_text = (f"{target.get('rr', TARGET_RR_DEFAULT):g}x the risked distance"
                   if target.get("kind") == "rr"
                   else f"{target.get('pct', TARGET_PCT_DEFAULT):g}% from entry")
    return f"Stop: {stop_text}. Target: {target_text}."


def validate_spec(spec, vocabulary: Vocabulary | None = None) -> list[str]:
    """Structural + semantic validation. Empty list = usable spec."""
    vocabulary = resolve_vocabulary(vocabulary)
    if not isinstance(spec, dict):
        return ["spec must be a JSON object"]
    errs: list[str] = []
    if spec.get("version") != VERSION:
        errs.append(f"spec version must be {VERSION} (got {spec.get('version')!r})")
    if not str(spec.get("name", "")).strip():
        errs.append("spec needs a name")
    if not names(spec.get("timeframe", "1h"), TIMEFRAMES):
        errs.append(f"timeframe must be one of {TIMEFRAMES}")
    sides = [s for s in ("long", "short") if s in spec]
    if not sides:
        errs.append("spec needs at least one of long/short entry groups")
    budget = _Budget()
    # An unusable timeframe is already reported above; fall back to an intraday
    # one so the cross-timeframe walk below adds nothing on top of that, and
    # never trips over a non-string.
    tf = spec.get("timeframe", "1h")
    tf = tf if tf in TIMEFRAMES else "1h"
    for side in sides:
        _check_group(spec[side], side, errs, budget, vocabulary)
        _check_session_aligned_refs(spec[side], tf, side, errs, vocabulary)
    if "exits" in spec:
        _check_exits(spec["exits"], errs, budget, vocabulary)
        if isinstance(spec["exits"], dict) and "exit" in spec["exits"]:
            _check_session_aligned_refs(spec["exits"]["exit"], tf,
                                        "exits.exit", errs, vocabulary)
    errs.extend(validate_risk(spec.get("risk", {})))
    return errs


def validate_condition_group(group, path: str = "conditions",
                             tf: str = "1h", *,
                             vocabulary: Vocabulary | None = None) -> list[str]:
    """Standalone validation of one all/any/not condition group with a fresh
    budget. The screener's whole schema is one such group; validate_spec's
    per-side entry groups go through the same walker. `tf` is the timeframe the
    group is evaluated on, which decides whether a cross-timeframe reference
    inside it has a visibility cutoff at all.

    `vocabulary` is keyword-only, and that is load-bearing rather than a style
    choice. It sits behind `tf`, so a caller writing the shape that reads
    naturally, validate_condition_group(group, "conditions", vocab), would bind
    the Vocabulary to `tf` and leave vocabulary None. Nothing would raise: both
    `tf in SESSION_ALIGNED` and `tf if tf in TIMEFRAMES else "1h"` simply read
    False for a non-string, so the call would return a clean-looking error list
    validated against the CORE vocabulary while the caller believed it had
    injected one. A keyword-only parameter turns that into a TypeError at the
    call site."""
    vocabulary = resolve_vocabulary(vocabulary)
    errs: list[str] = []
    _check_group(group, path, errs, _Budget(), vocabulary)
    _check_session_aligned_refs(group, tf, path, errs, vocabulary)
    return errs


def group_text(group: dict, vocabulary: Vocabulary | None = None) -> str:
    """Public name for the group renderer; the screener readback reuses it."""
    return _group_text(group, resolve_vocabulary(vocabulary))


def _expr_text(node, vocabulary: Vocabulary) -> str:
    if isinstance(node, (int, float)):
        try:
            return f"{node:g}"
        except (OverflowError, ValueError):
            # `:g` goes through float, which overflows on an int past the float
            # range. The grammar accepts any JSON number, so the readback
            # renders any JSON number: exactly, here, rather than refusing a
            # spec that is merely eccentric. Refusing it at validation was the
            # first fix and it changed the verdict on an input that was legal.
            return str(node)
    if "src" in node:
        return node["src"] if "tf" not in node else f"{node['src']}[{node['tf']}]"
    if "op" in node:
        op, args = node["op"], node["args"]
        if op == "abs":
            return f"abs({_expr_text(args[0], vocabulary)})"
        return "(" + f" {op} ".join(_expr_text(a, vocabulary) for a in args) + ")"
    if "ind" in node:
        name = node["ind"]
        # A describer renders whatever it is handed. An unknown or non-string
        # name is the validator's to refuse, not this function's to raise on:
        # it is called on a spec the caller has usually validated first, and
        # "usually" is not a contract.
        if not names(name, vocabulary.indicators):
            return repr(name)
        term = vocabulary.indicators[name]
        args = {**term.defaults,
                **{k: v for k, v in node.items() if k not in ("ind", "of", "tf")}}
        field = args.pop("field", None)
        parts = [f"{v}" for v in args.values()]
        if term.kind != "bar":
            of = node.get("of", {"src": "close"})
            if of != {"src": "close"}:
                parts.append(f"of={_expr_text(of, vocabulary)}")
        inner = ", ".join(parts)
        text = f"{name}({inner})" if inner else name
        if "tf" in node:
            text += f"[{node['tf']}]"
        return f"{text}.{field}" if field else text
    name = node["prim"]
    if not names(name, vocabulary.primitives):
        return repr(name)
    term = vocabulary.primitives[name]
    condition_args = {a for a, rule in term.args.items() if is_condition_rule(rule)}
    args = {**term.defaults,
            **{k: v for k, v in node.items()
               if k not in ("prim", "tf", *condition_args)}}
    if "minutes" in args:
        minutes = args.pop("minutes")
        parts = [f"{minutes}m"] + [f"{v}" for v in args.values()]
    else:
        parts = [f"{v}" for v in args.values()]
    # Rendered in readable {lhs op rhs} form via _condition_text, not the
    # generic f"{v}" path just above: that would stringify the condition dict
    # with Python's default repr, which is what a user approves before saving
    # or backtesting an imported or NL-built strategy.
    parts += [_condition_text(node[a], vocabulary)
              for a in sorted(condition_args) if a in node]
    inner = ", ".join(parts)
    text = f"{name}({inner})" if inner else name
    if "tf" in node:
        text += f"[{node['tf']}]"
    return text


_OP_TEXT = {">": "is above", "<": "is below", ">=": "is at or above",
            "<=": "is at or below", "crosses_above": "crosses above",
            "crosses_below": "crosses below"}


def _condition_text(cond: dict, vocabulary: Vocabulary) -> str:
    # `_OP_TEXT[op]` is the last lookup in this file that took a caller value
    # straight to a subscript. An op the grammar does not define renders as
    # itself, which is what a reader needs to see anyway.
    op = cond.get("op")
    text = _OP_TEXT[op] if names(op, _OP_TEXT) else repr(op)
    return (f"{_expr_text(cond.get('lhs'), vocabulary)} {text} "
            f"{_expr_text(cond.get('rhs'), vocabulary)}")


def _group_text(group, vocabulary: Vocabulary, depth: int = 0) -> str:
    """One group's readback, every line indented by its OWN depth already.

    Recursion passes `depth + 1` and nothing else, so a nested block's lines
    are correct the moment they are produced and the caller never re-indents a
    child's text after the fact. The previous string-`.replace()` scheme did
    exactly that second pass, and it double-counted: a grandchild leaf came
    out six spaces deep instead of four. Nothing caught it because no test
    asserted exact whitespace on a nested group; N3-D11 freezes the `not`
    readback as goldens, which is what exposed it.
    """
    indent = "  " * depth
    key, val = next(iter(group.items()))
    if key == "not":
        # N3-D11's frozen shape: NOT prefixes the inner group's own joiner
        # line IN PLACE, at the same depth, rather than adding a level of its
        # own. Everything under it (its list items, or a nested NOT's own
        # prefix) keeps the depth it would have had without the negation, so
        # the scope of the negation is visible from indentation alone.
        inner = _group_text(val, vocabulary, depth)
        head, _, tail = inner.partition("\n")
        return f"{indent}NOT {head[len(indent):]}" + (f"\n{tail}" if tail else "")
    joiner = "ALL of:" if key == "all" else "ANY of:"
    lines = [f"{indent}{joiner}"]
    for item in val:
        if is_group_node(item):
            lines.append(_group_text(item, vocabulary, depth + 1))
        else:
            lines.append(f"{indent}  - {_condition_text(item, vocabulary)}")
    return "\n".join(lines)


def _exits_text(exits: dict, vocabulary: Vocabulary) -> list[str]:
    lines = []
    if "exit" in exits:
        lines.append("Exit early when " + _group_text(exits["exit"], vocabulary))
    if "trailing" in exits:
        t = exits["trailing"]
        if t["kind"] == "atr":
            lines.append(
                f"Trailing stop: {t.get('mult', TRAILING_ATR_MULT_DEFAULT):g}x "
                f"ATR({int(t.get('n', TRAILING_ATR_N_DEFAULT))}).")
        else:
            lines.append(f"Trailing stop: {t.get('pct', TRAILING_PCT_DEFAULT):g}% "
                         "from the high water mark.")
    if "time_stop" in exits:
        lines.append(f"Time stop: {exits['time_stop']['bars']} 15-minute bars.")
    if "breakeven_at" in exits:
        lines.append(f"Move stop to breakeven at {exits['breakeven_at']['rr']:g}R.")
    return lines


def describe_spec(spec: dict, vocabulary: Vocabulary | None = None) -> str:
    """Plain-English restatement of a validated spec: the trust step shown to
    the user before they save or backtest an imported/NL-built strategy."""
    vocabulary = resolve_vocabulary(vocabulary)
    # A describer is a validator's twin: both are handed whatever arrived, and
    # both are read by a surface that must not 500. This one is called on a spec
    # the caller has usually validated first, but "usually" is not a contract.
    if not isinstance(spec, dict):
        return "Not a strategy spec."
    lines = [f"Strategy \"{spec.get('name', 'unnamed')}\" on {spec.get('timeframe', '1h')} bars."]
    for side in ("long", "short"):
        if side in spec:
            lines.append(f"Enter {side} when " + _group_text(spec[side], vocabulary))
    if "exits" in spec:
        lines.extend(_exits_text(spec["exits"], vocabulary))
    risk = spec.get("risk", DEFAULT_RISK)
    lines.append(risk_text(risk))
    return "\n".join(lines)
