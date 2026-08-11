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
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
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


def _prims_in(node, names) -> set[str]:
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
            if item.get("prim") in names:
                found.add(item["prim"])
            stack.extend(item.values())
        elif isinstance(item, list):
            seen.add(id(item))
            stack.extend(item)
    return found


# What a one-bar session does to each primitive that cannot survive it. Said
# per primitive because the NL compiler retries against this text, and "needs
# intraday bars" alone gives it nothing to reason with.
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
        term = vocabulary.primitives.get(prim)
        if term is not None and term.driving_frame_intraday and src_tf in SESSION_ALIGNED:
            errs.append(f"{at}: {prim} needs intraday bars and this one is "
                        f"evaluated on {src_tf!r}, where "
                        f"{_ONE_BAR_SESSION[prim]}; move it to an intraday "
                        "timeframe")
        stack.extend((v, src_tf, f"{at}.{k}")
                     for k, v in item.items() if k != "tf")


def _check_tf(node: dict, path: str, errs: list[str],
              allowed_extra: tuple[str, ...] = ()) -> None:
    """Rejects a `tf` not in TIMEFRAMES, and (for src leaves) any extra key
    beyond src/tf."""
    if "tf" in node and node["tf"] not in TIMEFRAMES:
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
        return
    if not isinstance(node, dict):
        errs.append(f"{path}: operand must be a number or an expression object")
        return
    if "src" in node:
        if node["src"] not in SOURCES:
            errs.append(f"{path}: unknown source {node['src']!r} (valid: {SOURCES})")
        _check_tf(node, path, errs, allowed_extra=("src",))
        return
    if "op" in node:
        op = node["op"]
        if op not in MATH_OPS:
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
        if name not in vocabulary.indicators:
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
        if name not in vocabulary.primitives:
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
    if op not in OPS:
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
    if spec.get("timeframe", "1h") not in TIMEFRAMES:
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
        return f"{node:g}"
    if "src" in node:
        return node["src"] if "tf" not in node else f"{node['src']}[{node['tf']}]"
    if "op" in node:
        op, args = node["op"], node["args"]
        if op == "abs":
            return f"abs({_expr_text(args[0], vocabulary)})"
        return "(" + f" {op} ".join(_expr_text(a, vocabulary) for a in args) + ")"
    if "ind" in node:
        name = node["ind"]
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
    return (f"{_expr_text(cond['lhs'], vocabulary)} {_OP_TEXT[cond['op']]} "
            f"{_expr_text(cond['rhs'], vocabulary)}")


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
    lines = [f"Strategy \"{spec.get('name', 'unnamed')}\" on {spec.get('timeframe', '1h')} bars."]
    for side in ("long", "short"):
        if side in spec:
            lines.append(f"Enter {side} when " + _group_text(spec[side], vocabulary))
    if "exits" in spec:
        lines.extend(_exits_text(spec["exits"], vocabulary))
    risk = spec.get("risk", DEFAULT_RISK)
    lines.append(risk_text(risk))
    return "\n".join(lines)
