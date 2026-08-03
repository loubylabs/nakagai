"""Walking a validated RuleSpec into one PineProgram.

Three rules hold the whole file together.

IDENTITY IS THE PATH. Every number a spec fixes becomes an input named after
the RuleSpec path that fixed it (nk_long_all_0_lhs_sma_n), so recompiling the
same strategy keeps a chart's saved settings, and a sanitized name that would
land on a path another one already owns stops generation rather than quietly
merging two knobs.

ONE NODE, ONE CALCULATION. Nodes are memoized on their canonical content with
the field stripped, so sma_cross's two sides read one pair of moving averages
and a MACD line and its signal come out of one ta.macd. That memo is scoped per
emission block, because a calculation made inside a request.security function
is local to that function and handing its identifier to the next request would
emit Pine that does not compile.

A FOREIGN TIMEFRAME IS ALWAYS A FUNCTION. Every request wraps its expression in
a generated function whose body ends in `value[1]`, which is the confirmed
non-repainting form: the offset is applied INSIDE the requested context, never
to the result. Doing it uniformly, rather than inlining the simple cases, is
what lets a multi-field indicator come back as one tuple and keeps the offset
off a parenthesized expression, which Pine does not index.
"""

import json
from contextlib import contextmanager
from dataclasses import replace

from nakagai.strategies.rules.canon import canonical_expr, spec_hash
from nakagai.strategies.rules.pine.lowerings import DIV, HELPERS
from nakagai.strategies.rules.pine.model import (
    GENERATOR_VERSION, PineCompileError, PineHelper, PineInput, PineProgram,
    RulePath, TermCall,
)
from nakagai.strategies.rules.spec import (
    BREAKEVEN_RR_BOUNDS, DEFAULT_RISK, STOP_ATR_MULT_BOUNDS, STOP_ATR_N_BOUNDS,
    STOP_PCT_BOUNDS, TARGET_PCT_BOUNDS, TARGET_RR_BOUNDS, TIME_STOP_BOUNDS,
    TIMEFRAMES, TRAILING_ATR_MULT_BOUNDS, TRAILING_ATR_N_BOUNDS,
    TRAILING_PCT_BOUNDS,
)
from nakagai.strategies.rules.vocabulary import Vocabulary

# The engine's timeframes in Pine's spelling. Every timeframe the grammar
# admits needs an entry; one without would otherwise reach request.security as
# a string TradingView reads as minutes.
PINE_TIMEFRAMES = {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}
REQUEST = ('request.security(syminfo.tickerid, "{tf}", {call}(), '
           "lookahead=barmerge.lookahead_on, gaps=barmerge.gaps_off)")
# Group keys carry no meaning for a reader of the settings dialog, so a label
# drops them; the index comes back only when two labels would otherwise read
# alike (see _resolve_labels).
STRUCTURAL = ("all", "any")
COMPARISONS = {"crosses_above": "ta.crossover", "crosses_below": "ta.crossunder"}


def _sanitize(part: str) -> str:
    return "".join(c if c.isalnum() and c.isascii() else "_"
                   for c in str(part)).lower()


def _label(parts, with_index: bool = False) -> str:
    keep = [p for p in parts
            if p not in STRUCTURAL and (with_index or not p.isdigit())]
    return " · ".join([keep[0].capitalize(), *keep[1:]])


def _content(node, vocabulary: Vocabulary) -> str:
    """A node's canonical form, minus the field, as one comparable string.

    The field goes because two nodes that differ only in which member of a
    tuple they read are one calculation; everything else stays, so two nodes
    that differ in an argument or a timeframe stay two.
    """
    canon = dict(canonical_expr(node, vocabulary))
    canon.pop("field", None)
    return json.dumps(canon, sort_keys=True, separators=(",", ":"))


def _typed(value, bounds, path: RulePath, term: str):
    """One number's Pine input type, default, and bounds, decided together.

    int only when both the value and its bounds are whole: TradingView refuses
    an input.int carrying a float minval, and it is the bounds rather than the
    literal that say what the knob means, so `n: 20.0` and `n: 20` (one
    canonical spec) give one program.
    """
    where = {"path": path.text, "term": term}
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PineCompileError(
            "pine_bad_input", f"{path.text}: {value!r} is not a number, so it "
            "cannot become a Pine input", **where)
    if bounds is not None:
        if (not isinstance(bounds, tuple) or len(bounds) != 2
                or any(isinstance(b, bool) or not isinstance(b, (int, float))
                       for b in bounds) or bounds[0] > bounds[1]):
            raise PineCompileError(
                "pine_bad_input", f"{path.text}: {bounds!r} is not a numeric "
                "(low, high) pair", **where)
        if not bounds[0] <= value <= bounds[1]:
            raise PineCompileError(
                "pine_bad_input", f"{path.text}: the value {value!r} is outside "
                f"its declared bounds {bounds}", **where)
    whole = float(value).is_integer() and all(
        float(b).is_integer() for b in (bounds or ()))
    cast = int if whole else float
    return (("int" if whole else "float"), cast(value),
            None if bounds is None else tuple(cast(b) for b in bounds))


class PineContext:
    """The mutable half of a lowering: symbols, inputs, statements, helpers.

    Everything a term's emit function is allowed to touch lives here, and every
    identifier it can mint passes through claim(), so the program can never
    hold two meanings for one name.
    """

    def __init__(self):
        self.warnings: list[str] = []
        self._inputs: list[tuple[RulePath, PineInput]] = []
        self._helpers: set[str] = set()
        self._owner: dict[str, str] = {}
        self._counts: dict[str, int] = {}
        self._shared: dict[tuple[str, str], str] = {}
        self._fields: dict[str, dict[str, str]] = {}
        self._sinks: list[list[str]] = [[]]
        self._scopes: list[dict[tuple[str, str], dict[str, str]]] = [{}]

    # -- statements ----------------------------------------------------
    def statement(self, text: str) -> None:
        self._sinks[-1].append(text)

    @contextmanager
    def block(self):
        """Collect statements, and memoize nodes, somewhere other than the top.

        Both stacks move together: a calculation emitted into a function body
        is only reachable from inside that body, so its memo entry has to
        disappear with it.
        """
        self._sinks.append([])
        self._scopes.append({})
        try:
            yield self._sinks[-1]
        finally:
            self._sinks.pop()
            self._scopes.pop()

    @property
    def nodes(self) -> dict:
        return self._scopes[-1]

    def calculations(self) -> tuple[str, ...]:
        """The top-level statements, in the order the walk produced them."""
        return tuple(self._sinks[0])

    # -- names ---------------------------------------------------------
    def claim(self, name: str, path: RulePath) -> str:
        owner = self._owner.get(name)
        if owner is not None and owner != path.text:
            raise PineCompileError(
                "pine_identifier_collision",
                f"{path.text}: the Pine identifier {name!r} is already owned by "
                f"{owner!r}", path=path.text)
        self._owner[name] = path.text
        return name

    def slot(self, stem: str, path: RulePath) -> str:
        """Reserve the identifier stem one node's calculations hang off."""
        self._counts[stem] = self._counts.get(stem, 0) + 1
        return self.claim(f"{stem}_{self._counts[stem]}", path)

    # -- inputs --------------------------------------------------------
    def number(self, path: RulePath, value, bounds=None, *, share=None,
               term: str = "") -> str:
        if share is not None and share in self._shared:
            return self._shared[share]
        name = self.claim("nk_" + "_".join(_sanitize(p) for p in path.parts),
                          path)
        kind, default, bounds = _typed(value, bounds, path, term)
        self._inputs.append(
            (path, PineInput(name, _label(path.parts), kind, default, bounds)))
        if share is not None:
            self._shared[share] = name
        return name

    def arg(self, call: TermCall, name: str) -> str:
        """The input identifier for one of a term's declared arguments."""
        if name not in call.term.args:
            raise PineCompileError(
                "pine_bad_input",
                f"{call.path.text}: {call.term.name} declares no argument "
                f"{name!r}, so its Pine form cannot read one",
                path=call.path.text, term=call.term.name)
        return self.number(call.path.child(call.term.name, name),
                           call.args[name], call.term.args[name],
                           share=(call.content, name), term=call.term.name)

    def inputs(self) -> tuple[PineInput, ...]:
        return _resolve_labels(self._inputs)

    # -- calculations --------------------------------------------------
    def calc(self, call: TermCall, text: str) -> str:
        """The whole term as one named calculation."""
        self.statement(f"{call.slot} = {text}")
        return call.slot

    def local(self, call: TermCall, suffix: str, text: str) -> str:
        """One named part of a term that needs more than a single line."""
        name = self.claim(f"{call.slot}_{suffix}", call.path)
        self.statement(f"{name} = {text}")
        return name

    def destructure(self, call: TermCall, suffixes, text: str) -> dict[str, str]:
        """A Pine function returning a tuple, unpacked in its own order."""
        names = {s: self.claim(f"{call.slot}_{s}", call.path) for s in suffixes}
        self.statement("[" + ", ".join(names.values()) + f"] = {text}")
        return names

    def fields(self, call: TermCall, mapping) -> dict[str, str]:
        """Record which identifier answers each of a term's field choices.

        A single-valued term never calls this. A multi-field one must, because
        a request.security lifting the node has to carry every member across,
        not only the field this particular node happened to ask for.
        """
        self._fields[call.slot] = dict(mapping)
        return self._fields[call.slot]

    def take_fields(self, slot: str) -> dict[str, str] | None:
        return self._fields.pop(slot, None)

    # -- helpers and prose ---------------------------------------------
    def helper(self, helper_id: str, path: RulePath, term: str = "") -> str:
        if helper_id not in HELPERS:
            raise PineCompileError(
                "pine_unknown_helper",
                f"{path.text}: no Pine helper is registered as {helper_id!r}",
                path=path.text, term=term)
        self._helpers.add(helper_id)
        return helper_id

    def uses(self, helper_id: str) -> bool:
        return helper_id in self._helpers

    def warn(self, text: str) -> None:
        self.warnings.append(text)

    def helper_sources(self) -> tuple[PineHelper, ...]:
        """Every required helper, dependencies first, in one stable order."""
        out: list[PineHelper] = []
        seen: set[str] = set()

        def visit(helper_id: str, path: RulePath) -> None:
            if helper_id in seen:
                return
            seen.add(helper_id)
            helper = HELPERS[self.helper(helper_id, path)]
            for dependency in sorted(helper.dependencies):
                visit(dependency, path)
            out.append(helper)

        for helper_id in sorted(self._helpers):
            visit(helper_id, RulePath(("helpers",)))
        return tuple(out)


def _resolve_labels(items) -> tuple[PineInput, ...]:
    """Give every input a label no other input in the program shares.

    Two conditions in one group differ only by their index, which a label
    drops, so `sma(20) > 0 and sma(50) > 0` would put two identical rows in the
    settings dialog with no way to tell which moved which line. The index comes
    back for exactly those, and stays out of the rest.
    """
    counts: dict[str, int] = {}
    for _, item in items:
        counts[item.label] = counts.get(item.label, 0) + 1
    return tuple(item if counts[item.label] == 1
                 else replace(item, label=_label(path.parts, with_index=True))
                 for path, item in items)


class SpecLowerer:
    """One spec, one walk, one program. Built by lower_pine, not directly."""

    def __init__(self, spec: dict, vocabulary: Vocabulary):
        self.spec = spec
        self.vocabulary = vocabulary
        self.chart = spec.get("timeframe", "1h")
        self.ctx = PineContext()
        self.lifted: set[str] = set()

    def run(self) -> PineProgram:
        long_decision = self._side("long")
        short_decision = self._side("short")
        exits = self._exits()
        risk = self._risk()
        return PineProgram(
            title=str(self.spec["name"]).strip(),
            spec_hash=spec_hash(self.spec, self.vocabulary),
            generator_version=GENERATOR_VERSION,
            inputs=self.ctx.inputs(),
            helpers=self.ctx.helper_sources(),
            calculations=self.ctx.calculations(),
            long_decision=long_decision,
            short_decision=short_decision,
            risk=risk,
            exits=exits,
            warnings=tuple(dict.fromkeys(self.ctx.warnings)),
            assumptions=self._assumptions(),
        )

    # -- blocks --------------------------------------------------------
    def _side(self, side: str) -> str:
        if side not in self.spec:
            return ""
        path = RulePath((side,))
        text = self._group(self.spec[side], path, self.chart)
        name = self.ctx.claim(f"nk_{side}_entry", path)
        self.ctx.statement(f"{name} = {text}")
        return name

    def _exits(self) -> dict[str, str]:
        exits = self.spec.get("exits", {})
        base, out = RulePath(("exits",)), {}
        if "exit" in exits:
            path = base.child("exit")
            text = self._group(exits["exit"], path, self.chart)
            name = self.ctx.claim("nk_exit_signal", path)
            self.ctx.statement(f"{name} = {text}")
            out["exit"] = name
        if "trailing" in exits:
            trailing, path = exits["trailing"], base.child("trailing")
            out["trailing_kind"] = trailing["kind"]
            if trailing["kind"] == "atr":
                out["trailing_distance"] = self._atr_distance(
                    path, "nk_exits_trailing_distance", trailing,
                    TRAILING_ATR_N_BOUNDS, TRAILING_ATR_MULT_BOUNDS)
            else:
                out["trailing_pct"] = self.ctx.number(
                    path.child("pct"), trailing.get("pct", 2.0),
                    TRAILING_PCT_BOUNDS)
        if "time_stop" in exits:
            out["time_stop_bars"] = self.ctx.number(
                base.child("time_stop", "bars"), exits["time_stop"]["bars"],
                TIME_STOP_BOUNDS)
        if "breakeven_at" in exits:
            out["breakeven_rr"] = self.ctx.number(
                base.child("breakeven_at", "rr"), exits["breakeven_at"]["rr"],
                BREAKEVEN_RR_BOUNDS)
        return out

    def _risk(self) -> dict[str, str]:
        risk = self.spec.get("risk", {})
        base, out = RulePath(("risk",)), {}
        stop = risk.get("stop", DEFAULT_RISK["stop"])
        out["stop_kind"] = stop["kind"]
        if stop["kind"] == "atr":
            out["stop_distance"] = self._atr_distance(
                base.child("stop"), "nk_risk_stop_distance", stop,
                STOP_ATR_N_BOUNDS, STOP_ATR_MULT_BOUNDS)
        else:
            out["stop_pct"] = self.ctx.number(
                base.child("stop", "pct"), stop.get("pct", 2.0), STOP_PCT_BOUNDS)
        target = risk.get("target", DEFAULT_RISK["target"])
        out["target_kind"] = target["kind"]
        if target["kind"] == "rr":
            out["target_rr"] = self.ctx.number(
                base.child("target", "rr"), target.get("rr", 2.0),
                TARGET_RR_BOUNDS)
        else:
            out["target_pct"] = self.ctx.number(
                base.child("target", "pct"), target.get("pct", 4.0),
                TARGET_PCT_BOUNDS)
        return out

    def _atr_distance(self, path: RulePath, name: str, block: dict,
                      n_bounds, mult_bounds) -> str:
        """An ATR-sized distance, as a top-level calculation.

        Named rather than inlined into the renderer because `ta.atr` has to run
        on every bar: called from inside a conditional block instead, it would
        smooth over the bars the condition happened to be true on.
        """
        n = self.ctx.number(path.child("n"), block.get("n", 14), n_bounds)
        mult = self.ctx.number(path.child("mult"), block.get("mult", 2.0),
                               mult_bounds)
        self.ctx.statement(f"{self.ctx.claim(name, path)} = ta.atr({n}) * {mult}")
        return name

    # -- conditions ----------------------------------------------------
    def _group(self, group: dict, path: RulePath, frame: str) -> str:
        key, items = next(iter(group.items()))
        joiner = " and " if key == "all" else " or "
        parts = []
        for i, item in enumerate(items):
            child = path.child(key, i)
            parts.append(self._group(item, child, frame)
                         if "all" in item or "any" in item
                         else self._condition(item, child, frame))
        return "(" + joiner.join(parts) + ")"

    def _condition(self, cond: dict, path: RulePath, frame: str) -> str:
        lhs = self._expr(cond["lhs"], path.child("lhs"), frame)
        rhs = self._expr(cond["rhs"], path.child("rhs"), frame)
        cross = COMPARISONS.get(cond["op"])
        # Comparisons bind tighter than `and`/`or` in Pine, so the enclosing
        # group's parentheses are the only ones a condition needs.
        return f"{cross}({lhs}, {rhs})" if cross else f"{lhs} {cond['op']} {rhs}"

    # -- expressions ---------------------------------------------------
    def _expr(self, node, path: RulePath, frame: str) -> str:
        if isinstance(node, (int, float)):
            return self.ctx.number(path, node)
        if "src" in node:
            return self._node(node, path, frame, f"nk_{node['src']}",
                              lambda _frame: {"": node["src"]})[""]
        if "op" in node:
            return self._math(node, path, frame)
        kind = "ind" if "ind" in node else "prim"
        name = node[kind]
        term = self.vocabulary.resolve(
            "primitive" if kind == "prim" else "indicator", name)
        if term.pine is None:
            raise PineCompileError(
                "pine_unsupported",
                f"{path.text}: {name} has no Pine lowering, so this spec "
                "cannot be generated", path=path.text, term=name)
        fields = self._node(node, path, frame, f"nk_{name}",
                            lambda inner: self._term(node, term, path, inner))
        field = str(node.get("field", term.defaults.get("field", "")))
        if field not in fields:
            raise PineCompileError(
                "pine_bad_input",
                f"{path.text}: {name}'s Pine form answers {sorted(fields)!r}, "
                f"not the field {field!r} this node asks for",
                path=path.text, term=name)
        return fields[field]

    def _math(self, node: dict, path: RulePath, frame: str) -> str:
        op = node["op"]
        args = [self._expr(a, path.child("args", i), frame)
                for i, a in enumerate(node["args"])]
        if op == "abs":
            return f"math.abs({args[0]})"
        if op == "/":
            return f"{self.ctx.helper(DIV, path)}({args[0]}, {args[1]})"
        if op in ("min", "max"):
            folded = args[0]
            for arg in args[1:]:
                folded = f"math.{op}({folded}, {arg})"
            return folded
        return "(" + f" {op} ".join(args) + ")"

    def _term(self, node: dict, term, path: RulePath,
              frame: str) -> dict[str, str]:
        args = {**term.defaults,
                **{k: v for k, v in node.items()
                   if k not in ("ind", "prim", "of", "tf", "cond")}}
        source = ""
        if term.kind in ("series", "frame"):
            source = self._expr(node.get("of", {"src": "close"}),
                                path.child("of"), frame)
        call = TermCall(term=term, args=args, path=path,
                        slot=self.ctx.slot(f"nk_{term.name}", path),
                        source=source, content=_content(node, self.vocabulary))
        for helper_id in term.pine.helpers:
            self.ctx.helper(helper_id, path, term.name)
        expr = term.pine.emit(self.ctx, call)
        return self.ctx.take_fields(call.slot) or {"": expr.text}

    # -- timeframes ----------------------------------------------------
    def _node(self, node: dict, path: RulePath, frame: str, stem: str,
              produce) -> dict[str, str]:
        """One source or term node, on its own frame or lifted onto the chart's."""
        key = (frame, _content(node, self.vocabulary))
        if key in self.ctx.nodes:
            return self.ctx.nodes[key]
        src_tf = node.get("tf", frame)
        if src_tf == frame:
            fields = produce(frame)
        elif frame != self.chart:
            raise PineCompileError(
                "pine_nested_timeframe",
                f"{path.text}: {src_tf} is read inside a {frame} request, and "
                "request.security does not nest; move the reference up to the "
                "spec's own timeframe", path=path.text)
        else:
            fields = self._lift(node, path, src_tf, stem, produce)
        self.ctx.nodes[key] = fields
        return fields

    def _lift(self, node: dict, path: RulePath, src_tf: str, stem: str,
              produce) -> dict[str, str]:
        if src_tf not in PINE_TIMEFRAMES:
            raise PineCompileError(
                "pine_unsupported",
                f"{path.text}: {src_tf!r} has no Pine timeframe string",
                path=path.text)
        with self.ctx.block() as body:
            inner = produce(src_tf)
        function = self.ctx.slot("nk_htf", path)
        # The offset sits on the values INSIDE the function, so it is applied
        # in the requested context: the request returns the last bar that had
        # already closed, which is what makes the read non-repainting.
        offsets = [f"{identifier}[1]" for identifier in inner.values()]
        returned = offsets[0] if len(offsets) == 1 else f"[{', '.join(offsets)}]"
        lines = "\n".join(f"    {line}" for line in [*body, returned])
        self.ctx.statement(f"{function}() =>\n{lines}")
        slot = self.ctx.slot(stem, path)
        outer = {field: self.ctx.claim(f"{slot}_{field}" if field else slot, path)
                 for field in inner}
        names = list(outer.values())
        target = names[0] if len(names) == 1 else f"[{', '.join(names)}]"
        self.ctx.statement(f"{target} = " + REQUEST.format(
            tf=PINE_TIMEFRAMES[src_tf], call=function))
        self.lifted.add(src_tf)
        return outer

    def _assumptions(self) -> tuple[str, ...]:
        out = [f"The chart must be on {self.chart} bars: the spec's own "
               "timeframe is charted rather than requested.",
               "Every condition is read on the close of its bar.",
               "Pine seeds its recursive averages (ema, rsi, atr and the terms "
               "built on them) differently from the engine, so the first bars "
               "of a chart can differ."]
        out += [f"{tf} values are read with request.security on the last "
                f"confirmed {tf} bar, so they do not repaint."
                for tf in TIMEFRAMES if tf in self.lifted]
        if self.ctx.uses(DIV):
            out.append("A zero denominator reads as na, so a condition over it "
                       "is false.")
        return tuple(dict.fromkeys(out))
