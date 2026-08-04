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
from nakagai.strategies.rules.pine.lowerings import DAY_OF_WEEK, DIV, HELPERS
from nakagai.strategies.rules.pine.model import (
    GENERATOR_VERSION, PineCompileError, PineExits, PineHelper, PineInput,
    PineProgram, PineRisk, RulePath, TermCall,
)
from nakagai.strategies.rules.spec import (
    BREAKEVEN_RR_BOUNDS, DEFAULT_RISK, SESSION_ALIGNED, STOP_ATR_MULT_BOUNDS,
    STOP_ATR_MULT_DEFAULT, STOP_ATR_N_BOUNDS, STOP_ATR_N_DEFAULT,
    STOP_PCT_BOUNDS, STOP_PCT_DEFAULT, TARGET_PCT_BOUNDS, TARGET_PCT_DEFAULT,
    TARGET_RR_BOUNDS, TARGET_RR_DEFAULT, TIME_STOP_BOUNDS, TIMEFRAMES,
    TRAILING_ATR_MULT_BOUNDS, TRAILING_ATR_MULT_DEFAULT, TRAILING_ATR_N_BOUNDS,
    TRAILING_ATR_N_DEFAULT, TRAILING_PCT_BOUNDS, TRAILING_PCT_DEFAULT,
)
from nakagai.strategies.rules.vocabulary import Vocabulary, is_choice_rule

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


def _label(parts, with_index: bool = False, with_block: bool = True) -> str:
    """One input's row in the settings dialog, read off its path.

    The leading block name goes when with_block is False, which is how an
    input more than one block reaches is labelled: sma_cross's two sides share
    one moving average, and heading its row "Long" would tell a chart's user
    that moving it retunes the long side when it retunes both.
    """
    keep = [p for p in parts
            if p not in STRUCTURAL and (with_index or not p.isdigit())]
    if not with_block:
        return " · ".join(keep[1:])
    return " · ".join([keep[0].capitalize(), *keep[1:]])


def _term_name(node) -> str:
    """The vocabulary term a node names, or "" for a source or a math op."""
    return str(node.get("ind") or node.get("prim") or "")


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


# One memoized node: the identifier answering each of its fields, and every
# input its production reached.
_Memo = tuple[dict[str, str], frozenset[str]]


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
        # One memo entry per node: the identifiers it answers, and every input
        # its production touched, so a later block hitting the memo can still
        # say it reached those inputs.
        self._scopes: list[dict[tuple[str, str], _Memo]] = [{}]
        self._blocks: dict[str, set[str]] = {}
        self._reaching: list[set[str]] = []
        self._history = 0

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
            return self._touch(self._shared[share], path)
        name = self.claim("nk_" + "_".join(_sanitize(p) for p in path.parts),
                          path)
        kind, default, bounds = _typed(value, bounds, path, term)
        # The label is left blank here and settled once, in _resolve_labels:
        # what a row should read depends on the whole program, not on this path.
        self._inputs.append((path, PineInput(name, "", kind, default, bounds)))
        if share is not None:
            self._shared[share] = name
        return self._touch(name, path)

    def _rule(self, call: TermCall, name: str):
        """One of a term's declared argument rules, or a refusal naming it."""
        if name not in call.term.args:
            raise PineCompileError(
                "pine_bad_input",
                f"{call.path.text}: {call.term.name} declares no argument "
                f"{name!r}, so its Pine form cannot read one",
                path=call.path.text, term=call.term.name)
        return call.term.args[name]

    def arg(self, call: TermCall, name: str) -> str:
        """The input identifier for one of a term's declared arguments.

        The rule is read before the value, so an argument the schema never
        declared is refused by name rather than raising a KeyError on args.
        """
        rule = self._rule(call, name)
        return self.number(call.path.child(call.term.name, name),
                           call.args[name], rule,
                           share=(call.content, name), term=call.term.name)

    def choice(self, call: TermCall, name: str) -> str:
        """One of a term's declared choice arguments, as the name it picks.

        A choice never becomes an input, and that is the point: it decides
        which Pine the lowering writes (which gaps a scan keeps, which way a
        retracement is measured), so a knob for it on the chart would let a
        user select a strategy the compiler never generated. The rule is read
        the same way arg() reads a numeric one, so a term whose own default
        drifted outside its declared choices is refused by name rather than
        silently baking an unknown literal into the artifact.
        """
        rule = self._rule(call, name)
        value = call.args.get(name)
        if not is_choice_rule(rule) or value not in rule:
            raise PineCompileError(
                "pine_bad_input",
                f"{call.path.text}: {call.term.name}.{name} must be one of "
                f"{rule}, got {value!r}",
                path=call.path.text, term=call.term.name)
        return str(value)

    def inputs(self) -> tuple[PineInput, ...]:
        return _resolve_labels(self._inputs, self._blocks)

    # -- reach ---------------------------------------------------------
    def _touch(self, name: str, path: RulePath) -> str:
        """Record that the block at `path` reaches the input `name`."""
        if path.parts:
            self._blocks.setdefault(name, set()).add(path.parts[0])
        for frame in self._reaching:
            frame.add(name)
        return name

    @contextmanager
    def reaching(self):
        """Collect every input the enclosed production reaches.

        Nested frames merge outwards, so a node's set covers the inputs of the
        nodes it is built from as well as its own.
        """
        frame: set[str] = set()
        self._reaching.append(frame)
        try:
            yield frame
        finally:
            self._reaching.pop()
            for outer in self._reaching:
                outer |= frame

    def reached(self, names, path: RulePath) -> None:
        """Attribute an already-built node's inputs to another path's block."""
        for name in names:
            self._touch(name, path)

    def needs_history(self, call: TermCall, name: str, reach: int = 1, *,
                      window: bool = False) -> None:
        """Declare that this term indexes history by a non-constant offset.

        TradingView reads a series' historical buffer off the constant offsets
        it can see, so an offset carried by an input makes it refuse with
        "Pine cannot determine the referencing length of series" unless the
        script declares max_bars_back. What the script owes is a BUFFER SIZE,
        which is one more than the deepest offset it indexes: reading `x[20]`
        needs 21 values, this bar's and twenty behind it.

        The argument's own upper bound says how deep that reach goes, in one of
        two readings, and they differ by exactly that one:

        - an OFFSET (the default). `reach` multiplies it where the term indexes
          a multiple: a swing confirmed k bars after its pivot compares that
          pivot against the k bars on each side, so it indexes [2k] and needs
          2k + 1 values.
        - a WINDOW of that many bars ending at this one, which is what an
          end-anchored scan takes. A 200-bar lookback indexes [0] through
          [199], so the count IS the buffer.
        """
        deepest = reach * int(self._rule(call, name)[1])
        self._history = max(self._history, deepest if window else deepest + 1)

    def max_bars_back(self) -> int:
        return self._history

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

        The mapping is checked against the term's declared choices here rather
        than after emit returns, because every lowering ends in
        `ctx.fields(...)[call.field]`: a field the mapping missed would be a
        raw KeyError out of the emit function, naming neither term nor path.
        """
        missing = [f for f in self._rule(call, "field") if f not in mapping]
        if missing:
            raise PineCompileError(
                "pine_bad_input",
                f"{call.path.text}: {call.term.name}'s Pine form answers "
                f"{sorted(mapping)!r} and so leaves the declared field(s) "
                f"{missing!r} with nothing behind them",
                path=call.path.text, term=call.term.name)
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
        """Every required helper, dependencies first, in one stable order.

        A Pine function may only call a function already defined above it, so
        the order is a topological sort of the graph the required helpers
        reach. Ties break lexically, and a whole ready layer is emitted at once
        in that order, so the same set of helpers comes out in the same order
        whatever order the walk happened to reach them in. That is what makes
        two compilations of one spec byte-identical rather than merely
        equivalent.

        Both ways the graph can be wrong stop generation rather than emitting
        Pine that TradingView reports from the chart: a dependency no helper is
        registered as would call an undefined function, and a cycle has no
        order at all.
        """
        needed: dict[str, PineHelper] = {}
        frontier = [(helper_id, "the program") for helper_id in self._helpers]
        while frontier:
            helper_id, named_by = frontier.pop()
            if helper_id in needed:
                continue
            helper = HELPERS.get(helper_id)
            if helper is None:
                raise PineCompileError(
                    "pine_generation_failed",
                    f"{named_by} needs the Pine helper {helper_id!r}, which no "
                    "helper is registered as")
            needed[helper_id] = helper
            frontier.extend((d, repr(helper_id)) for d in helper.dependencies)
        out: list[PineHelper] = []
        emitted: set[str] = set()
        while len(emitted) < len(needed):
            ready = sorted(i for i, h in needed.items()
                           if i not in emitted and set(h.dependencies) <= emitted)
            if not ready:
                raise PineCompileError(
                    "pine_generation_failed",
                    "the Pine helpers "
                    f"{sorted(set(needed) - emitted)} call each other in a "
                    "cycle, so no order defines each one above its callers")
            out.extend(needed[i] for i in ready)
            emitted.update(ready)
        # The closure, not just what the walk declared. uses() answers what the
        # PROGRAM reaches, and a helper pulled in behind another one is as
        # present in the artifact as a directly declared one: without this, a
        # future helper depending on nk_div would emit the divide and drop the
        # zero-denominator assumption that explains it.
        self._helpers.update(needed)
        return tuple(out)


def _resolve_labels(items, blocks: dict[str, set[str]]) -> tuple[PineInput, ...]:
    """Settle every label at once: the one decision that needs the whole program.

    Two things are only knowable here. An input that more than one top-level
    block reaches belongs to none of them, so its label drops the block name
    rather than claiming a side it does not own. And two conditions in one
    group differ only by their index, which a label drops, so
    `sma(20) > 0 and sma(50) > 0` would otherwise put two identical rows in the
    settings dialog with no way to tell which moved which line: the index comes
    back for exactly those, and stays out of the rest.
    """

    def label(path: RulePath, item: PineInput, with_index: bool = False) -> str:
        return _label(path.parts, with_index=with_index,
                      with_block=len(blocks.get(item.name, ())) < 2)

    plain = [label(path, item) for path, item in items]
    counts: dict[str, int] = {}
    for text in plain:
        counts[text] = counts.get(text, 0) + 1
    return tuple(replace(item, label=text if counts[text] == 1
                         else label(path, item, with_index=True))
                 for (path, item), text in zip(items, plain))


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
        # Resolved before the assumptions, and that order is load-bearing:
        # helper_sources settles the transitive closure, and _assumptions asks
        # what the program reaches.
        helpers = self.ctx.helper_sources()
        return PineProgram(
            title=str(self.spec["name"]).strip(),
            spec_hash=spec_hash(self.spec, self.vocabulary),
            generator_version=GENERATOR_VERSION,
            inputs=self.ctx.inputs(),
            helpers=helpers,
            calculations=self.ctx.calculations(),
            long_decision=long_decision,
            short_decision=short_decision,
            risk=risk,
            exits=exits,
            max_bars_back=self.ctx.max_bars_back(),
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

    def _exits(self) -> PineExits:
        exits = self.spec.get("exits", {})
        base, out = RulePath(("exits",)), {}
        if "exit" in exits:
            path = base.child("exit")
            text = self._group(exits["exit"], path, self.chart)
            name = self.ctx.claim("nk_exit_signal", path)
            self.ctx.statement(f"{name} = {text}")
            out["signal"] = name
        if "trailing" in exits:
            trailing, path = exits["trailing"], base.child("trailing")
            out["trailing_kind"] = trailing["kind"]
            out["trailing"] = (
                self._atr_distance(
                    path, "nk_exits_trailing_distance", trailing,
                    (TRAILING_ATR_N_BOUNDS, TRAILING_ATR_N_DEFAULT),
                    (TRAILING_ATR_MULT_BOUNDS, TRAILING_ATR_MULT_DEFAULT))
                if trailing["kind"] == "atr" else
                self.ctx.number(path.child("pct"),
                                trailing.get("pct", TRAILING_PCT_DEFAULT),
                                TRAILING_PCT_BOUNDS))
        if "time_stop" in exits:
            out["time_stop_bars"] = self.ctx.number(
                base.child("time_stop", "bars"), exits["time_stop"]["bars"],
                TIME_STOP_BOUNDS)
        if "breakeven_at" in exits:
            out["breakeven_rr"] = self.ctx.number(
                base.child("breakeven_at", "rr"), exits["breakeven_at"]["rr"],
                BREAKEVEN_RR_BOUNDS)
        return PineExits(**out)

    def _risk(self) -> PineRisk:
        risk = self.spec.get("risk", {})
        base = RulePath(("risk",))
        stop = risk.get("stop", DEFAULT_RISK["stop"])
        target = risk.get("target", DEFAULT_RISK["target"])
        return PineRisk(
            stop_kind=stop["kind"],
            stop=(self._atr_distance(
                      base.child("stop"), "nk_risk_stop_distance", stop,
                      (STOP_ATR_N_BOUNDS, STOP_ATR_N_DEFAULT),
                      (STOP_ATR_MULT_BOUNDS, STOP_ATR_MULT_DEFAULT))
                  if stop["kind"] == "atr" else
                  self.ctx.number(base.child("stop", "pct"),
                                  stop.get("pct", STOP_PCT_DEFAULT),
                                  STOP_PCT_BOUNDS)),
            target_kind=target["kind"],
            target=(self.ctx.number(base.child("target", "rr"),
                                    target.get("rr", TARGET_RR_DEFAULT),
                                    TARGET_RR_BOUNDS)
                    if target["kind"] == "rr" else
                    self.ctx.number(base.child("target", "pct"),
                                    target.get("pct", TARGET_PCT_DEFAULT),
                                    TARGET_PCT_BOUNDS)))

    def _atr_distance(self, path: RulePath, name: str, block: dict,
                      n_rule, mult_rule) -> str:
        """An ATR-sized distance, as a top-level calculation.

        Named rather than inlined into the renderer because `ta.atr` has to run
        on every bar: called from inside a conditional block instead, it would
        smooth over the bars the condition happened to be true on.

        n_rule and mult_rule are each the (bounds, default) pair spec.py names
        for that argument, so the chart starts where the engine does.
        """
        (n_bounds, n_default), (mult_bounds, mult_default) = n_rule, mult_rule
        n = self.ctx.number(path.child("n"), block.get("n", n_default), n_bounds)
        mult = self.ctx.number(path.child("mult"),
                               block.get("mult", mult_default), mult_bounds)
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
        op = cond["op"]
        if op not in COMPARISONS:
            # Comparisons bind tighter than `and`/`or` in Pine, so the
            # enclosing group's parentheses are the only ones a plain one needs.
            return f"{lhs} {op} {rhs}"
        if self._end_anchored(cond["rhs"]):
            # frame_eval._cross_prev broadcasts an end-anchored operand rather
            # than shifting it, and ta.crossover shifts both sides. The
            # difference is not a rounding: between two bars the nearest gap
            # can become a DIFFERENT gap at a different price, so the level one
            # bar ago is another object rather than this level's history, and
            # comparing against it registers a crossing where price crossed
            # nothing. Ask the question the trader is asking, on both bars:
            # did price cross THIS level. The grammar admits an end-anchored
            # operand bare on the right of a cross and nowhere else, which is
            # exactly the shape handled here.
            series = self._named(lhs, path.child("lhs"))
            held, crossed = (("<=", ">") if op == "crosses_above"
                             else (">=", "<"))
            return (f"({series}[1] {held} {rhs} and "
                    f"{series} {crossed} {rhs})")
        return f"{COMPARISONS[op]}({lhs}, {rhs})"

    def _end_anchored(self, node) -> bool:
        if not isinstance(node, dict):
            return False
        term = self.vocabulary.primitives.get(node.get("prim"))
        return term is not None and term.end_anchored

    def _named(self, text: str, path: RulePath) -> str:
        """Bind an expression to an identifier, so Pine can index its history.

        Pine's [] operator reads an identifier, not a parenthesized expression,
        so an operand that lowered to `(high - low)` has to be named before the
        cross above can ask it for its previous bar. An operand that is already
        an identifier (a source, or a term's own calculation) is left alone.
        """
        if text.isidentifier():
            return text
        name = self.ctx.claim("nk_" + "_".join(_sanitize(p) for p in path.parts),
                              path)
        self.ctx.statement(f"{name} = {text}")
        return name

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
        # ctx.fields already refuses a mapping that misses a declared choice,
        # so what is left for this guard is the lowering that never called it:
        # a term declaring fields whose Pine form answers one unnamed value.
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
        elif "cond" in node:
            # bars_since measures a condition rather than a series, and a
            # condition is the walk's to lower: an emit function is handed
            # operands, never spec shapes. Same slot, because it is the same
            # concept, the operand the term is applied to.
            source = self._condition(node["cond"], path.child("cond"), frame)
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
        memo = self.ctx.nodes.get(key)
        if memo is not None:
            # The node itself is already built, but this path is a new way in:
            # its block reaches every input the first production touched, which
            # is what stops a shared row heading itself after one side only.
            fields, touched = memo
            self.ctx.reached(touched, path)
            return fields
        src_tf = node.get("tf", frame)
        term = _term_name(node)
        if src_tf != frame and frame != self.chart:
            raise PineCompileError(
                "pine_nested_timeframe",
                f"{path.text}: {src_tf} is read inside a {frame} request, and "
                "request.security does not nest; move the reference up to the "
                "spec's own timeframe", path=path.text, term=term)
        with self.ctx.reaching() as touched:
            fields = (produce(frame) if src_tf == frame
                      else self._lift(node, path, src_tf, stem, produce, term))
        self.ctx.nodes[key] = (fields, frozenset(touched))
        return fields

    def _lift(self, node: dict, path: RulePath, src_tf: str, stem: str,
              produce, term: str) -> dict[str, str]:
        if src_tf not in PINE_TIMEFRAMES:
            raise PineCompileError(
                "pine_unsupported",
                f"{path.text}: {src_tf!r} has no Pine timeframe string",
                path=path.text, term=term)
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
        if self.chart in SESSION_ALIGNED and self.ctx.uses(DAY_OF_WEEK):
            # The one premise the daily weekday rests on, and the only one a
            # chart could break without saying so. The engine reads a session
            # bar's weekday off the UTC date of its label, which is that
            # session's date; the Pine converts the bar's own timestamp to New
            # York, which agrees only while a daily bar is stamped at its
            # session's start. It is for a US equity on TradingView, and the
            # engine models no other exchange, but a daily bar labelled 00:00
            # UTC would land on the prior evening and read a weekday early.
            out.append("A daily bar's timestamp is its session's start in New "
                       "York, which is TradingView's convention for a US "
                       "equity, so converting it gives the session's own "
                       "weekday.")
        if self.ctx.uses(DIV):
            out.append("A zero denominator reads as na, so a condition over it "
                       "is false.")
        return tuple(dict.fromkeys(out))
