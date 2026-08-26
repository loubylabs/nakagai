"""Walking a validated RuleSpec into one PineProgram.

Three rules hold the whole file together.

IDENTITY IS THE PATH. Every number a spec fixes becomes an input named after
the RuleSpec path that fixed it (nk_long_all_0_lhs_sma_n), so recompiling the
same strategy keeps a chart's saved settings, and a sanitized name that would
land on a path another one already owns stops generation rather than quietly
merging two knobs.

ONE NODE, ONE CALCULATION, AND ONE REQUEST PER TIMEFRAME. Nodes are memoized on
their canonical content with the field stripped, so sma_cross's two sides read
one pair of moving averages and a MACD line and its signal come out of one
ta.macd. The memo is keyed on the HOST the node is emitted into, which is the
chart or the inside of one timeframe's function, so a node read natively and
read again on the chart are two entries: the native one is a function local and
handing its identifier to a chart expression would emit Pine that does not
compile. Everything a program needs from one timeframe lands in that
timeframe's single function, so two nodes of the same timeframe share their
sub-calculations and the script makes one request of it.

THE CHART IS THE DRIVING CADENCE, NEVER THE SPEC'S OWN TIMEFRAME. Engine.run
replays 15m bars whatever a spec's `timeframe` says, so a play's own timeframe
is REQUESTED and its conditions are lifted onto the chart, exactly as
frame_eval.driving_group lifts them. Charting the spec's timeframe instead
would decide on different bars at different prices, which is the same strategy
in name only.

A FOREIGN TIMEFRAME IS ALWAYS A REQUEST, A GATE, AND A LATCH. Every request
wraps its expression in a generated function, reads the result ONLY on the bar
the requested bar closes on, and latches it into a `var` that every other
reader reads. Three separate reasons, and dropping any one of them is a real
bug rather than a style change:

- the REQUEST is a function so that a multi-field indicator comes back as one
  tuple and so that an offset never lands on a parenthesized expression, which
  Pine does not index.
- the GATE is what makes the read land where the engine's does. Pine aligns a
  request by containment, so the confirmed `value[1]` form first answers on the
  chart bar that OPENS the next requested bar, one driving bar after the engine
  sees it (engine/context.visible_counts measures at the driving bar's CLOSE).
  Reading the unoffset value on the bar where the two clocks coincide is the
  same number, one bar earlier, and is not a lookahead there: the requested bar
  closes at that instant.
- the LATCH is what keeps the unoffset read honest everywhere else. Off the
  gate that value IS future data, so nothing may read it there. Nothing can:
  the raw identifier is read once, inside the gate, and every consumer reads
  the `var` instead, which carries the last closed value forward exactly as
  _align carries it.
"""

import json
from contextlib import contextmanager
from dataclasses import replace

from nakagai.strategies.rules.canon import canonical_expr, spec_hash
from nakagai.strategies.rules.pine.lowerings import (
    DAY_OF_WEEK, DIV, HELPERS, NEW_SESSION, SESSION_OPEN_BAR, emit_window,
)
from nakagai.strategies.rules.pine.model import (
    GENERATOR_VERSION, PineCompileError, PineExits, PineHelper, PineInput,
    PineProgram, PineRisk, RulePath, TermCall,
)
from nakagai.strategies.rules.spec import (
    BREAKEVEN_RR_BOUNDS, DEFAULT_RISK, DRIVING, SESSION_ALIGNED,
    STOP_ATR_MULT_BOUNDS,
    STOP_ATR_MULT_DEFAULT, STOP_ATR_N_BOUNDS, STOP_ATR_N_DEFAULT,
    STOP_PCT_BOUNDS, STOP_PCT_DEFAULT, TARGET_PCT_BOUNDS, TARGET_PCT_DEFAULT,
    TARGET_RR_BOUNDS, TARGET_RR_DEFAULT, TIME_STOP_BOUNDS, TIMEFRAMES,
    TRAILING_ATR_MULT_BOUNDS, TRAILING_ATR_MULT_DEFAULT, TRAILING_ATR_N_BOUNDS,
    TRAILING_ATR_N_DEFAULT, TRAILING_PCT_BOUNDS, TRAILING_PCT_DEFAULT,
    is_group_node,
)
from nakagai.strategies.rules.vocabulary import (
    Vocabulary, is_choice_rule, is_condition_rule, is_json_number,
    is_range_rule)
from nakagai.strategies.rules.windows import PRIOR_DAY

# The engine's timeframes in Pine's spelling. Every timeframe the grammar
# admits needs an entry; one without would otherwise reach request.security as
# a string TradingView reads as minutes.
PINE_TIMEFRAMES = {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}
REQUEST = ('request.security(syminfo.tickerid, "{tf}", {call}(), '
           "lookahead=barmerge.lookahead_on, gaps=barmerge.gaps_off)")
# What a latch holds before its first update. Exactly frame_eval._align's own
# dtype branch: a row no bar is visible on is False for a boolean series and
# NaN for a float one, so an ungated condition reads false and an ungated
# distance is no distance. Seeding a boolean `na` instead would be the same
# behaviour by accident rather than by construction.
SEED = {"bool": "false", "float": "na"}
# The helpers the WALK reaches rather than a term: the two rules that say when
# a session-aligned bar becomes readable and when a play on one may signal.
# Named here so "no dead Pine in the registry" can still tell a helper the
# compiler itself calls from one nothing calls at all.
GATE_HELPERS = (NEW_SESSION, SESSION_OPEN_BAR)
# Group keys carry no meaning for a reader of the settings dialog, so a label
# drops them; the index comes back only when two labels would otherwise read
# alike (see _resolve_labels).
#
# Deliberately NOT spec.GROUP_KEYS, which is one key longer. This answers a
# different question: not "what does the grammar admit as a group" but "which
# path parts does a settings-dialog label drop", and the two only happen to
# coincide. `not` is missing because it cannot reach here at all: _refuse_not
# fires in both group walks BEFORE either builds a child path, so no compiled
# program carries a "not" part for a label to drop. Widening this to the
# grammar's set would couple a presentation rule to the grammar, so the next
# group key added would silently restyle every label, and it would eat a
# term's arg named "not" from a label that should have shown it.
STRUCTURAL = ("all", "any")
COMPARISONS = {"crosses_above": "ta.crossover", "crosses_below": "ta.crossunder"}
LOW_IEX_WARNING = (
    "Window {name!r} uses US-equity extended-hours IEX data, which can be "
    "sparse.")


def _tuple(names: list[str]) -> str:
    """One name, or Pine's bracketed tuple of several."""
    return names[0] if len(names) == 1 else f"[{', '.join(names)}]"


def _refuse_not(key: str, path: RulePath) -> None:
    """N3-D8: Pine refuses `not` loudly; it does not learn to compile it.

    Pine is undeployed, so compiling negation buys nothing today, and a
    mis-compiled chart that disagrees with the engine is worse than one that
    does not render. The refusal has to be explicit because the recognizer
    below now admits `not`: without this, a negation would route into the
    all/any joiner logic, which reads `key == "all"` and otherwise assumes
    "or", so it would silently render as ANY. Falling through to the leaf
    lowerer instead, which is what a narrow recognizer did, reads `lhs` off a
    key string. Silence is the one option not available.
    """
    if key == "not":
        raise PineCompileError(
            "pine_unsupported",
            f"{path.text}: `not` has no Pine lowering; rewrite the spec "
            "without a negation to generate a chart",
            path=path.text)


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
    if not is_json_number(value):
        raise PineCompileError(
            "pine_bad_input", f"{path.text}: {value!r} is not a number, so it "
            "cannot become a Pine input", **where)
    if bounds is not None:
        if not is_range_rule(bounds) or bounds[0] > bounds[1]:
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
        # say it reached those inputs. ONE dict for the whole program, not a
        # stack under the sinks: see block().
        self.nodes: dict[tuple[str, str], _Memo] = {}
        self._blocks: dict[str, set[str]] = {}
        self._reaching: list[set[str]] = []
        self._history = 0

    # -- statements ----------------------------------------------------
    def statement(self, text: str) -> None:
        self._sinks[-1].append(text)

    @contextmanager
    def block(self):
        """Collect statements somewhere other than the top, memo untouched.

        The sink moves and the memo does NOT, and that is the whole shape: a
        timeframe's request owns one function, everything computed for that
        timeframe lands in it, so a node two blocks reach is one calculation
        there exactly as it is on a 15m chart. Scoping the memo alongside the
        sink would give sma_cross's two sides a moving average each.

        The memo entries are keyed on the host frame (see SpecLowerer._node),
        which is what keeps a function local from being handed to a
        chart-level expression even though one dict holds both.
        """
        self._sinks.append([])
        try:
            yield self._sinks[-1]
        finally:
            self._sinks.pop()

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
        # THE CHART IS THE DRIVING CADENCE, ALWAYS, and it is read off the
        # engine's own TimeframeSet rather than spelled here, so the two cannot
        # drift. `frame` is the different question: which bars this play's
        # CONDITIONS are computed on before they are lifted onto that chart.
        self.chart = DRIVING
        self.frame = str(spec.get("timeframe", "1h"))
        self.ctx = PineContext()
        self.lifted: set[str] = set()
        self.gates: dict[str, str] = {}
        self.decision_gate = ""
        # A play that reads more than one timeframe has to emit in DEPENDENCY
        # order rather than in walk order, so the pieces are collected and
        # assembled by run(): one request per timeframe, then the gate-cadence
        # snapshots over their latches, then what the chart composes out of
        # both. Insertion order decides which request block comes first, which
        # is deterministic because the walk is.
        self._requests: dict[str, dict] = {}
        self._chart_after: list[str] = []
        self._samples: list[tuple[str, str]] = []

    def run(self) -> PineProgram:
        long_decision = self._side("long")
        short_decision = self._side("short")
        exits = self._exits()
        risk = self._risk()
        # Everything a request produces goes first, because every chart-level
        # statement below reads a latch. Then the snapshots over those latches,
        # then the walk's own chart statements, then the compositions held back
        # by _decide.
        front: list[str] = []
        for tf, request in self._requests.items():
            self._request_lines(tf, request, front)
        # The freshness gate belongs to the PLAY's timeframe, not to whatever
        # its conditions happened to put in a request. It used to be resolved
        # inside the request emission, behind an early return, so a play whose
        # conditions are all foreign AND whose risk is not ATR requested
        # nothing of its own frame and lost the gate with it: every decision
        # came out ungated and fired on all four 15m bars of an hour. Whether a
        # play is gated cannot depend on the shape of its risk block.
        if self.frame != self.chart:
            self.decision_gate = self._fresh_gate(front)
            # The gate leans on the same TradingView session-anchoring premise
            # a request does (`time_close("60") == time_close` is only the
            # engine's hour if the chart's hours fall where Nakagai's do), so
            # the play's own frame is pinned whether or not anything asked for
            # it. Without this a play gated on 1h that requests nothing of its
            # own frame carried no premise sentence at all, and the artifact is
            # the only place that premise is written down.
            self.lifted.add(self.frame)
        self._sample_lines(front)
        calculations = (tuple(front) + self.ctx.calculations()
                        + tuple(self._chart_after))
        # Resolved before the assumptions, and that order is load-bearing:
        # helper_sources settles the transitive closure, and _assumptions asks
        # what the program reaches.
        helpers = self.ctx.helper_sources()
        return PineProgram(
            title=str(self.spec["name"]).strip(),
            spec_hash=spec_hash(self.spec, self.vocabulary),
            generator_version=GENERATOR_VERSION,
            chart=self.chart,
            decision_gate=self.decision_gate,
            inputs=self.ctx.inputs(),
            helpers=helpers,
            calculations=calculations,
            long_decision=long_decision,
            short_decision=short_decision,
            risk=risk,
            exits=exits,
            max_bars_back=self.ctx.max_bars_back(),
            warnings=tuple(dict.fromkeys(self.ctx.warnings)),
            assumptions=self._assumptions(),
        )

    # -- requested timeframes -------------------------------------------
    def _request_for(self, tf: str) -> dict:
        """The one request this program makes of `tf`, created on first use.

        ONE request per timeframe, not one per value and not one per node.
        Every value the program needs off `tf` is a member of a single tuple,
        so the nodes underneath them are shared the way `ONE NODE, ONE
        CALCULATION` promises, the gate is checked once, and the script spends
        one of TradingView's per-script request budget rather than one per
        operand. A request per node gave mfi_bounce two reads of the daily
        frame and ote_pullback four.
        """
        if tf not in self._requests:
            if tf not in PINE_TIMEFRAMES:
                raise PineCompileError(
                    "pine_unsupported",
                    f"timeframe: {tf!r} has no Pine timeframe string",
                    path="timeframe")
            self._requests[tf] = {"body": [], "values": []}
            self.lifted.add(tf)
        return self._requests[tf]

    def _frame_value(self, path: RulePath, name: str, kind: str,
                     produce) -> str:
        """One value the engine computes on the play's OWN timeframe.

        The identifier handed back names the LATCH, not the raw read, so every
        caller downstream is reading the last closed spec-timeframe value
        whatever bar it reads on. That is `_align`, and it is why no consumer
        of this needs to know whether a request happened at all.

        A 15m play needs none of it: the chart IS its timeframe, so this is a
        plain calculation and the artifact carries no request at all.
        """
        self.ctx.claim(name, path)
        if self.frame == self.chart:
            self.ctx.statement(f"{name} = {produce(self.frame)}")
            return name
        with self.ctx.block() as body:
            text = produce(self.frame)
        request = self._request_for(self.frame)
        request["body"] += [*body, f"{kind} {name}_native = {text}"]
        request["values"].append((path, name, kind, f"{name}_native"))
        return name

    def _requested(self, tf: str, path: RulePath, stem: str,
                   produce) -> dict[str, str]:
        """One NODE computed on `tf`, as members of the one request for `tf`.

        Only the SINK moves; the memo is one dict for the whole program and
        stays reachable in here. Every node of a timeframe lands in that
        timeframe's single function, so two of them sharing a sub-calculation
        is correct rather than a dangling local, which is the opposite of what
        a function per node required.
        """
        with self.ctx.block() as body:
            inner = produce(tf)
        request = self._request_for(tf)
        request["body"] += body
        slot = self.ctx.slot(stem, path)
        out = {}
        for field, expr in inner.items():
            name = self.ctx.claim(f"{slot}_{field}" if field else slot, path)
            request["values"].append((path, name, "float", expr))
            out[field] = name
        return out

    def _request_lines(self, tf: str, request: dict, out: list[str]) -> None:
        """One timeframe's function, request, latches and the gate over them."""
        path = RulePath(("timeframe",))
        # A session-aligned bar is read one bar back and an intraday one is
        # not, and the difference is the gate each is paired with. An intraday
        # gate fires where the requested bar CLOSES, so the unoffset value is
        # that bar's own and is confirmed. A session-aligned gate fires where
        # the new day OPENS, where the current daily bar has not happened yet,
        # so the confirmed value there is the previous one. Both are exactly
        # what closed_before hands the engine on the same bar.
        offset = "[1]" if tf in SESSION_ALIGNED else ""
        values = request["values"]
        # The gate leads, because it is the chart-level fact everything under
        # it is conditioned on and a reader meets it before the latch it guards.
        gate = self._gate(tf, out)
        returned = [f"{member}{offset}" for _p, _n, _k, member in values]
        function = self.ctx.slot("nk_frame" if tf == self.frame else "nk_htf",
                                 path)
        body = "\n".join(f"    {line}"
                          for line in [*request["body"], _tuple(returned)])
        out.append(f"{function}() =>\n{body}")
        raws = [self.ctx.claim(f"{name}_raw", p) for p, name, _k, _m in values]
        out.append(f"{_tuple(raws)} = " + REQUEST.format(
            tf=PINE_TIMEFRAMES[tf], call=function))
        out += [f"var {kind} {name} = {SEED[kind]}"
                for _p, name, kind, _m in values]
        out.append(f"if {gate}\n" + "\n".join(
            f"    {name} := {name}_raw" for _p, name, _k, _m in values))

    def _sample_lines(self, out: list[str]) -> None:
        """The play's operands as its OWN timeframe's bars saw them, and before.

        Only a cross composed on the chart needs these. frame_eval._cross_prev
        shifts an operand by one row of the SPEC's index, so "previous" is the
        previous spec-timeframe bar and never the previous chart bar. A snapshot
        pair advanced on the visibility gate is exactly that shift, and it sits
        after every latch has been updated so the pair samples the same instant.
        """
        if not self._samples:
            return
        for name, kind in self._samples:
            out.append(f"var {kind} {name}_gated = {SEED[kind]}")
            out.append(f"var {kind} {name}_prior = {SEED[kind]}")
        out.append(f"if {self._gate(self.frame, out)}\n" + "\n".join(
            f"    {name}_prior := {name}_gated\n    {name}_gated := {name}"
            for name, _kind in self._samples))

    def _gate(self, tf: str, out: list[str]) -> str:
        """The chart bar a newly closed `tf` bar first becomes readable on.

        engine/context.visible_counts, in Pine, with the engine's own two
        branches, because the two kinds of timeframe answer "closed" from
        different evidence:

        - an INTRADAY bar is visible from the driving bar whose close is its
          close, which is `searchsorted(dst_close - delta, side="right")`.
          `time_close(tf)` is the containing bar's close time and costs no
          request of its own, so the equality says exactly that.
        - a SESSION-ALIGNED bar carries a date rather than a close time, and
          closed_before makes it visible once the New York calendar date
          arrives. That is the same new-session rule every session primitive
          here already uses, rather than a second notion of a session.
        """
        if tf not in self.gates:
            path = RulePath(("timeframe",))
            expr = (f"{self.ctx.helper(NEW_SESSION, path)}()"
                    if tf in SESSION_ALIGNED
                    else f'time_close("{PINE_TIMEFRAMES[tf]}") == time_close')
            name = self.ctx.claim(
                f"nk_visible_{_sanitize(PINE_TIMEFRAMES[tf])}", path)
            out.append(f"{name} = {expr}")
            self.gates[tf] = name
        return self.gates[tf]

    def _fresh_gate(self, out: list[str]) -> str:
        """The chart bar the engine lets this play SIGNAL on: RuleStrategy._fresh.

        Visibility and freshness are different questions and the engine asks
        both. Visibility says which spec-timeframe bar a driving bar may read,
        and manage() reads on every bar. Freshness says whether on_bar may act
        at all, and a 1h play's condition stays true for all four 15m bars of
        the hour while the engine signals on exactly one of them.

        For an intraday play the two coincide, so the same identifier answers
        both rather than a second one saying the same thing. For a
        session-aligned play they do not: visibility starts when the New York
        date arrives, and first_bar_of_session gates on the 09:30 open.
        """
        if self.frame in SESSION_ALIGNED:
            path = RulePath(("timeframe",))
            name = self.ctx.claim("nk_frame_fresh", path)
            out.append(f"{name} = {self.ctx.helper(SESSION_OPEN_BAR, path)}()")
            return name
        return self._gate(self.frame, out)

    # -- blocks --------------------------------------------------------
    def _decide(self, group: dict, path: RulePath, name: str) -> str:
        """One condition tree, read at the cadence the engine reads it.

        The whole tree is computed on the play's own timeframe and lifted as
        ONE boolean whenever it can be, which is what driving_group does. It
        can be unless the tree also reads another timeframe, because
        request.security does not nest.

        When it does, the tree is split at the smallest place that works: the
        LARGEST subtree reading only the play's own frame is still one lifted
        boolean, and only what sits above it is composed on the chart out of
        latched operands. Composing there is the same answer, and measurably
        so: aligning a 1d value onto 1h and lifting that onto 15m gives, at
        every bar the gate admits, what requesting the 1d straight onto the
        chart gives, because a gated chart bar closes at the same instant its
        1h bar does and both ask which daily bar had closed by then.
        """
        if self.frame == self.chart or not self._foreign(group):
            # A play ON the driving frame never needs splitting: its host IS
            # the chart, so a foreign operand is simply lifted onto it and the
            # whole tree stays one expression.
            return self._frame_value(
                path, name, "bool",
                lambda frame: self._group(group, path, frame, frame))
        # Only the composition is held back. Whatever the walk emits under it
        # (a foreign frame's request, gate and latch) goes out where it is
        # reached, so it stands above both the play's own request and the
        # snapshots, which is the order Pine needs to read them in.
        text = self._tree(group, path)
        self._chart_after.append(f"{self.ctx.claim(name, path)} = {text}")
        return name

    def _tree(self, node, path: RulePath) -> str:
        """A mixed tree, split into native subtrees and chart-level composition."""
        if not self._foreign(node):
            # Named after its path, like everything else here, so a reader of
            # the chart-level composition can see which part of the spec each
            # lifted boolean came from.
            return self._frame_value(
                path, "nk_" + "_".join(_sanitize(p) for p in path.parts), "bool",
                lambda frame: (self._group(node, path, frame, frame)
                               if is_group_node(node)
                               else self._condition(node, path, frame, frame)))
        if not is_group_node(node):
            return self._condition(node, path, self.frame, self.chart)
        key, items = next(iter(node.items()))
        _refuse_not(key, path)
        joiner = " and " if key == "all" else " or "
        return "(" + joiner.join(
            self._tree(item, path.child(key, i))
            for i, item in enumerate(items)) + ")"

    def _foreign(self, node) -> bool:
        """True when this subtree reads any timeframe but the play's own."""
        stack = [node]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                if str(item.get("tf", self.frame)) != self.frame:
                    return True
                stack += [v for k, v in item.items() if k != "tf"]
            elif isinstance(item, list):
                stack += item
        return False

    def _side(self, side: str) -> str:
        if side not in self.spec:
            return ""
        path = RulePath((side,))
        return self._decide(self.spec[side], path, f"nk_{side}_entry")

    def _exits(self) -> PineExits:
        exits = self.spec.get("exits", {})
        base, out = RulePath(("exits",)), {}
        if "exit" in exits:
            path = base.child("exit")
            # Latched but NOT freshness-gated, because manage() is not: it asks
            # the exit group on every driving bar a position is open on, and
            # only on_bar is gated.
            out["signal"] = self._decide(exits["exit"], path, "nk_exit_signal")
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
        """An ATR-sized distance, measured on the play's own timeframe.

        Two separate reasons it is a named value rather than something the
        renderer inlines, and the second one is what the 15m chart made true:

        - `ta.atr` has to run on every bar. Called from inside a conditional
          block it would smooth over only the bars that condition was true on.
        - the ATR is the SPEC's, not the chart's. risk.stop_target reads `ref`
          off ctx.driving_bars and the ATR off ctx.bars[spec timeframe], so the
          engine's stop is a hybrid: a 15m reference price, a spec-timeframe
          distance. Leaving this on the chart would quietly make a daily play's
          stop a 15m ATR, which is roughly five times tighter.

        n_rule and mult_rule are each the (bounds, default) pair spec.py names
        for that argument, so the chart starts where the engine does.
        """
        (n_bounds, n_default), (mult_bounds, mult_default) = n_rule, mult_rule
        n = self.ctx.number(path.child("n"), block.get("n", n_default), n_bounds)
        mult = self.ctx.number(path.child("mult"),
                               block.get("mult", mult_default), mult_bounds)
        return self._frame_value(path, name, "float",
                                 lambda _frame: f"ta.atr({n}) * {mult}")

    # -- conditions ----------------------------------------------------
    #
    # Two timeframes travel together through the walk and they are not the same
    # question, which the engine also keeps apart. FRAME is what a node without
    # a `tf` inherits, exactly as FrameEval._eval passes `tf` down and an `of`
    # chain inherits its parent's. HOST is the context the Pine is being
    # emitted into, which is either the chart or the inside of one request. A
    # node is lifted when its own timeframe is not the host's, and the two
    # differ only where a condition has to be composed on the chart out of
    # operands that still belong to the play's own frame.
    def _group(self, group: dict, path: RulePath, frame: str, host: str) -> str:
        key, items = next(iter(group.items()))
        _refuse_not(key, path)
        joiner = " and " if key == "all" else " or "
        parts = []
        for i, item in enumerate(items):
            child = path.child(key, i)
            parts.append(self._group(item, child, frame, host)
                         if is_group_node(item)
                         else self._condition(item, child, frame, host))
        return "(" + joiner.join(parts) + ")"

    def _condition(self, cond: dict, path: RulePath, frame: str,
                   host: str) -> str:
        lhs = self._expr(cond["lhs"], path.child("lhs"), frame, host)
        rhs = self._expr(cond["rhs"], path.child("rhs"), frame, host)
        op = cond["op"]
        if op not in COMPARISONS:
            # Comparisons bind tighter than `and`/`or` in Pine, so the
            # enclosing group's parentheses are the only ones a plain one needs.
            return f"{lhs} {op} {rhs}"
        if host != frame:
            # A cross composed on the chart, because one of its operands comes
            # off another timeframe. "Previous" still has to mean the previous
            # bar of the PLAY's timeframe, which is the row _cross_prev shifts
            # by, so both sides are read off gate-cadence snapshots rather than
            # off the chart's own history. Comparing `[1]` here would ask what
            # the previous 15m bar said, which is the error this whole file
            # exists to avoid.
            held, crossed = (("<=", ">") if op == "crosses_above"
                             else (">=", "<"))
            lhs_now, lhs_prev = self._sampled(lhs, cond["lhs"])
            rhs_now, rhs_prev = self._sampled(rhs, cond["rhs"])
            return (f"({lhs_prev} {held} {rhs_prev} and "
                    f"{lhs_now} {crossed} {rhs_now})")
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

    def _sampled(self, ident: str, node) -> tuple[str, str]:
        """One cross operand as (this spec-timeframe bar, the one before).

        Two operands answer their own previous value rather than a snapshot:

        - a NUMBER, because a constant series shifted by one row is the same
          constant, and a snapshot pair would be two more globals saying so.
        - an END-ANCHORED primitive, because _cross_prev broadcasts those
          instead of shifting them. The nearest gap one bar ago is a different
          object, not this level's history, so both sides of the cross ask
          about THIS level. That reading is deliberate in the engine and the
          same reading is deliberate here.
        """
        if isinstance(node, (int, float)) or self._end_anchored(node):
            return ident, ident
        if (ident, "float") not in self._samples:
            self._samples.append((ident, "float"))
        return f"{ident}_gated", f"{ident}_prior"

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
    def _expr(self, node, path: RulePath, frame: str, host: str) -> str:
        if isinstance(node, (int, float)):
            return self.ctx.number(path, node)
        if "src" in node:
            return self._node(node, path, frame, host, f"nk_{node['src']}",
                              lambda _frame: {"": node["src"]})[""]
        if "op" in node:
            return self._math(node, path, frame, host)
        kind = "ind" if "ind" in node else "prim"
        name = node[kind]
        term = self.vocabulary.resolve(
            "primitive" if kind == "prim" else "indicator", name)
        if term.pine is None and not (
                node.get("window") is not None
                and term.window_reduce is not None):
            raise PineCompileError(
                "pine_unsupported",
                f"{path.text}: {name} has no Pine lowering, so this spec "
                "cannot be generated", path=path.text, term=name)
        fields = self._node(node, path, frame, host, f"nk_{name}",
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

    def _math(self, node: dict, path: RulePath, frame: str, host: str) -> str:
        op = node["op"]
        args = [self._expr(a, path.child("args", i), frame, host)
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
        # Keyed on the arg's declared TYPE, the way canon.py and spec.py key
        # theirs, rather than on the literal name "cond": a condition-taking
        # term whose arg is called anything else reached its emit with an
        # empty source and the raw condition dict still in args, and the emit
        # wrote an empty call rather than refusing.
        condition_args = {a for a, rule in term.args.items()
                          if is_condition_rule(rule)}
        window_name = node.get("window")
        window = (self.vocabulary.windows[str(window_name)]
                  if window_name is not None else
                  PRIOR_DAY if term.name == "gap_pct" else None)
        args = {**term.defaults,
                **{k: v for k, v in node.items()
                   if k not in ("ind", "prim", "of", "tf", "window")
                   and k not in condition_args}}
        if window_name is not None:
            args.pop("n", None)
        source = ""
        if term.kind in ("series", "frame"):
            # A term's own operands inherit ITS timeframe and are emitted
            # where it is, so frame and host are one and the same here.
            source = self._expr(node.get("of", {"src": "close"}),
                                path.child("of"), frame, frame)
        elif condition_args & set(node):
            # bars_since measures a condition rather than a series, and a
            # condition is the walk's to lower: an emit function is handed
            # operands, never spec shapes. Same slot, because it is the same
            # concept, the operand the term is applied to.
            #
            # Exactly one to find, and that is a fact about Term rather than an
            # assumption here: N3-D13 refuses a default on a condition-typed
            # arg, so the spec supplies every one a term declares, and Term
            # refuses a term declaring more than one, because this slot holds
            # one. Reading it off the set used to rest on the first half alone,
            # which does not imply the second: two declared args both arrived
            # and one was dropped without a word.
            (arg,) = condition_args & set(node)
            source = self._condition(node[arg], path.child(arg), frame, frame)
        call = TermCall(term=term, args=args, path=path,
                        slot=self.ctx.slot(f"nk_{term.name}", path),
                        source=source, content=_content(node, self.vocabulary),
                        window=window)
        if window is not None and window.confidence == "low_iex":
            self.ctx.warn(LOW_IEX_WARNING.format(name=window.name))
        for helper_id in (() if term.pine is None else term.pine.helpers):
            self.ctx.helper(helper_id, path, term.name)
        expr = (emit_window(self.ctx, call) if window_name is not None else
                term.pine.emit(self.ctx, call))
        return self.ctx.take_fields(call.slot) or {"": expr.text}

    # -- timeframes ----------------------------------------------------
    def _node(self, node: dict, path: RulePath, frame: str, host: str,
              stem: str, produce) -> dict[str, str]:
        """One source or term node, on the host's frame or lifted onto it.

        The memo is keyed on the HOST as well as the content, so the same node
        read natively inside a request and read again on the chart are two
        entries. They have to be: the native one is a function local, and
        handing that identifier to a chart-level expression would emit Pine
        that does not compile.
        """
        key = (host, _content(node, self.vocabulary))
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
        if src_tf != host and host != self.chart:
            # Genuinely inexpressible: this node names a timeframe other than
            # the one whose request it sits inside, and request.security does
            # not nest. A condition's own operands never land here, because
            # _decide composes those on the chart instead; what does is a
            # reference buried under a term, where there is no chart-level
            # composition to hoist it into.
            #
            # `pine_unsupported`, not a code of its own, and the reason is the
            # caller's rather than this walk's: the validator accepts this
            # shape and the engine evaluates it happily, so it reaches here
            # from an ordinary saved spec. That is a 422 (Pine cannot say
            # this), never a 500 (the compiler broke), and a distinct code
            # would fall through to the latter on the caller's side. The
            # message still names the nesting exactly.
            raise PineCompileError(
                "pine_unsupported",
                f"{path.text}: {src_tf} is read inside a {host} request, and "
                "request.security does not nest; move the reference up to the "
                "spec's own timeframe", path=path.text, term=term)
        with self.ctx.reaching() as touched:
            fields = (produce(host) if src_tf == host
                      else self._lift(node, path, src_tf, stem, produce, term))
        self.ctx.nodes[key] = (fields, frozenset(touched))
        return fields

    def _lift(self, node: dict, path: RulePath, src_tf: str, stem: str,
              produce, term: str) -> dict[str, str]:
        """One node whose timeframe is not the host's, joined to that request.

        There is nothing special left about a FOREIGN timeframe. Whether the
        node names the play's own frame (reached from a condition composed on
        the chart) or another one, it becomes a member of the single request
        for the timeframe it names. The refusal here is the one _request_for
        cannot phrase as well, because a node can name its path and its term.
        """
        if src_tf not in PINE_TIMEFRAMES:
            raise PineCompileError(
                "pine_unsupported",
                f"{path.text}: {src_tf!r} has no Pine timeframe string",
                path=path.text, term=term)
        return self._requested(src_tf, path, stem, produce)

    def _assumptions(self) -> tuple[str, ...]:
        out = [f"The chart must be on {self.chart} bars. Nakagai replays every "
               f"play on its {self.chart} driving cadence whatever the play's "
               "own timeframe says, so the script charts that cadence and "
               "requests the play's own timeframe rather than charting it."]
        if self.frame != self.chart:
            out.append(
                f"This play is on {self.frame}. Its conditions are computed on "
                f"{self.frame} bars and read on the {self.chart} bar the engine "
                f"reads them on, and it signals once per {self.frame} bar "
                "rather than on every chart bar underneath one.")
        out += ["Every condition is read on the close of its bar.",
                "Pine seeds its recursive averages (ema, rsi, atr and the terms "
                "built on them) differently from the engine, so the first bars "
                "of a chart can differ."]
        out += [(f"{tf} values are the last completed {tf} bar's, latched on "
                 "the first chart bar of a new New York day, so they neither "
                 "repaint nor lead the engine."
                 if tf in SESSION_ALIGNED else
                 f"The chart bar that closes with the {tf} bar is where the "
                 f"engine first reads {tf}, whether latching a requested "
                 f"value there or only checking time_close(\"{PINE_TIMEFRAMES[tf]}\") "
                 "for a gate, so it neither repaints nor leads the engine.")
                for tf in TIMEFRAMES if tf in self.lifted]
        intraday = sorted(self.lifted - SESSION_ALIGNED,
                          key=lambda tf: TIMEFRAMES.index(tf))
        if intraday:
            # THE ONE PREMISE OF THIS EXPORT THAT HAS NOT BEEN MEASURED ON A
            # CHART, and every non-15m play rests on it, so the artifact says
            # so in its own voice rather than leaving it in a design document.
            #
            # A requested value is TradingView's aggregate, never Nakagai's own
            # bars, and TradingView anchors an intraday aggregate to the
            # chart's SESSION. On a regular-hours chart that session opens at
            # 09:30 New York and the hourly bars run 09:30 to 10:30, which are
            # not the bars the engine has: Alpaca buckets on the wall clock and
            # data/resample.py anchors 4h at 04:00 / 08:00 / 12:00 / 16:00 /
            # 20:00 Eastern. With extended hours the session opens at 04:00, so
            # both fall back onto whole wall-clock boundaries and the two
            # agree. That is why the extended-hours guard is a correctness
            # requirement and not merely a coverage one.
            names = " and ".join(PINE_TIMEFRAMES[tf] for tf in intraday)
            out.append(
                f"This play reads {' and '.join(intraday)} off TradingView's "
                "idea of where each bar closes, whether by requesting the "
                "bar itself or only gating on its close, and either way "
                "TradingView anchors that close to the chart's session. With "
                "extended trading hours enabled that session opens at 04:00 "
                f"New York, so its {names} minute boundaries fall on the "
                "same wall-clock lines Nakagai aggregates on and the two "
                "agree. That is the one premise of this export nobody has "
                f"measured on a chart: plot time(\"{PINE_TIMEFRAMES[intraday[0]]}\") "
                "and read where the boundaries land before trusting a result.")
        if (self.ctx.uses(DAY_OF_WEEK)
                and ({self.frame} | self.lifted) & SESSION_ALIGNED):
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
