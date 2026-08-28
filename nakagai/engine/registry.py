"""The frozen registry: a closed bundle, pure dependencies, fresh runtimes.

A registry is a value. It is built once from a complete set of definitions,
sorted by `definition_digest` so the order a caller supplied them in can never
reach the digest, and it answers exactly two questions afterwards: what is this
bundle, and what is the definition behind this name.

Three properties make that safe to hash and to replay.

- **Closed.** Every member a definition lowers onto is itself in the bundle,
  under its own name and its own digest, and no member tree reaches back into
  itself. Both are proven when the registry is built, before any factory runs,
  so a replay never discovers a missing member halfway through a symbol.
- **Pure.** `dependencies(params)` answers what a play reads without building
  the thing that reads it. A composite asks its members and returns the union.
  Nothing here constructs a strategy, opens a cache, or resolves a workspace.
- **Fresh.** `factory(params)` is a constructor, never a cache. Every call
  builds a new strategy, and a composite call rebuilds its whole member tree,
  so two candidates over one definition share no object and therefore no
  votes, no memo, and no ratchet.

Core knows definitions and nothing else. There is no adapter kind, manifest
bundle, workspace, saved play, or module path here: the platform resolves all
of that before it builds a bundle, and hands core the resulting values.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Protocol, TypeAlias, runtime_checkable

import pandas as pd

from nakagai.data.schema import DEFAULT_TIMEFRAMES, TimeframeSet
from nakagai.engine.bars import (
    BAR_TIMEFRAMES,
    BASE_TIMEFRAME,
    ReplayDependencies,
    _require_timeframe,
)
from nakagai.engine.canonical import _digest, definition_digest
from nakagai.engine.portfolio_types import (
    JSONValue,
    PortfolioReplayRequest,
    _fail,
    _require_digest,
    _require_instance,
    _require_items,
    _require_name,
    _require_params,
    _require_symbol,
    _set,
)
from nakagai.strategies.base import Strategy, strategy_operation
from nakagai.strategies.composite.strategy import CompositeStrategy, member_blocks
from nakagai.strategies.rules.margins import spec_margin
from nakagai.strategies.rules.strategy import (
    ReferencePair,
    SPEC_TIMEFRAME_DEFAULT,
    RuleStrategy,
    spec_reference_pairs,
    spec_timeframes,
)
from nakagai.strategies.rules.vocabulary import (
    Term,
    VocabularyFactory,
    core_vocabulary,
)
from nakagai.strategies.rules.windows import WindowSpec

# Bumped when the shape of a registry changes rather than its contents, so a
# candidate identity moves when the contract under it moves.
REGISTRY_CONTRACT_VERSION = "1"

# The graded factor: params, symbol, a read-only causal view clipped at
# `test_end`, and the observation timestamps, returning one finite or null
# margin per timestamp in the same order. The view type is `CausalFactorBars`
# in `engine/ic.py`, which imports this module to resolve definitions, so it
# is reached here through its attributes rather than by name: the dependency
# stays one-way and the parameters stay unconstrained.
IcFactor: TypeAlias = Callable[..., tuple[float | None, ...]]

# The observation axis that factor is graded on, from the same canonical
# params. It is a function rather than a value because a private play carries
# its spec in `params`, so the frame its conditions are evaluated on is not
# known until a play names it.
IcTimeframe: TypeAlias = Callable[[Mapping[str, JSONValue]], str]


@dataclass(frozen=True)
class StrategyDependencies:
    """Everything one play reads, and the grammar it is read under.

    The base timeframe is part of it rather than assumed: both builders below
    declare it whatever a spec names, because `on_bar` decides on the driving
    frame however the conditions were evaluated.

    Timeframes deduplicate into the fixed order `15m`, `1h`, `4h`, `1d`.
    Reference pairs uppercase symbols, validate both members, deduplicate, and
    sort lexically, so two declarations of one data closure are one value. A
    blank or unsupported entry is refused before a manifest could publish it.

    Normalization runs through the same `_require_timeframe`, `_require_symbol`
    and `BAR_TIMEFRAMES` the bar boundary uses. Two independent spellings of
    one rule drift; one spelling cannot.
    """

    timeframes: tuple[str, ...]
    reference_pairs: tuple[ReferencePair, ...]
    vocabulary_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.timeframes, tuple):
            raise _fail("invalid_type", "value must be a tuple", field="timeframes")
        if not isinstance(self.reference_pairs, tuple):
            raise _fail("invalid_type", "value must be a tuple", field="reference_pairs")
        declared = {_require_timeframe(value, "timeframes") for value in self.timeframes}
        if not declared:
            # A definition that declares no frame at all reads nothing, so
            # nothing would be hydrated for it and it could only ever emit
            # nothing. That is the silent-missing-timeframe failure, one level
            # up from the bar boundary, so it is a refusal here too.
            raise _fail(
                "invalid_value", "a definition declares at least one timeframe",
                field="timeframes",
            )
        pairs: set[ReferencePair] = set()
        for pair in self.reference_pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise _fail(
                    "invalid_type", "each value must be a (symbol, timeframe) tuple",
                    field="reference_pairs",
                )
            symbol, timeframe = pair
            pairs.add((
                _require_symbol(symbol, "reference_pairs"),
                _require_timeframe(timeframe, "reference_pairs"),
            ))
        _set(self, "timeframes",
             tuple(value for value in BAR_TIMEFRAMES if value in declared))
        _set(self, "reference_pairs", tuple(sorted(pairs)))
        _set(self, "vocabulary_digest",
             _require_digest(self.vocabulary_digest, "vocabulary_digest"))


@dataclass(frozen=True)
class StrategyDefinition:
    """One strategy the bundle can build, as an immutable value.

    `members` is the lowered tree: a composite carries the resolved
    definitions of the strategies it votes over, a leaf carries none. It is
    the tree its factory builds from, so the registry can prove the closure is
    complete and acyclic without calling anything.

    `ic_factor` and `ic_timeframe` are one thing in two fields and arrive
    together or not at all. A margin series says nothing without the axis it
    was graded on: the IC lens has to know which schedule record supplies the
    observation instants, and which series the realized forward return is
    taken over. A factor without an axis would leave the lens guessing, and an
    axis without a factor would describe a measurement nobody can take.

    `vocabulary_factory` is the GRAMMAR this definition is read under, and it
    is one field because a replay has to answer that question once. Three
    things ask it and they must not disagree: the factory builds a runtime
    that validates its spec under it, the IC lens grades that runtime's factor
    under it, and the replay evaluates the runtime's conditions under it, by
    handing this grammar to the context the runtime decides from. Two of those
    reading the definition's grammar while the third took the core's is a play
    deciding under one grammar and graded under another, which shows up as a
    wrong number rather than as an error: `vocabulary_digest` covers what a
    term DECLARES, so a redefined implementation moves no digest at all.
    """

    name: str
    definition_digest: str
    dependencies: Callable[[Mapping[str, JSONValue]], StrategyDependencies]
    factory: Callable[[Mapping[str, JSONValue]], Strategy]
    ic_factor: IcFactor | None
    ic_timeframe: IcTimeframe | None = None
    members: tuple["StrategyDefinition", ...] = ()
    vocabulary_factory: VocabularyFactory = core_vocabulary

    def __post_init__(self) -> None:
        _set(self, "name", _require_name(self.name, "name"))
        _set(self, "definition_digest",
             _require_digest(self.definition_digest, "definition_digest"))
        for field in ("dependencies", "factory", "vocabulary_factory"):
            if not callable(getattr(self, field)):
                raise _fail("invalid_type", "value must be callable", field=field)
        for field in ("ic_factor", "ic_timeframe"):
            value = getattr(self, field)
            if value is not None and not callable(value):
                raise _fail("invalid_type", "value must be callable", field=field)
        if (self.ic_factor is None) != (self.ic_timeframe is None):
            raise _fail(
                "invalid_value", "a graded factor and its axis arrive together",
                field="ic_timeframe", name=self.name,
            )
        _require_items(self.members, "members", StrategyDefinition)


@runtime_checkable
class StrategyRegistry(Protocol):
    registry_digest: str

    def resolve(self, name: str) -> StrategyDefinition: ...


class FrozenStrategyRegistry:
    """The one registry core replays against. Built through `from_definitions`.

    Nothing mutates after construction and nothing is lazy: the bundle, its
    order, and its digest are all decided before the first caller sees it.
    """

    __slots__ = ("_by_name", "_registry_digest")

    def __init__(self, definitions: Iterable[StrategyDefinition]) -> None:
        """Freeze a complete bundle, or refuse it.

        The whole pipeline lives here rather than behind the classmethod
        below, because Python has no private constructor: a validating
        `from_definitions` beside a bare `__init__` would leave an unchecked
        way to build a registry that cannot answer `resolve`.

        The sort is on `definition_digest` alone, which is total because two
        definitions may not share one: a repeated digest would make a play's
        `definition_digest` name two different strategies. The order is not
        readable afterwards and does not need to be: it exists so that
        `registry_digest` cannot depend on the order a caller supplied.
        """
        ordered = tuple(sorted(_require_definitions(definitions),
                               key=lambda item: item.definition_digest))
        registered = _validate_definition_digests(ordered)
        _require_acyclic_and_closed(_definition_graph(ordered, registered))
        self._by_name = MappingProxyType(registered)
        self._registry_digest = _bundle_digest(ordered)

    @classmethod
    def from_definitions(
        cls, definitions: Iterable[StrategyDefinition],
    ) -> "FrozenStrategyRegistry":
        """The named door platform and the fixtures build a bundle through."""
        return cls(definitions)

    @property
    def registry_digest(self) -> str:
        return self._registry_digest

    def resolve(self, name: str) -> StrategyDefinition:
        definition = self._by_name.get(_require_name(name, "strategy"))
        if definition is None:
            raise _fail(
                "unknown_strategy", "this bundle registers no such definition",
                field="strategy", strategy=name,
            )
        return definition


# ------------------------------------------------ the two preflight doors


def validate_registry(
    request: PortfolioReplayRequest, registry: StrategyRegistry,
) -> StrategyRegistry:
    """Resolve every play's definition and prove it is the one the play names.

    Two different digests meet here, and they name two different things.
    `StrategyDefinition.definition_digest` is the BASE digest of a strategy
    body, which `spec_base_digest` below computes for a rule spec.
    `canonical.definition_digest(base, params)` binds that base to ONE play's
    parameters, and it is what a `PlayRequest` carries.

    Checking the pair is what stops a play naming any params under any
    definition. Nothing upstream can: the registry digest deliberately covers
    names and base digests alone, since param-dependent values belong to a
    candidate's identity rather than to the bundle's, so a play that swapped
    its params would leave every other digest in the request intact.

    Resolution happens here and not later because it is the registry step of
    the preflight: a play naming a strategy this bundle never registered is
    refused before any dependency is asked for and long before a factory runs.
    """
    _require_instance(request, "request", PortfolioReplayRequest)
    if not isinstance(registry, StrategyRegistry):
        raise _fail("invalid_type", "value must be a strategy registry",
                    field="registry")
    for play in request.plays:
        definition = registry.resolve(play.strategy)
        expected = definition_digest(definition.definition_digest, play.params)
        if play.definition_digest != expected:
            raise _fail(
                "definition_digest_mismatch",
                "a play's digest does not bind this definition to these params",
                field="definition_digest", play_id=play.play_id,
                strategy=play.strategy, expected=expected,
                actual=play.definition_digest,
            )
    return registry


def dependencies_for(
    request: PortfolioReplayRequest, registry: StrategyRegistry,
) -> ReplayDependencies:
    """The union of what every play reads, as one closure for the whole replay.

    Pure: a definition answers for its params and nothing is constructed, so
    this runs inside the preflight without putting a strategy on the clock.

    The base timeframe is injected rather than assumed present. A definition
    declares what its own conditions read, and a daily-only play legitimately
    declares `1d` alone; the ACCOUNT still fills, marks, and settles on the
    base clock, so the replay reads it whatever a strategy chose. Without the
    injection the union of such a portfolio would be a tuple
    `ReplayDependencies` refuses outright, and a valid request would fail as a
    contract error rather than run.

    Ordering and deduplication belong to `ReplayDependencies`, which puts the
    timeframes into fixed order and sorts exact reference pairs, so the closure
    cannot depend on which play was asked first.
    """
    _require_instance(request, "request", PortfolioReplayRequest)
    timeframes: list[str] = [BASE_TIMEFRAME]
    reference_pairs: list[ReferencePair] = []
    for play in request.plays:
        definition = registry.resolve(play.strategy)
        with strategy_operation("dependencies", strategy=definition.name,
                                play_id=play.play_id):
            declared = definition.dependencies(play.params)
        _require_instance(declared, "dependencies", StrategyDependencies)
        timeframes.extend(declared.timeframes)
        reference_pairs.extend(declared.reference_pairs)
    return ReplayDependencies(timeframes=tuple(timeframes),
                              reference_pairs=tuple(reference_pairs))


def _require_definitions(value: object) -> tuple[StrategyDefinition, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Iterable):
        raise _fail(
            "invalid_type", "definitions must be an iterable of definitions",
            field="definitions", seen=type(value).__name__,
        )
    return _require_items(tuple(value), "definitions", StrategyDefinition)


def _validate_definition_digests(
    definitions: tuple[StrategyDefinition, ...],
) -> Mapping[str, StrategyDefinition]:
    """One definition per name and one name per digest, or no bundle at all."""
    registered: dict[str, StrategyDefinition] = {}
    by_digest: dict[str, str] = {}
    for definition in definitions:
        if definition.name in registered:
            raise _fail(
                "duplicate_value", "two definitions share one name",
                field="name", name=definition.name,
            )
        claimed = by_digest.get(definition.definition_digest)
        if claimed is not None:
            raise _fail(
                "duplicate_value", "two definitions share one digest",
                field="definition_digest", name=definition.name, other=claimed,
            )
        registered[definition.name] = definition
        by_digest[definition.definition_digest] = definition.name
    return registered


def _definition_graph(
    definitions: tuple[StrategyDefinition, ...],
    registered: Mapping[str, StrategyDefinition],
) -> Mapping[str, frozenset[str]]:
    """Every member edge in the bundle, keyed by name, closure proven complete.

    The walk descends into captured members rather than reading only the top
    level, because a member is itself a definition and may lower onto members
    of its own. Every one of them must be the definition this bundle registers
    under that name, judged by digest: a tree whose leaf disagrees with the
    bundle would replay something the registry digest does not describe.

    The `seen` guard is on object identity, so a tree knotted back into itself
    terminates the walk here instead of exhausting the stack, and the name
    cycle it produces is refused by `_require_acyclic_and_closed`.
    """
    edges: dict[str, set[str]] = {}
    seen: dict[int, StrategyDefinition] = {}
    stack = list(definitions)
    while stack:
        definition = stack.pop()
        if id(definition) in seen:
            continue
        seen[id(definition)] = definition
        edges.setdefault(definition.name, set())
        for member in definition.members:
            known = registered.get(member.name)
            if known is None:
                raise _fail(
                    "unknown_member", "a member is absent from the bundle",
                    field="members", name=definition.name, member=member.name,
                )
            if known.definition_digest != member.definition_digest:
                raise _fail(
                    "member_digest_mismatch",
                    "a member disagrees with the definition of that name",
                    field="members", name=definition.name, member=member.name,
                )
            edges[definition.name].add(member.name)
            stack.append(member)
    return {name: frozenset(members) for name, members in edges.items()}


def _require_acyclic_and_closed(graph: Mapping[str, frozenset[str]]) -> None:
    """No definition reaches itself through its members.

    Completeness was proven while the graph was built, since an unregistered
    member never becomes an edge. What is left is the cycle, and it matters
    before anything is constructed: a member tree that reaches back into
    itself has no finite runtime, so a factory would recurse until the
    interpreter stopped it, one symbol into a replay.

    Iterative, not recursive, so a long chain of definitions raises the
    contract's own refusal rather than a `RecursionError` from outside it.
    """
    finished: set[str] = set()
    for root in sorted(graph):
        if root in finished:
            continue
        on_path: set[str] = set()
        stack: list[tuple[str, bool]] = [(root, False)]
        while stack:
            name, leaving = stack.pop()
            if leaving:
                on_path.discard(name)
                finished.add(name)
                continue
            if name in finished:
                continue
            if name in on_path:
                raise _fail(
                    "member_cycle", "a definition reaches back into itself",
                    field="members", name=name,
                )
            on_path.add(name)
            stack.append((name, True))
            for member in sorted(graph.get(name, ())):
                stack.append((member, False))


def _bundle_digest(definitions: tuple[StrategyDefinition, ...]) -> str:
    """Names, digests, the core vocabulary, and the contract version.

    Param-dependent outputs stay out on purpose: they belong to a candidate's
    identity, not to the bundle's. Two workers holding the same code and the
    same definitions agree here whatever order they were handed.
    """
    return _digest({
        "registry_contract_version": REGISTRY_CONTRACT_VERSION,
        "core_vocabulary_digest": vocabulary_digest(core_vocabulary),
        "definitions": [
            {"name": item.name, "definition_digest": item.definition_digest}
            for item in definitions
        ],
    })


# --------------------------------------------------------- vocabulary digests


@cache
def vocabulary_digest(vocabulary_factory: VocabularyFactory) -> str:
    """The digest of the grammar a definition is read under.

    Keyed on the FACTORY and never on the Vocabulary. A Vocabulary holds two
    mappings, so it is unhashable and a cache keyed on one fails at call time
    rather than at import; the catalog loaders take a factory for the same
    reason.

    A term function has no canonical encoding, so this covers what a term
    DECLARES: its name, kind, argument schema, defaults, causal flags, and
    window reducer contract. A change to what a declared term computes does
    not move the digest, which is why core is version pinned rather than digest
    pinned.
    """
    vocabulary = vocabulary_factory()
    return _digest({
        "expression_scopes": ["sym", "tf", "window"],
        "indicators": [_term_projection(term)
                       for term in sorted(vocabulary.indicators.values(),
                                          key=lambda item: item.name)],
        "primitives": [_term_projection(term)
                       for term in sorted(vocabulary.primitives.values(),
                                          key=lambda item: item.name)],
        "windows": [_window_projection(row)
                    for row in sorted(vocabulary.windows.values(),
                                      key=lambda item: item.name)],
    })


def _term_projection(term: Term) -> dict:
    return {
        "name": term.name,
        "kind": term.kind,
        "args": dict(term.args),
        "defaults": dict(term.defaults),
        "end_anchored": term.end_anchored,
        "session_scoped": term.session_scoped,
        "driving_frame_intraday": term.driving_frame_intraday,
        "window_reduce": term.window_reduce,
        "window_required": term.window_required,
    }


def _window_projection(row: WindowSpec) -> dict:
    return {
        "name": row.name,
        "tz": row.tz,
        "start": row.start.strftime("%H:%M"),
        "end": row.end.strftime("%H:%M"),
        "recurrence": row.recurrence,
        "confidence": row.confidence,
    }


def spec_base_digest(
    spec: Mapping[str, JSONValue],
    vocabulary_factory: VocabularyFactory = core_vocabulary,
) -> str:
    """The base digest of a definition whose body is one immutable spec.

    The vocabulary participates because one spec read under two grammars is
    two strategies. `definition_digest(base, params)` in the canonical codec
    then binds this to one play's params.
    """
    return _digest((_require_params(spec, "spec"),
                    vocabulary_digest(vocabulary_factory)))


# ------------------------------------------------------------- the two bodies


def rules_definition(
    name: str,
    definition_digest: str,
    *,
    spec: Mapping[str, JSONValue] | None = None,
    vocabulary_factory: VocabularyFactory = core_vocabulary,
) -> StrategyDefinition:
    """A definition over one RuleSpec.

    `spec` is bound here for a catalog or published play, and left None for a
    private play whose spec travels in `params`. Supplying both is refused at
    the factory rather than merged: a play that could replace the spec its
    digest was taken over would replay something the digest does not describe.

    Every RuleSpec definition is graded, and the factor is not a parameter.
    The graded margin of a rule tree is `spec_margin`, there is exactly one of
    it, and a definition that could be built without it would report a null
    correlation over zero observations, which reads as a lens that measured
    and found nothing rather than as one that never ran.
    """
    bound = None if spec is None else _require_params(spec, "spec")
    digest = vocabulary_digest(vocabulary_factory)

    def dependencies(params: Mapping[str, JSONValue]) -> StrategyDependencies:
        effective = _rules_params(bound, params)
        # The base timeframe is always read: `on_bar` decides on the driving
        # frame whatever frame the spec's conditions are evaluated on.
        declared = (BASE_TIMEFRAME, *spec_timeframes(effective.get("spec") or {}))
        return StrategyDependencies(
            timeframes=declared,
            reference_pairs=spec_reference_pairs(effective.get("spec") or {}),
            vocabulary_digest=digest,
        )

    def factory(params: Mapping[str, JSONValue]) -> Strategy:
        return RuleStrategy(_rules_params(bound, params),
                            vocabulary=vocabulary_factory(), name=name)

    def ic_timeframe(params: Mapping[str, JSONValue]) -> str:
        return _spec_axis(_rules_params(bound, params).get("spec") or {})

    def ic_factor(params: Mapping[str, JSONValue], symbol: str, bars,
                  timestamps: tuple) -> tuple[float | None, ...]:
        return _spec_margins(
            _rules_params(bound, params).get("spec") or {},
            bars, vocabulary_factory,
        )

    return StrategyDefinition(
        name=name, definition_digest=definition_digest,
        dependencies=dependencies, factory=factory,
        ic_factor=ic_factor, ic_timeframe=ic_timeframe,
        vocabulary_factory=vocabulary_factory,
    )


def _spec_axis(spec: Mapping[str, JSONValue]) -> str:
    """The frame a spec's conditions are evaluated on, and observed at.

    An EMPTY spec is the inert strategy, which reads nothing and grades
    nothing, so it answers with the base timeframe rather than the grammar's
    default. `spec_timeframes` returns nothing for it too, so a definition
    declaring only the base frame would otherwise carry an axis outside its
    own data closure and the lens would refuse the replay over a play that
    can only ever report an empty measurement.
    """
    if not spec:
        return BASE_TIMEFRAME
    return spec.get("timeframe", SPEC_TIMEFRAME_DEFAULT)


def _spec_margins(spec: Mapping[str, JSONValue], bars,
                  vocabulary_factory: VocabularyFactory) -> tuple[float | None, ...]:
    """`spec_margin` over the IC lens's causal view, one value per label.

    `bars` is the lens's `CausalFactorBars`: frames already cut at `test_end`,
    the axis those observations lie on, and the axis row each one is labeled
    at. Nothing here reaches for a bar the view does not carry, which is what
    keeps the IC tail out of a graded margin.

    The span is not optional. Without one the end-anchored primitives default
    to the whole frame and walk every row of history to produce values no
    observation reads; with it they produce the values the selected rows
    themselves would have seen. `FrameEval` refuses a span that moves after
    anything is cached, so this builds one evaluator per call.

    `spec_margin` answers on the index it was handed, row for row, so the
    values travel out in the order they arrived and nothing realigns them. An
    inert spec is the one exception: it grades an EMPTY series rather than a
    series of nulls, which means the same thing to the lens and is spelled out
    here rather than left to an alignment step.
    """
    from nakagai.strategies.rules.frame_eval import FrameEval
    frames = dict(bars.frames)
    axis = bars.timeframe
    index = pd.DatetimeIndex(list(bars.labels))
    evaluator = FrameEval(
        frames,
        TimeframeSet(
            driving=BASE_TIMEFRAME,
            higher=tuple(tf for tf in frames if tf != BASE_TIMEFRAME),
            deltas=DEFAULT_TIMEFRAMES.deltas,
            session_aligned=DEFAULT_TIMEFRAMES.session_aligned,
        ),
        vocabulary=vocabulary_factory(),
    )
    rows = frames[axis].index
    evaluator.set_span(axis, int(rows.searchsorted(index[0], side="left")),
                       int(rows.searchsorted(index[-1], side="right")))
    margin = spec_margin(spec, evaluator, index)
    if margin.empty:
        return (None,) * len(index)
    return tuple(None if pd.isna(value) else float(value) for value in margin)


def composite_definition(
    name: str,
    definition_digest: str,
    *,
    members: Mapping[str, StrategyDefinition],
    vocabulary_factory: VocabularyFactory = core_vocabulary,
) -> StrategyDefinition:
    """A definition over a lowered composite tree.

    `members` are already resolved definitions, so a composite is a value like
    any other: its factory captures them and rebuilds the whole tree for every
    candidate, and its dependency function asks each member what it reads
    without building anything.

    The graded factor is null in Phase 1, for a composite specifically: its
    members' margins are not one series, and inventing one would report a
    correlation nothing computed.
    """
    resolved = _require_members(members)
    digest = vocabulary_digest(vocabulary_factory)
    factories = MappingProxyType(
        {key: item.factory for key, item in resolved.items()})

    def dependencies(params: Mapping[str, JSONValue]) -> StrategyDependencies:
        spec = _plain(_require_params(params, "params")).get("spec") or {}
        timeframes: list[object] = [BASE_TIMEFRAME]
        reference_pairs: list[ReferencePair] = []
        for block, strategy, block_params in member_blocks(spec):
            # The block id identifies the offender, never the value it holds:
            # an unbound `strategy` is whatever the params carried, and a
            # value with no canonical encoding would raise out of the closed
            # taxonomy on its way into the refusal details.
            member = resolved.get(strategy) if isinstance(strategy, str) else None
            if member is None:
                raise _fail(
                    "unknown_member", "a block names a strategy this tree never bound",
                    field="blocks", name=name, block=block,
                )
            declared = member.dependencies(block_params)
            if declared.vocabulary_digest != digest:
                raise _fail(
                    "invalid_value", "a member is read under another vocabulary",
                    field="vocabulary_digest", name=name, member=member.name,
                )
            timeframes.extend(declared.timeframes)
            reference_pairs.extend(declared.reference_pairs)
        return StrategyDependencies(
            timeframes=tuple(timeframes), reference_pairs=tuple(reference_pairs),
            vocabulary_digest=digest,
        )

    def factory(params: Mapping[str, JSONValue]) -> Strategy:
        return CompositeStrategy(_plain(_require_params(params, "params")),
                                 members=factories, name=name)

    return StrategyDefinition(
        name=name, definition_digest=definition_digest,
        dependencies=dependencies, factory=factory,
        ic_factor=None, ic_timeframe=None,
        members=tuple(resolved[key] for key in sorted(resolved)),
        vocabulary_factory=vocabulary_factory,
    )


def _require_members(
    members: object,
) -> Mapping[str, StrategyDefinition]:
    if not isinstance(members, Mapping):
        raise _fail("invalid_type", "members must be a mapping", field="members")
    resolved: dict[str, StrategyDefinition] = {}
    for key, item in members.items():
        member = _require_name(key, "members")
        _require_instance(item, f"members.{member}", StrategyDefinition)
        if item.name != member:
            raise _fail(
                "invalid_value", "a member is bound under another name",
                field="members", key=member, name=item.name,
            )
        resolved[member] = item
    return resolved


def _rules_params(
    bound: Mapping[str, JSONValue] | None, params: Mapping[str, JSONValue],
) -> dict:
    given = _plain(_require_params(params, "params"))
    if bound is None:
        return given
    if "spec" in given:
        raise _fail(
            "invalid_value", "this definition already binds its rule spec",
            field="params.spec",
        )
    return {"spec": _plain(bound), **given}


def _plain(value: object) -> object:
    """A fresh, plain-JSON copy of a canonical value.

    Params reach core frozen: `_require_params` turns every object into a
    `MappingProxyType` and every array into a tuple. The rule and composite
    grammars test `isinstance(node, dict)` and `isinstance(items, list)`
    exactly, so a frozen spec would fail validation on its own shape.

    Thawing per call is also what keeps two runtimes apart. Each call rebuilds
    the whole structure, so no two strategies hold one spec object and a
    strategy that writes into its own params cannot reach the definition, the
    caller's mapping, or its sibling.
    """
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value
