"""CompositeStrategy: interprets a CompositeSpec against MarketContext.

Registered as "composite" in the strategy registry, so saved composites run through the
existing backtest/scan machinery unchanged: params = {"spec": {...}}. With no
spec it is inert; the scanner can instantiate it harmlessly. The engine only
accepts self-contained specs; config refs must be resolved by the API first.
"""

from collections.abc import Callable, Mapping
from typing import ClassVar

import pandas as pd

from nakagai.engine.portfolio_types import Signal
from nakagai.strategies.base import Direction, MarketContext, Strategy, call_on_bar
from nakagai.strategies.composite import spec as cspec
from nakagai.strategies.risk import stop_target
from nakagai.strategies.rules.spec import DEFAULT_RISK
from nakagai.strategies.rules.vocabulary import core_vocabulary
from nakagai.strategies.util import rr_signal

# What a member is, to everything that builds one: params in, a fresh strategy
# out. A registry definition's factory and a plain strategy class both satisfy
# it, so membership has one shape whoever supplied it.
MemberFactory = Callable[[dict], Strategy]


def member_blocks(spec: Mapping) -> tuple[tuple[str, object, Mapping], ...]:
    """Every buildable block of a composite spec, in DECLARED order.

    Declared order, never sorted, because it is the order members are built
    and evaluated in: a member that raises aborts the replay, so which one is
    reached first is observable. It cannot become part of a play's identity,
    since the canonical codec sorts object keys and two specs differing only
    in block order therefore carry one digest. Reading them in the order the
    spec lists them is what keeps that promise true.

    A block still holding a saved-config ref is absent rather than an error
    here. The platform inlines those before a spec reaches core, and the
    construction door refuses the ones that arrive unresolved.
    """
    blocks = spec.get("blocks") if isinstance(spec, Mapping) else None
    if not isinstance(blocks, Mapping):
        return ()
    return tuple(
        (str(bid), block.get("strategy"), block.get("params") or {})
        for bid, block in blocks.items()
        if isinstance(block, Mapping) and "config" not in block
    )


class CompositeStrategy(Strategy):
    name = "composite"
    title = "Composite"
    description = ("Joins catalog strategies and rule specs into one bigger "
                   "strategy: members vote by signalling, boolean all/any "
                   "trees decide entries, the composite owns stop and target.")
    category = "custom"
    tags = ("custom", "composite")
    DEFAULT_PARAMS = {}

    # The strategies a block may reference on the retired singleton path,
    # bound per registry via bound(). The canonical path passes member
    # factories to __init__ instead, so nothing about a composite's membership
    # lives on a class there. Either way, a composite that was handed no
    # membership only accepts an empty (inert) spec.
    MEMBERS: ClassVar[dict[str, type[Strategy]]] = {}

    @classmethod
    def bound(cls, members: dict[str, type[Strategy]]) -> type["CompositeStrategy"]:
        return type("BoundCompositeStrategy", (cls,), {"MEMBERS": dict(members)})

    def __init__(self, params: dict | None = None, *,
                 members: Mapping[str, MemberFactory] | None = None,
                 name: str | None = None):
        super().__init__(params)
        if name is not None:
            # A registry definition names its own runtime; the assignment
            # shadows the class attribute on this instance alone.
            self.name = name
        self.spec = self.params.get("spec") or {}
        self._members: dict[str, Strategy] = {}
        # Bad specs fail loudly at construction (backtest submission), not
        # silently per bar. Empty spec = intentionally inert.
        if self.spec:
            builders = type(self).MEMBERS if members is None else members
            errs = cspec.validate_composite_spec(self.spec, builders,
                                                 allow_refs=False)
            if errs:
                raise ValueError("; ".join(errs))
            # One fresh member per block, every construction. Nothing is
            # cached and nothing is shared: a registry definition hands over
            # member FACTORIES, so this composite's members are its own.
            self._members = {bid: builders[strategy](block_params)
                             for bid, strategy, block_params
                             in member_blocks(self.spec)}
        # engine.py resolves a strategy's vocabulary as
        # getattr(self.strategy, "vocabulary", core_vocabulary()). RuleStrategy
        # carries one through bound(); a composite did not, so the Engine
        # built its context on core's terms while a scan built the same
        # play's context on the house's terms. One play answering two
        # different questions on two surfaces, and it fails as a wrong number
        # rather than as an error.
        #
        # Taken from the members rather than defaulted here: the members ARE
        # the specs being evaluated, so their vocabulary is by definition the
        # one this composite speaks. A composite whose members disagree is
        # already invalid for other reasons, so the first member's answer is
        # the composite's.
        self.vocabulary = next(
            (m.vocabulary for m in self._members.values()
             if getattr(m, "vocabulary", None) is not None),
            core_vocabulary())
        # direction -> block id -> (vote ts, member Signal); one engine run =
        # one symbol replayed in order, so plain instance state is safe.
        self._votes: dict[Direction, dict[str, tuple[pd.Timestamp, Signal]]] = {
            Direction.LONG: {}, Direction.SHORT: {}}
        self._passing = {"long": False, "short": False}

    def on_bar(self, ctx: MarketContext) -> tuple[Signal, ...]:
        if not self.spec or ctx.driving_bars.empty:
            return ()
        ref = float(ctx.driving_bars["close"].iloc[-1])
        for bid, member in self._members.items():
            # Members go through the same boundary as any other strategy, in
            # declared block order: a member that raises fails the run rather
            # than dropping a vote, and every signal it returned is counted,
            # rather than only the first.
            for sig in call_on_bar(member, ctx, deciding_close=ref, block=bid):
                self._votes[sig.direction][bid] = (ctx.now, sig)
        # A vote cast on bar i stays live through bar i + window_bars - 1;
        # window_bars=1 is strict same-bar agreement.
        window = int(self.spec.get("window_bars",
                                   cspec.DEFAULT_WINDOW_BARS)) * ctx.tfs.step
        name = str(self.spec.get("name", "composite"))
        out: list[Signal] = []
        for side, direction in (("long", Direction.LONG), ("short", Direction.SHORT)):
            tree = self.spec.get(side)
            if not tree:
                continue
            votes = self._votes[direction]
            live = {bid for bid, (ts, _) in votes.items() if ctx.now - ts < window}
            passing = cspec.eval_tree(tree, live)
            was, self._passing[side] = self._passing[side], passing
            if not passing or was:
                continue  # edge-trigger: one setup fires once, not per window bar
            voters = sorted(live & cspec.tree_block_ids(tree))
            stop, target, rr = stop_target(self.spec.get("risk", DEFAULT_RISK),
                                           ctx, ctx.driving_bars, direction)
            conf = [votes[b][1].confidence for b in voters]
            sig = rr_signal(ctx, direction, stop, rr,
                            ("composite", name, *voters),
                            f"{name}: {side} logic satisfied by {', '.join(voters)}",
                            confidence=sum(conf) / len(conf) if conf else 0.5,
                            target=target)
            if sig:
                out.append(sig)
        return tuple(out)
