"""CompositeStrategy: interprets a CompositeSpec against MarketContext.

Registered as "composite" in the strategy registry, so saved composites run through the
existing backtest/scan machinery unchanged: params = {"spec": {...}}. With no
spec it is inert; the scanner can instantiate it harmlessly. The engine only
accepts self-contained specs; config refs must be resolved by the API first.
"""

from typing import ClassVar

import pandas as pd

from nakagai.strategies.base import Direction, MarketContext, Signal, Strategy
from nakagai.strategies.composite import spec as cspec
from nakagai.strategies.risk import stop_target
from nakagai.strategies.rules.spec import DEFAULT_RISK
from nakagai.strategies.util import rr_signal


class CompositeStrategy(Strategy):
    name = "composite"
    title = "Composite"
    description = ("Joins catalog strategies and rule specs into one bigger "
                   "strategy: members vote by signalling, boolean all/any "
                   "trees decide entries, the composite owns stop and target.")
    category = "custom"
    tags = ("custom", "composite")
    DEFAULT_PARAMS = {}

    # The strategies a block may reference. Bound per registry via bound();
    # the base class knows no catalog, so an unbound composite only accepts
    # an empty (inert) spec.
    MEMBERS: ClassVar[dict[str, type[Strategy]]] = {}

    @classmethod
    def bound(cls, members: dict[str, type[Strategy]]) -> type["CompositeStrategy"]:
        return type("BoundCompositeStrategy", (cls,), {"MEMBERS": dict(members)})

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.spec = self.params.get("spec") or {}
        self._members: dict[str, Strategy] = {}
        # Bad specs fail loudly at construction (backtest submission), not
        # silently per bar. Empty spec = intentionally inert.
        if self.spec:
            members = type(self).MEMBERS
            errs = cspec.validate_composite_spec(self.spec, members,
                                                 allow_refs=False)
            if errs:
                raise ValueError("; ".join(errs))
            self._members = {bid: members[b["strategy"]](b.get("params", {}))
                             for bid, b in self.spec["blocks"].items()}
        # direction -> block id -> (vote ts, member Signal); one engine run =
        # one symbol replayed in order, so plain instance state is safe.
        self._votes: dict[Direction, dict[str, tuple[pd.Timestamp, Signal]]] = {
            Direction.LONG: {}, Direction.SHORT: {}}
        self._passing = {"long": False, "short": False}

    def on_bar(self, ctx: MarketContext) -> list[Signal]:
        if not self.spec or ctx.driving_bars.empty:
            return []
        for bid, member in self._members.items():
            try:
                signals = member.on_bar(ctx)
            except Exception as e:
                # A silently dropped vote would corrupt results, so fail the run.
                raise RuntimeError(f"composite block {bid!r} ({member.name}) "
                                   f"failed on {ctx.symbol} @ {ctx.now}: {e}") from e
            for sig in signals:
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
        return out
