"""A rule's past fires on one symbol, and what followed them against a benchmark.

Each side's entry group is evaluated natively over the spec's own frame, once
per closed bar, which is the bar the live scan acts on: it reads that bar at
its first 15m close after the bar is complete, gated by `first_bar_of_session`
or `first_bar_of_week` on a session-aligned frame. Long takes precedence, as in
`RuleStrategy.on_bar`, so a short fire is a bar where short holds and long does
not.

A move is the symbol's return over `horizon` bars minus the benchmark's over
the same bars. The fires' moves and the baseline (every evaluated bar with a
horizon) come out side by side, so a caller can set what the rule picked
against what any bar would have given. No statistics are computed here.

A rule that reads something with no history in the bars (a fact, or a term the
caller names in `refused_terms`) is refused rather than evaluated: its past
fires would be today's value read back over every bar.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nakagai.data.schema import DEFAULT_TIMEFRAMES
from nakagai.strategies.rules.vocabulary import Vocabulary, resolve_vocabulary


class PastFiresRefused(ValueError):
    """A rule whose past fires cannot be read honestly. `code` is
    "unhistoried", "composite" or "insufficient_history"."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SideFires:
    fire_bars: tuple[int, ...]         # row positions in the spec-tf frame
    fire_moves: tuple[float, ...]      # scored fires only, side-signed
    baseline_moves: tuple[float, ...]  # every evaluated bar with a horizon, side-signed


@dataclass(frozen=True)
class SymbolFires:
    symbol: str
    timeframe: str
    horizon: int
    bars: int     # evaluated bars (n - warmup)
    first: str    # ISO label of the first evaluated bar
    last: str     # ISO label of the last evaluated bar
    long: SideFires | None
    short: SideFires | None


def unhistoried_reads(spec: Mapping, refused_terms: Collection[str]) -> tuple[str, ...]:
    """Sorted reads in `spec` that have no history: every fact node, as
    "fact:<name>", and every indicator or primitive named in `refused_terms`."""
    found: set[str] = set()
    stack: list[object] = [spec]
    while stack:
        node = stack.pop()
        if isinstance(node, Mapping):
            fact = node.get("fact")
            if isinstance(fact, str):
                found.add(f"fact:{fact}")
            for key in ("ind", "prim"):
                name = node.get(key)
                if isinstance(name, str) and name in refused_terms:
                    found.add(name)
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
    return tuple(sorted(found))


def past_fires(spec: Mapping, frames: Mapping[tuple[str, str], pd.DataFrame],
               symbol: str, *, benchmark: str, horizon: int, warmup: int,
               vocabulary: Vocabulary | None = None,
               refused_terms: Collection[str] = ()) -> SymbolFires:
    """`spec`'s fires on `symbol` over `frames[(symbol, spec["timeframe"])]`,
    with each fire's and each baseline bar's `horizon`-bar move against
    `benchmark`. Rows before `warmup` are neither fires nor baseline.

    `frames` is pair-keyed as `(symbol, timeframe)` and must hold the
    benchmark on the spec's timeframe. A missing frame raises KeyError, and a
    rule the frames cannot evaluate raises what `FrameEval` raises."""
    # At call time: frame_eval imports engine.context, which imports this
    # package, as engine/context.py's own FrameEval import notes.
    from nakagai.strategies.rules.frame_eval import FrameEval
    if horizon < 1 or warmup < 0:
        raise ValueError(f"horizon {horizon} must be >= 1 and warmup {warmup} >= 0")
    if "blocks" in spec:
        raise PastFiresRefused(
            "composite", "a composite has no single entry rule to read back")
    unhistoried = unhistoried_reads(spec, refused_terms)
    if unhistoried:
        raise PastFiresRefused(
            "unhistoried",
            f"the rule reads {', '.join(unhistoried)}, which has no history")
    tf = spec["timeframe"]
    own = frames[(symbol, tf)]
    n = len(own)
    if n < warmup + horizon + 2:
        raise PastFiresRefused(
            "insufficient_history",
            f"{n} {tf} bars; warmup {warmup} and horizon {horizon} need "
            f"{warmup + horizon + 2}")

    # Reference symbols are read on the traded symbol's own index, as
    # build_context does.
    aligned = {
        (sym, pair_tf): (frame if sym == symbol
                         else frame.reindex(frames[(symbol, pair_tf)].index))
        for (sym, pair_tf), frame in frames.items()
    }
    fe = FrameEval(symbol, aligned, DEFAULT_TIMEFRAMES,
                   vocabulary=resolve_vocabulary(vocabulary))

    c = own["close"].to_numpy(dtype="float64")
    b = frames[(benchmark, tf)]["close"].reindex(own.index).to_numpy(dtype="float64")
    move = np.full(n, np.nan)
    move[:n - horizon] = ((c[horizon:] / c[:-horizon] - 1)
                          - (b[horizon:] / b[:-horizon] - 1))
    evaluated = np.arange(n) >= warmup
    scored = evaluated & ~np.isnan(move)

    held = {side: fe.group_series(spec[side], tf).to_numpy(dtype=bool)
            for side in ("long", "short") if spec.get(side)}
    if "short" in held and "long" in held:
        held["short"] = held["short"] & ~held["long"]

    def side_fires(side: str) -> SideFires | None:
        if side not in held:
            return None
        sign = 1.0 if side == "long" else -1.0
        fires = held[side] & evaluated
        return SideFires(
            fire_bars=tuple(int(i) for i in np.flatnonzero(fires)),
            fire_moves=tuple(float(sign * m) for m in move[fires & scored]),
            baseline_moves=tuple(float(sign * m) for m in move[scored]),
        )

    return SymbolFires(
        symbol=symbol, timeframe=tf, horizon=horizon, bars=n - warmup,
        first=own.index[warmup].isoformat(), last=own.index[-1].isoformat(),
        long=side_fires("long"), short=side_fires("short"))
