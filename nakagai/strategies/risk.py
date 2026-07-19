"""Shared stop/target computation from a risk block.

Both RuleStrategy and CompositeStrategy carry a {"stop": ..., "target": ...}
risk object (see rules/spec.py DEFAULT_RISK); this turns it into concrete
prices against the current bars. Kept out of rules/ so composite doesn't
import a strategy class just for arithmetic.
"""

import pandas as pd

from nakagai.strategies import indicators as ind
from nakagai.strategies.base import Direction, MarketContext
from nakagai.strategies.rules.spec import DEFAULT_RISK


def stop_target(risk: dict, ctx: MarketContext, bars: pd.DataFrame,
                direction: Direction) -> tuple[float, float | None, float]:
    """-> (stop, target, rr); target None means rr-derived in rr_signal."""
    stop_spec = risk.get("stop", DEFAULT_RISK["stop"])
    target_spec = risk.get("target", DEFAULT_RISK["target"])
    ref = float(ctx.driving_bars["close"].iloc[-1])
    if stop_spec["kind"] == "atr":
        a = ind.atr(bars, stop_spec.get("n", 14)).iloc[-1]
        dist = float(stop_spec.get("mult", 2.0)) * (a if not pd.isna(a) else float("nan"))
    else:
        dist = ref * float(stop_spec.get("pct", 2.0)) / 100
    stop = ref - dist if direction == Direction.LONG else ref + dist
    if target_spec["kind"] == "rr":
        return stop, None, float(target_spec.get("rr", 2.0))
    tdist = ref * float(target_spec.get("pct", 4.0)) / 100
    target = ref + tdist if direction == Direction.LONG else ref - tdist
    return stop, target, 1.0
