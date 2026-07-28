"""Whole-frame spec evaluation: each node computed once per replay.

Replay used to recompute every indicator over the full visible prefix on every
bar, which made it O(bars x history). Every node in the grammar is causal (row i
depends only on rows <= i), so a node computed once over the whole frame and
indexed at row i gives the same number as the prefix computation did, verified
bit-for-bit across the indicator library.

Two node kinds need care and both are handled here rather than by the caller:

- CROSS-TIMEFRAME nodes. ffilling another timeframe onto the driving index is
  only lookahead-safe if the source is first restricted to bars already closed.
  The visibility map from engine/context.py supplies exactly that, per row.
- END-ANCHORED primitives (fvg_nearest, order_block) are not series at all: they
  return one float from the tail of the frame handed to them. They are evaluated
  row by row over the replay's span instead of broadcast.
"""

import json

import numpy as np
import pandas as pd

from nakagai.data.schema import DEFAULT_TIMEFRAMES, TimeframeSet
from nakagai.engine.context import visible_counts
from nakagai.strategies.rules.exprs import (_BAR_FNS, _FRAME_FNS, _SERIES_FNS,
                                            _math)
from nakagai.strategies.rules.primitives import ARG_DEFAULTS as PRIM_DEFAULTS
from nakagai.strategies.rules.primitives import (END_ANCHORED, PRIMITIVES,
                                                 end_anchored_series)
from nakagai.strategies.rules.spec import ARG_DEFAULTS, BAR_INDICATORS


class FrameEval:
    """Replay-scoped node cache over one symbol's untruncated frames."""

    def __init__(self, frames: dict, tfs: TimeframeSet = DEFAULT_TIMEFRAMES,
                 symbol: str = ""):
        self._frames = {tf: f for tf, f in frames.items()}
        self.tfs = tfs
        self.symbol = symbol
        self._cache: dict = {}
        self._maps: dict = {}
        self._spans: dict = {}

    def on(self, tf: str) -> pd.DataFrame:
        return self._frames[tf]

    def set_span(self, tf: str, lo: int, hi: int) -> None:
        self._spans[tf] = (max(int(lo), 0), int(hi))

    def _span(self, tf: str) -> tuple[int, int]:
        return self._spans.get(tf, (0, len(self._frames[tf])))

    def _positions(self, src_tf: str, dst_tf: str) -> np.ndarray:
        """Row -> index of the last `src_tf` bar closed at that row's close.

        A session-aligned destination has no honest close time derivable from
        bar labels alone: a daily bar's session closes at 16:00 NY, but the
        label carries only the date. margins.py used to approximate it as
        label + 1 day, which is correct only while the cache stays RTH-only.
        Rather than carry that assumption into the signal path, refuse it. No
        catalog spec pairs a session-aligned timeframe with a cross-timeframe
        reference, and a user spec that does deserves an error, not a guess.
        """
        key = (src_tf, dst_tf)
        if key not in self._maps:
            if dst_tf in self.tfs.session_aligned:
                raise ValueError(
                    f"spec timeframe {dst_tf!r} is session-aligned, so a "
                    f"reference to {src_tf!r} has no well-defined visibility "
                    "cutoff; move the spec to an intraday timeframe")
            dst = self._frames[dst_tf].index
            counts = visible_counts(self._frames[src_tf].index,
                                    dst + self.tfs.deltas[dst_tf], src_tf, self.tfs)
            self._maps[key] = counts - 1
        return self._maps[key]

    def to_driving(self, v, src_tf: str):
        """Lift a native series onto the driving index by visibility.

        The signal path reads at a driving-bar cursor, but conditions must be
        computed on their own timeframe first: `crosses_above` on a 1h spec has
        to compare consecutive 1h bars, which is what the per-bar path did when
        it took .iloc[-2] and .iloc[-1] of a 1h prefix. Computing natively and
        lifting the BOOLEAN preserves that exactly; comparing on the driving
        index would compare against the previous 15m bar instead.
        """
        return self._align(v, src_tf, self.tfs.driving)

    def _align(self, v, src_tf: str, dst_tf: str):
        """Carry a native series onto another timeframe's index by visibility.

        dtype is preserved: booleans lift as booleans (invisible rows False),
        floats as floats (invisible rows NaN). A float cast here would turn a
        condition series into 0.0/1.0 and silently make `not visible` truthy.
        """
        if not isinstance(v, pd.Series) or src_tf == dst_tf:
            return v
        pos = self._positions(src_tf, dst_tf)
        ok = pos >= 0
        index = self._frames[dst_tf].index
        if v.dtype == bool:
            out = np.zeros(len(pos), dtype=bool)
            out[ok] = v.to_numpy()[pos[ok]]
            return pd.Series(out, index=index)
        vals = v.to_numpy(dtype="float64")
        out = np.full(len(pos), np.nan)
        out[ok] = vals[pos[ok]]
        return pd.Series(out, index=index)

    def series(self, node, tf: str):
        if isinstance(node, (int, float)):
            return float(node)
        key = (tf, json.dumps(node, sort_keys=True))
        if key in self._cache:
            return self._cache[key]
        self._cache[key] = out = self._eval(node, tf)
        return out

    def _eval(self, node: dict, tf: str):
        src_tf = node.get("tf", tf)
        frame = self._frames[src_tf]
        if "src" in node:
            return self._align(frame[node["src"]], src_tf, tf)
        if "op" in node:
            return _math(node["op"], [self.series(a, tf) for a in node["args"]])
        if "ind" in node:
            name = node["ind"]
            a = {**ARG_DEFAULTS.get(name, {}),
                 **{k: v for k, v in node.items() if k not in ("ind", "of", "tf")}}
            if name in BAR_INDICATORS:
                out = _BAR_FNS[name](frame, a)
            else:
                of = node.get("of", {"src": "close"})
                s = self.series(of, src_tf)
                if isinstance(s, float):
                    s = pd.Series(s, index=frame.index)
                fn = _SERIES_FNS.get(name)
                out = fn(s, a["n"]) if fn else _FRAME_FNS[name](s, a)
            if isinstance(out, pd.DataFrame):
                out = out[a["field"]]
            return self._align(out, src_tf, tf)
        name = node["prim"]
        a = {**PRIM_DEFAULTS.get(name, {}),
             **{k: v for k, v in node.items() if k not in ("prim", "tf")}}
        if name in END_ANCHORED:
            lo, hi = self._span(src_tf)
            lo = max(lo - int(a.get("lookback", 40)), 0)
            part = end_anchored_series(name, None, frame, lo, hi, **a)
            out = pd.Series(np.nan, index=frame.index)
            out.iloc[lo:hi] = part.to_numpy()
            return self._align(out, src_tf, tf)
        if name == "bars_since":
            a["eval_fn"] = lambda cond, b: self.condition_series(cond, src_tf)
        return self._align(PRIMITIVES[name]["fn"](None, frame, **a), src_tf, tf)

    def condition_series(self, cond: dict, tf: str) -> pd.Series:
        """Elementwise boolean series for a comparison condition."""
        index = self._frames[tf].index
        lhs, rhs = self.series(cond["lhs"], tf), self.series(cond["rhs"], tf)
        if not isinstance(lhs, pd.Series):
            lhs = pd.Series(lhs, index=index)
        if not isinstance(rhs, pd.Series):
            rhs = pd.Series(rhs, index=index)
        op = cond["op"]
        if op == "crosses_above":
            out = (lhs.shift(1) <= rhs.shift(1)) & (lhs > rhs)
        elif op == "crosses_below":
            out = (lhs.shift(1) >= rhs.shift(1)) & (lhs < rhs)
        else:
            out = {">": lhs > rhs, "<": lhs < rhs,
                   ">=": lhs >= rhs, "<=": lhs <= rhs}[op]
        na = lhs.isna() | rhs.isna()
        if op in ("crosses_above", "crosses_below"):
            na = na | lhs.shift(1).isna() | rhs.shift(1).isna()
        return out.where(~na, False).astype(bool)

    def group_series(self, group: dict, tf: str) -> pd.Series:
        """all/any tree as one boolean series on `tf`'s index."""
        key, items = next(iter(group.items()))
        parts = [self.group_series(i, tf) if ("all" in i or "any" in i)
                 else self.condition_series(i, tf) for i in items]
        both = pd.concat(parts, axis=1)
        return both.all(axis=1) if key == "all" else both.any(axis=1)

    def driving_group(self, group: dict, tf: str) -> pd.Series:
        """`group` as a boolean series on the DRIVING index, computed once.

        This is what the signal path reads at its cursor, and it has to be
        memoized like series() is. group_series walks the tree and to_driving
        lifts the result; both are O(frame), so calling them once per bar would
        put replay straight back on O(bars x history) with the indicator cache
        doing nothing but hiding it.
        """
        key = ("group", tf, json.dumps(group, sort_keys=True))
        if key not in self._cache:
            self._cache[key] = self.to_driving(self.group_series(group, tf), tf)
        return self._cache[key]
