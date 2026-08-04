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
from nakagai.strategies.rules.primitives import end_anchored_series
from nakagai.strategies.rules.vocabulary import Vocabulary, resolve_vocabulary


def _as_series(v, like):
    if isinstance(v, pd.Series):
        return v
    idx = like.index if isinstance(like, pd.Series) else None
    return pd.Series(v, index=idx)


def _math(op: str, args: list):
    """The RuleSpec math ops over scalars and series, mixed freely.

    Division maps a zero denominator to NaN rather than raising or producing an
    infinity: a condition over NaN reads False, which is the honest answer for
    a ratio that does not exist on that bar.
    """
    if op == "abs":
        return args[0].abs() if isinstance(args[0], pd.Series) else abs(args[0])
    out = args[0]
    for a in args[1:]:
        if op == "+":
            out = out + a
        elif op == "-":
            out = out - a
        elif op == "*":
            out = out * a
        elif op == "/":
            denom = a.replace(0.0, float("nan")) if isinstance(a, pd.Series) else \
                (float("nan") if a == 0 else a)
            out = out / denom
        elif op in ("min", "max"):
            both = pd.concat([_as_series(out, a), _as_series(a, out)], axis=1)
            out = both.min(axis=1) if op == "min" else both.max(axis=1)
    return out


def _cross_prev(node, v: pd.Series, vocabulary: Vocabulary) -> pd.Series:
    """The operand's PREVIOUS-bar value inside a cross comparison.

    indicators.crossed_above reads `b.iloc[-2], b.iloc[-1]` when `b` is a
    Series but `(b, b)` when it is a scalar. fvg_nearest and order_block used
    to return one float per replayed bar, so that scalar branch compared BOTH
    bars of the cross against the single level known at the current bar:
    close[i-1] <= L[i] and close[i] > L[i]. They are series now, and shifting
    them would test the earlier bar against L[i-1] instead, which is a real
    semantic change that moves trades on every play crossing one of them.
    Reproduce the broadcast: an end-anchored operand has no honest history to
    shift, so its "previous" value is the current row's.

    That was first kept because it was the old behavior. It is kept now because
    it is the right reading, and the difference is worth stating so nobody
    "fixes" it later. fvg_nearest and order_block answer "the nearest such
    level as of now". Between bar i-1 and bar i the nearest gap can become a
    DIFFERENT gap at a different price, so L[i-1] is not the same level one bar
    ago; it is another object. Comparing close[i-1] to L[i-1] and close[i] to
    L[i] therefore registers a crossing when the level was re-identified and
    price crossed nothing, which is not the question a trader is asking. The
    broadcast asks the question they are: did price cross THIS level on this
    bar. Measured over the catalog corpus (57 plays x 10 symbols x 32 windows),
    switching to the shifted reading moves 39 of 570 pairs and 42,227 trades
    to 41,647, all of it inside fvg_bounce, ifvg_reversal, ob_bounce and the
    smc_confluence composite that contains them.

    Nesting an end-anchored primitive inside a math op would evade the check
    below and get the shifted reading anyway, so spec._check_condition refuses
    that shape outright rather than leaving two readings reachable.

    What the .shift(1) on everything else preserves, exactly, and what it does
    not:

    - A NATIVE-timeframe operand is preserved. Row i-1 of the whole-frame
      series is what the prefix ending at i-1 computed, which is what
      `.iloc[-2]` of the cut frame read.
    - A CROSS-TIMEFRAME operand is NOT. The number moved on this branch, and it
      is not lookahead. exprs._align used to ffill by LABEL out of a source
      frame already cut at `now`, so a 15m spec reading 1h at now=16:00 had a
      cut 1h frame ending at the bar labelled 15:00, and row -2 (labelled
      15:30) inherited that bar even though it does not close until 16:00.
      _align now asks when each destination bar closed, so row -2 gets the bar
      labelled 14:00, the last one closed at 15:45. Both values were on the
      desk when the decision was taken, so neither reads the future; the
      earlier bar is now compared against the reference IT could see rather
      than the one the current bar can. No catalog play puts a cross-timeframe
      operand inside a cross, which is why the differential corpus over the
      catalog stayed byte-identical across this branch.

    Do not collapse this back into a plain shift on both sides.
    """
    if isinstance(node, dict):
        term = vocabulary.primitives.get(node.get("prim"))
        if term is not None and term.end_anchored:
            return v
    return v.shift(1)


class FrameEval:
    """Replay-scoped node cache over one symbol's untruncated frames."""

    # `vocabulary` is keyword-only: it sits behind an optional `tfs`, so
    # FrameEval(frames, vocab) would bind the Vocabulary to `tfs` and evaluate
    # every node against the core vocabulary instead. Nothing raises on that
    # path unless a cross-timeframe node reaches self.tfs, so the wrong answer
    # would look like a right one.
    def __init__(self, frames: dict, tfs: TimeframeSet = DEFAULT_TIMEFRAMES, *,
                 vocabulary: Vocabulary | None = None):
        self._frames = {tf: f for tf, f in frames.items()}
        self.tfs = tfs
        self.vocabulary = resolve_vocabulary(vocabulary)
        self._cache: dict = {}
        self._maps: dict = {}
        self._spans: dict = {}

    def on(self, tf: str) -> pd.DataFrame:
        return self._frames[tf]

    def set_span(self, tf: str, lo: int, hi: int) -> None:
        """Declare the rows of `tf` the end_anchored terms are computed on.

        The span is a correctness input for every end-anchored node, and it is
        deliberately NOT part of the memo key: keying on it would let one
        FrameEval hold two answers for the same node, and only one of them can
        be the values the caller's rows actually saw. So the span is fixed for
        the life of the cache, and moving it once anything is cached raises
        rather than serving the previous span's rows from behind the memo.

        Raising rather than dropping the affected entries is the deliberate
        half of that choice. This is the trap waiting for the next person to
        reuse one FrameEval across walk-forward windows: silence would make
        that reuse legal and wrong, while quietly invalidating would make it
        legal and pointless, since every node would be recomputed and the memo
        is the entire reason to reuse. An error puts the decision in front of
        whoever tries it. Build one FrameEval per span; both callers do.
        """
        span = (max(int(lo), 0), int(hi))
        if self._cache and self._span(tf) != span:
            raise ValueError(
                f"the {tf!r} span moved from {self._span(tf)} to {span} after "
                "nodes were already computed under the old one; build a new "
                "FrameEval per replay window rather than re-spanning this one")
        self._spans[tf] = span

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
            term = self.vocabulary.indicators[name]
            a = {**term.defaults,
                 **{k: v for k, v in node.items() if k not in ("ind", "of", "tf")}}
            if term.kind == "bar":
                out = term.fn(frame, a)
            else:
                of = node.get("of", {"src": "close"})
                s = self.series(of, src_tf)
                if isinstance(s, float):
                    s = pd.Series(s, index=frame.index)
                out = term.fn(s, a)
            if isinstance(out, pd.DataFrame):
                out = out[a["field"]]
            return self._align(out, src_tf, tf)
        name = node["prim"]
        term = self.vocabulary.primitives[name]
        a = {**term.defaults,
             **{k: v for k, v in node.items() if k not in ("prim", "tf")}}
        if term.end_anchored:
            # Exactly the span, with no warm-up margin in front of it. Row i is
            # the scalar function called on bars[:i+1], which does its own
            # `lookback` tail-trim, so a row needs nothing computed before it:
            # widening the range only calls the function for rows no reader can
            # reach, at `lookback` extra calls per node per replay.
            lo, hi = self._span(src_tf)
            part = end_anchored_series(term, None, frame, lo, hi, **a)
            out = pd.Series(np.nan, index=frame.index)
            out.iloc[lo:hi] = part.to_numpy()
            return self._align(out, src_tf, tf)
        if name == "bars_since":
            a["eval_fn"] = lambda cond, b: self.condition_series(cond, src_tf)
        return self._align(term.fn(None, frame, **a), src_tf, tf)

    def condition_series(self, cond: dict, tf: str) -> pd.Series:
        """Elementwise boolean series for a comparison condition."""
        index = self._frames[tf].index
        lhs, rhs = self.series(cond["lhs"], tf), self.series(cond["rhs"], tf)
        if not isinstance(lhs, pd.Series):
            lhs = pd.Series(lhs, index=index)
        if not isinstance(rhs, pd.Series):
            rhs = pd.Series(rhs, index=index)
        op = cond["op"]
        if op in ("crosses_above", "crosses_below"):
            lhs_prev = _cross_prev(cond["lhs"], lhs, self.vocabulary)
            rhs_prev = _cross_prev(cond["rhs"], rhs, self.vocabulary)
            out = ((lhs_prev <= rhs_prev) & (lhs > rhs) if op == "crosses_above"
                   else (lhs_prev >= rhs_prev) & (lhs < rhs))
            na = lhs.isna() | rhs.isna() | lhs_prev.isna() | rhs_prev.isna()
        else:
            out = {">": lhs > rhs, "<": lhs < rhs,
                   ">=": lhs >= rhs, "<=": lhs <= rhs}[op]
            na = lhs.isna() | rhs.isna()
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
