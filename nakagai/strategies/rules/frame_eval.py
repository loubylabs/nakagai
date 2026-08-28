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

import functools
import json
import operator

import numpy as np
import pandas as pd

from nakagai.data.schema import DEFAULT_TIMEFRAMES, TimeframeSet, _is_session_frame
from nakagai.engine.context import visible_counts
from nakagai.strategies.rules.primitives import end_anchored_series
from nakagai.strategies.rules.spec import is_group_node
from nakagai.strategies.rules.vocabulary import (
    Vocabulary, is_condition_rule, resolve_vocabulary,
)
from nakagai.strategies.rules.windows import aggregate_window


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
    """Replay-scoped expression cache over pair-keyed market frames."""

    def __init__(self, driving_symbol: str, frames: dict,
                 tfs: TimeframeSet = DEFAULT_TIMEFRAMES, *,
                 vocabulary: Vocabulary | None = None):
        bad = [key for key in frames
               if not (isinstance(key, tuple) and len(key) == 2
                       and all(isinstance(part, str) for part in key))]
        if bad:
            raise TypeError(
                "FrameEval frames must be pair-keyed as (symbol, timeframe); "
                f"invalid keys: {bad!r}")
        self.driving_symbol = driving_symbol
        self._frames = dict(frames)
        self.tfs = tfs
        self.vocabulary = resolve_vocabulary(vocabulary)
        self._cache: dict = {}
        self._maps: dict = {}
        self._spans: dict = {}
        self._missing: dict = {}

    def on(self, symbol: str, tf: str) -> pd.DataFrame:
        return self._frames[(symbol, tf)]

    def set_span(self, symbol: str, tf: str, lo: int, hi: int) -> None:
        """Fix the end-anchored evaluation span for one symbol pair."""
        pair = (symbol, tf)
        span = (max(int(lo), 0), int(hi))
        if self._cache and self._span(pair) != span:
            raise ValueError(
                f"the {pair!r} span moved from {self._span(pair)} to {span} "
                "after nodes were already computed under the old one; build "
                "a new FrameEval per replay window")
        self._spans[pair] = span

    def _span(self, pair: tuple[str, str]) -> tuple[int, int]:
        return self._spans.get(pair, (0, len(self._frames[pair])))

    def _own_missing(self, pair: tuple[str, str]) -> pd.Series:
        if pair not in self._missing:
            frame = self._frames[pair]
            self._missing[pair] = frame.isna().all(axis=1).astype(bool)
        return self._missing[pair]

    def _positions(self, src_pair: tuple[str, str],
                   dst_pair: tuple[str, str]) -> np.ndarray:
        """Row -> index of the last `src_tf` bar closed at that row's close.

        A session-aligned destination has no honest close time derivable from
        bar labels alone: a daily bar's session closes at 16:00 NY, but the
        label carries only the date. margins.py used to approximate it as
        label + 1 day, which is correct only while the cache stays RTH-only.
        Rather than carry that assumption into the signal path, refuse it. No
        catalog spec pairs a session-aligned timeframe with a cross-timeframe
        reference, and a user spec that does deserves an error, not a guess.
        """
        src_tf, dst_tf = src_pair[1], dst_pair[1]
        key = (*src_pair, *dst_pair)
        if key not in self._maps:
            if dst_tf in self.tfs.session_aligned:
                raise ValueError(
                    f"spec timeframe {dst_tf!r} is session-aligned, so a "
                    f"reference to {src_tf!r} has no well-defined visibility "
                    "cutoff; move the spec to an intraday timeframe")
            dst = self._frames[dst_pair].index
            counts = visible_counts(self._frames[src_pair].index,
                                    dst + self.tfs.deltas[dst_tf], src_tf, self.tfs)
            self._maps[key] = counts - 1
        return self._maps[key]

    def to_driving(self, v, src_symbol: str, src_tf: str):
        """Lift a native series onto the driving index by visibility.

        The signal path reads at a driving-bar cursor, but conditions must be
        computed on their own timeframe first: `crosses_above` on a 1h spec has
        to compare consecutive 1h bars, which is what the per-bar path did when
        it took .iloc[-2] and .iloc[-1] of a 1h prefix. Computing natively and
        lifting the BOOLEAN preserves that exactly; comparing on the driving
        index would compare against the previous 15m bar instead.
        """
        return self._align(
            v,
            (src_symbol, src_tf),
            (self.driving_symbol, self.tfs.driving),
        )

    def _align(self, v, src_pair: tuple[str, str],
               dst_pair: tuple[str, str]):
        """Carry a native series onto another timeframe's index by visibility.

        dtype is preserved: booleans lift as booleans (invisible rows False),
        floats as floats (invisible rows NaN). A float cast here would turn a
        condition series into 0.0/1.0 and silently make `not visible` truthy.
        """
        if not isinstance(v, pd.Series) or src_pair == dst_pair:
            return v
        pos = self._positions(src_pair, dst_pair)
        ok = pos >= 0
        index = self._frames[dst_pair].index
        if v.dtype == bool or isinstance(v.dtype, pd.BooleanDtype):
            out = pd.Series(pd.NA, index=index, dtype="boolean")
            if ok.any():
                values = v.astype("boolean").to_numpy()
                out.iloc[np.flatnonzero(ok)] = values[pos[ok]]
            if isinstance(v.dtype, pd.BooleanDtype):
                return out
            return out.fillna(False).astype(bool)
        vals = v.to_numpy(dtype="float64")
        out = np.full(len(pos), np.nan)
        out[ok] = vals[pos[ok]]
        return pd.Series(out, index=index)

    def _align_mask(self, mask: pd.Series, src_pair: tuple[str, str],
                    dst_pair: tuple[str, str]) -> pd.Series:
        if src_pair == dst_pair:
            return mask
        pos = self._positions(src_pair, dst_pair)
        ok = pos >= 0
        out = np.ones(len(pos), dtype=bool)
        out[ok] = mask.to_numpy(dtype=bool)[pos[ok]]
        return pd.Series(out, index=self._frames[dst_pair].index)

    @staticmethod
    def _masked(value, mask: pd.Series):
        if not isinstance(value, pd.Series):
            return value
        if value.dtype == bool or isinstance(value.dtype, pd.BooleanDtype):
            return value.astype("boolean").mask(mask, pd.NA)
        return value.mask(mask)

    def _align_result(self, result, src_pair: tuple[str, str],
                      dst_pair: tuple[str, str]):
        value, mask = result
        return (
            self._align(value, src_pair, dst_pair),
            self._align_mask(mask, src_pair, dst_pair),
        )

    def series(self, node, tf: str):
        """Evaluate from the traded symbol and return only the public value."""
        value, _ = self._series_result(
            node, (self.driving_symbol, tf))
        return value

    def _series_result(self, node, host_pair: tuple[str, str]):
        if isinstance(node, (int, float)):
            return (
                float(node),
                pd.Series(False, index=self._frames[host_pair].index),
            )
        native_pair = (
            node.get("sym", host_pair[0]),
            node.get("tf", host_pair[1]),
        )
        key = (*native_pair, json.dumps(node, sort_keys=True))
        if key in self._cache:
            native = self._cache[key]
        else:
            self._cache[key] = native = self._eval(node, native_pair)
        return self._align_result(native, native_pair, host_pair)

    def _eval(self, node: dict, pair: tuple[str, str]):
        frame = self._frames[pair]
        mask = self._own_missing(pair).copy()
        if "src" in node:
            return self._masked(frame[node["src"]], mask), mask
        if "op" in node:
            children = [self._series_result(a, pair) for a in node["args"]]
            for _, child_mask in children:
                mask |= child_mask
            out = _math(node["op"], [value for value, _ in children])
            return self._masked(out, mask), mask
        if "ind" in node:
            name = node["ind"]
            term = self.vocabulary.indicators[name]
            a = {**term.defaults,
                 **{k: v for k, v in node.items()
                    if k not in ("ind", "of", "tf", "sym", "window")}}
            if term.kind == "bar":
                out = term.fn(frame, a)
            else:
                of = node.get("of", {"src": "close"})
                s, child_mask = self._series_result(of, pair)
                mask |= child_mask
                if not isinstance(s, pd.Series):
                    s = pd.Series(s, index=frame.index, dtype="float64")
                if "window" in node:
                    out = aggregate_window(
                        s,
                        self.vocabulary.windows[node["window"]],
                        term.window_reduce,
                        session_aligned=_is_session_frame(frame.index),
                    )
                else:
                    out = term.fn(s, a)
            if isinstance(out, pd.DataFrame):
                out = out[a["field"]]
            return self._masked(out, mask), mask
        name = node["prim"]
        term = self.vocabulary.primitives[name]
        a = {**term.defaults,
             **{k: v for k, v in node.items()
                if k not in ("prim", "tf", "sym")}}
        condition_args = [arg for arg, rule in term.args.items()
                          if is_condition_rule(rule) and arg in node]
        if condition_args:
            # Generic per N3-D9, keyed on the arg TYPE rather than the name
            # "bars_since": a term registered outside core (a platform
            # vocabulary entry, added with no core PR) reaches this the same
            # way. The callback itself is unchanged; it stays injected rather
            # than imported to avoid a circular import (primitives.py
            # documents this), which typing the arg does not change.
            #
            # ABOVE the end_anchored branch, because that branch returns and a
            # term needs its evaluator whichever dispatch it takes. N3-D13
            # refuses a condition-typed arg outside kind 'primitive' and
            # nowhere else, so an end-anchored primitive declaring one is
            # constructible and validates; injecting only below here left it
            # raising at evaluation time instead.
            #
            # CUT TO THE FRAME THE TERM WAS HANDED, which is what makes that
            # placement safe rather than merely non-raising. end_anchored_series
            # calls term.fn on bars[:i+1], so a callback answering over the
            # whole frame hands row i a mask that includes rows after it: a term
            # summing it reads [4, 4, 4, 4] where the causal answer is
            # [1, 2, 3, 4]. That is a lookahead in the one node whose hard rule
            # 1 is causality, and verify_term cannot catch it, because it
            # exempts every end_anchored term. `.loc` rather than `.reindex`:
            # `b` is always a prefix of this same frame, so a missing label is a
            # miswiring and should raise here rather than become a silent NaN.
            for arg in condition_args:
                _, child_mask = self._condition_result(node[arg], pair)
                mask |= child_mask
            a["eval_fn"] = (
                lambda cond, b: self._condition_result(cond, pair)[0]
                .fillna(False).astype(bool).loc[b.index]
            )
        if term.end_anchored:
            # Exactly the span, with no warm-up margin in front of it. Row i is
            # the scalar function called on bars[:i+1], which does its own
            # `lookback` tail-trim, so a row needs nothing computed before it:
            # widening the range only calls the function for rows no reader can
            # reach, at `lookback` extra calls per node per replay.
            lo, hi = self._span(pair)
            part = end_anchored_series(term, None, frame, lo, hi, **a)
            out = pd.Series(np.nan, index=frame.index)
            out.iloc[lo:hi] = part.to_numpy()
            return self._masked(out, mask), mask
        out = term.fn(None, frame, **a)
        return self._masked(out, mask), mask

    def condition_series(self, cond: dict, tf: str) -> pd.Series:
        """Elementwise boolean series for a comparison condition. dtype ==
        bool: an unknown operand resolves to not-fired HERE, at the public
        boundary, per N3-D4. See _condition_series_na for the private
        Kleene-preserving leaf this is built from.
        """
        return self._condition_result(
            cond, (self.driving_symbol, tf))[0].fillna(False).astype(bool)

    def _condition_series_na(self, cond: dict, symbol: str,
                             tf: str) -> pd.Series:
        """Elementwise NULLABLE boolean series: dtype "boolean", pd.NA where
        an operand is unknown. Private per N3-D4: pd.NA lives only beneath
        group_series's and driving_group's public boundary, and the one other
        reader of an unknown condition, a condition-typed arg's injected
        eval_fn, reads condition_series (the bool-resolving public method)
        rather than this one, so an unknown condition still does not count as
        an occurrence there, which is today's behavior preserved deliberately.
        """
        return self._condition_result(cond, (symbol, tf))[0]

    def _condition_result(self, cond: dict, pair: tuple[str, str]):
        index = self._frames[pair].index
        lhs, lhs_mask = self._series_result(cond["lhs"], pair)
        rhs, rhs_mask = self._series_result(cond["rhs"], pair)
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
        # out is plain-bool: a float comparison against NaN reads False, never
        # NaN, so the cast to nullable "boolean" is exact and `na` is what
        # turns the right positions into pd.NA rather than a lossy False.
        mask = self._own_missing(pair) | lhs_mask | rhs_mask
        unknown = mask | na
        return out.astype("boolean").mask(unknown, pd.NA), mask

    def _group_reduce_na(self, group: dict, symbol: str,
                         tf: str) -> pd.Series:
        """The private Kleene-preserving reducer: nullable boolean throughout.

        N3-D2: all/any reduce element-wise with & / |, via functools.reduce,
        NEVER DataFrame.all/any(axis=1). Measured wrong for `all` in the
        dangerous direction: DataFrame.all(axis=1, skipna=False) over
        [True, NA] and over [NA, NA] both read True, where D8 requires NA.
        `any` happens to agree under both readings, which is why a test that
        only exercises `any` would pass over this defect.

        `not` is Kleene negation on the nullable dtype: ~NA is NA, never
        True, which is what stops a warming indicator from firing through a
        negation on its first bars.
        """
        key, val = next(iter(group.items()))
        if key == "not":
            return ~self._group_reduce_na(val, symbol, tf)
        parts = [self._group_reduce_na(i, symbol, tf) if is_group_node(i)
                 else self._condition_series_na(i, symbol, tf) for i in val]
        op = operator.and_ if key == "all" else operator.or_
        return functools.reduce(op, parts)

    def group_series(self, group: dict, tf: str) -> pd.Series:
        """all/any/not tree as one boolean series on `tf`'s index. dtype ==
        bool: an unknown group result resolves to not-fired HERE, at the
        public boundary, per N3-D4."""
        return self._group_reduce_na(
            group, self.driving_symbol, tf).fillna(False).astype(bool)

    def driving_group(self, group: dict, tf: str) -> pd.Series:
        """`group` as a boolean series on the DRIVING index, computed once.

        This is what the signal path reads at its cursor, and it has to be
        memoized like series() is. group_series walks the tree and to_driving
        lifts the result; both are O(frame), so calling them once per bar would
        put replay straight back on O(bars x history) with the indicator cache
        doing nothing but hiding it.
        """
        key = ("group", self.driving_symbol, tf,
               json.dumps(group, sort_keys=True))
        if key not in self._cache:
            native = self._group_reduce_na(group, self.driving_symbol, tf)
            lifted = self._align(
                native,
                (self.driving_symbol, tf),
                (self.driving_symbol, self.tfs.driving),
            )
            self._cache[key] = lifted.fillna(False).astype(bool)
        return self._cache[key]
