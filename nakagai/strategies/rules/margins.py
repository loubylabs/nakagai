"""Margin evaluation: a spec's rule tree as a graded signal-strength series.

Evaluation-only (the ICIR lens); nothing here feeds the signal path. Node
values come from FrameEval, which is the one walker over the grammar; this
module only turns comparisons into signed distances and ranks them.

Caveat that survives the collapse: group members are ranked within the full
evaluation window, so the combined margin is a diagnostic over that window, not
a tradable point-in-time series.
"""

import pandas as pd

from nakagai.strategies.rules.spec import is_group_node


def condition_margin(cond: dict, fe, tf: str) -> pd.Series:
    """Signed distance of a condition: positive = holds, magnitude = how
    strongly. Crosses grade the current gap; the cross event itself stays a
    signal-path concept."""
    index = fe.on(tf).index
    lhs, rhs = fe.series(cond["lhs"], tf), fe.series(cond["rhs"], tf)
    if not isinstance(lhs, pd.Series):
        lhs = pd.Series(lhs, index=index)
    if not isinstance(rhs, pd.Series):
        rhs = pd.Series(rhs, index=index)
    if cond["op"] in (">", ">=", "crosses_above"):
        return lhs - rhs
    return rhs - lhs


def group_margin(group: dict, fe, tf: str, index: pd.DatetimeIndex) -> pd.Series:
    """One all/any/not group as a percentile margin on `index` rows. Members
    are rank-transformed within `index` (the walk-forward window) so different
    native scales combine fairly; `all` takes the min (an unknown member keeps
    the row unknown), `any` the max (one known member suffices), and `not` the
    complement 1 - m.

    1 - m is the only reading of `not` this space admits, and it is forced
    rather than chosen. Everything here is a rank percentile, so the
    complement of a rank is one minus it, and that is exactly what makes De
    Morgan hold across the reduction:

        not(all(a, b)) -> 1 - min(ra, rb) == max(1 - ra, 1 - rb) ->
        any(not a, not b)

    Under any other complement the two spellings of one logically identical
    spec become two different factors, and the ICIR lens scores them apart.
    """
    key, val = next(iter(group.items()))
    if key == "not":
        # `not` takes a single nested GROUP, never a list of items (N3-D6), so
        # it recurses on that group rather than through the comprehension
        # below: `for i in val` over a dict would iterate its KEY STRINGS
        # instead of raising, and `"all" in "all"` (a substring test, which is
        # what the inline recognizer here used to do to a string) would send
        # the key itself back in as a group.
        return 1.0 - group_margin(val, fe, tf, index)
    members = [(group_margin(i, fe, tf, index) if is_group_node(i)
                else condition_margin(i, fe, tf).loc[index]).rank(pct=True)
               for i in val]
    both = pd.concat(members, axis=1)
    return both.min(axis=1, skipna=False) if key == "all" else both.max(axis=1)


def spec_margin(spec: dict, fe, index: pd.DatetimeIndex) -> pd.Series:
    """The spec as one graded factor on `index` rows: rank(long) minus
    rank(short), missing side = 0. Positive IC downstream always means the
    signal points the right way, for shorts too."""
    tf = spec.get("timeframe", "1h")
    sides = {side: group_margin(spec[side], fe, tf, index)
             for side in ("long", "short") if side in spec}
    if not sides:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=index)
    return sides.get("long", zero) - sides.get("short", zero)
