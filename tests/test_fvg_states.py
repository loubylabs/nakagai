"""FVG lifecycle: open, wick-filled, and close-through inversion."""

import pandas as pd
import pytest

from nakagai.strategies.base import Direction
from nakagai.strategies.ict.fvg import find_fvgs, find_unfilled_fvgs
from nakagai.strategies.rules.primitives import PRIMITIVES, fvg_nearest


def _bars(rows):
    idx = pd.date_range("2026-01-05 14:30", periods=len(rows), freq="15min", tz="UTC")
    df = pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"])
    df["volume"] = 1000.0
    return df


# 20 quiet bars keep ATR ~1, then a bullish FVG: low[i] > high[i-2] with a
# 2-point gap (bottom=101, top=103). The middle candle's low (100.4) overlaps
# the prior high (100.5) so no second, threshold-marginal gap sneaks in.
QUIET = [[100, 100.5, 99.5, 100]] * 20
GAP = QUIET + [[100, 101, 99.9, 100.9], [101, 102.5, 100.4, 102.4], [103.2, 104, 103, 103.8]]


def test_open_gap_reported_by_both():
    bars = _bars(GAP)
    open_gaps = find_unfilled_fvgs(bars)
    assert [g.direction for g in open_gaps] == [Direction.LONG]
    states = dict((g.ts, s) for g, s in find_fvgs(bars))
    assert states[open_gaps[0].ts] == "open"


def test_wick_fill_is_filled_not_inverted():
    # wick pierces the gap bottom (101) but the close holds above it
    bars = _bars(GAP + [[103.5, 103.6, 100.9, 103.0]])
    assert find_unfilled_fvgs(bars) == []
    (gap, state), = [t for t in find_fvgs(bars) if t[0].direction == Direction.LONG]
    assert state == "filled"


def test_close_through_inverts():
    # a bar CLOSES below the gap bottom: the bullish gap flips polarity
    bars = _bars(GAP + [[103.5, 103.6, 100.2, 100.5]])
    (gap, state), = [t for t in find_fvgs(bars) if t[0].direction == Direction.LONG]
    assert state == "inverted"


def test_fvg_nearest_inverted_maps_direction_to_the_new_side():
    bars = _bars(GAP + [[103.5, 103.6, 100.2, 100.5]])
    # formerly bullish gap, closed below: now resistance that supports shorts
    assert fvg_nearest(None, bars, direction="short", state="inverted",
                       field="bottom") == pytest.approx(101.0)
    # no inverted formerly-bearish gap exists, so the long side is NaN
    assert pd.isna(fvg_nearest(None, bars, direction="long", state="inverted"))


def test_fvg_nearest_open_default_unchanged():
    bars = _bars(GAP)
    assert fvg_nearest(None, bars, direction="long", field="top") == pytest.approx(103.0)


def test_min_size_atr_arg_filters():
    bars = _bars(GAP)
    assert pd.isna(fvg_nearest(None, bars, direction="long", min_size_atr=2.0))


def test_registry_schema():
    assert PRIMITIVES["fvg_nearest"]["args"] == {
        "direction": ("long", "short"), "field": ("top", "bottom", "mid"),
        "state": ("open", "inverted"),
        "min_size_atr": (0.05, 2.0), "lookback": (10, 200)}
