"""Margin evaluation: the ranking layer over FrameEval's node values.

The walk itself is FrameEval's and is covered in test_frame_eval.py; what is
left here is only what margins.py still owns, namely signed distances and the
rank-then-combine step.
"""

from datetime import time

import numpy as np
import pandas as pd
import pytest

from nakagai.data.schema import DEFAULT_TIMEFRAMES as TFS
from nakagai.strategies.indicators import rsi as _rsi
from nakagai.strategies.rules import validate_spec
from nakagai.strategies.rules.frame_eval import FrameEval
from nakagai.strategies.rules.margins import (condition_margin, group_margin,
                                              spec_margin)
from nakagai.strategies.rules.vocabulary import core_vocabulary
from nakagai.strategies.rules.windows import WindowSpec


def _bars(closes, start="2026-01-05 14:30", freq="15min"):
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5,
                         "close": c, "volume": 1000.0})


def _fe(b15, b1h=None, b1d=None):
    frames = {"15m": b15, "1h": b1h if b1h is not None else b15,
              "1d": b1d if b1d is not None else b15}
    return FrameEval(
        "SPY", {("SPY", tf): frame for tf, frame in frames.items()}, TFS)


def test_comparison_margin_is_signed_distance():
    b = _bars(np.linspace(100, 120, 40))
    fe = _fe(b)
    gt = condition_margin({"lhs": {"src": "close"}, "op": ">",
                           "rhs": {"ind": "sma", "n": 5}}, fe, "15m")
    lt = condition_margin({"lhs": {"src": "close"}, "op": "<",
                           "rhs": {"ind": "sma", "n": 5}}, fe, "15m")
    sma5 = b["close"].rolling(5).mean()
    assert gt.iloc[-1] == pytest.approx(b["close"].iloc[-1] - sma5.iloc[-1])
    assert lt.iloc[-1] == pytest.approx(sma5.iloc[-1] - b["close"].iloc[-1])
    assert (gt.dropna() + lt.dropna()).abs().max() < 1e-12


def test_cross_margin_is_the_current_gap():
    b = _bars([100.0] * 30 + [99.0, 103.0])
    fe = _fe(b)
    m = condition_margin({"lhs": {"src": "close"}, "op": "crosses_above",
                          "rhs": {"ind": "sma", "n": 5}}, fe, "15m")
    sma5 = b["close"].rolling(5).mean()
    assert m.iloc[-1] == pytest.approx(b["close"].iloc[-1] - sma5.iloc[-1])
    below = condition_margin({"lhs": {"src": "close"}, "op": "crosses_below",
                              "rhs": {"ind": "sma", "n": 5}}, fe, "15m")
    assert below.iloc[-1] == pytest.approx(sma5.iloc[-1] - b["close"].iloc[-1])


def test_scalar_rhs_broadcasts():
    b = _bars(np.linspace(10, 50, 30))
    m = condition_margin({"lhs": {"ind": "rsi", "n": 14}, "op": "<", "rhs": 30},
                         _fe(b), "15m")
    assert isinstance(m, pd.Series) and len(m) == len(b)
    expected_rsi = _rsi(b["close"], 14)
    assert m.iloc[-1] == pytest.approx(30 - expected_rsi.iloc[-1])


def test_windowed_aggregate_margin_is_a_series():
    b = _bars(np.linspace(100, 110, 30))
    vocabulary = core_vocabulary().with_windows(WindowSpec(
        "ny_open_30", "America/New_York", time(9, 30), time(10),
        "xnys_session", "standard"))
    m = condition_margin({"lhs": {"src": "close"}, "op": ">",
                          "rhs": {"ind": "highest", "of": {"src": "high"},
                                  "window": "ny_open_30"}},
                         FrameEval("SPY", {("SPY", "15m"): b}, TFS,
                                   vocabulary=vocabulary),
                         "15m")
    assert isinstance(m, pd.Series) and len(m) == len(b)


def test_group_margin_ranks_then_combines():
    b = _bars(np.linspace(100, 120, 40))
    fe = _fe(b)
    idx = b.index[10:]
    c1 = {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 5}}
    c2 = {"lhs": {"ind": "rsi", "n": 14}, "op": "<", "rhs": 70}
    m1 = condition_margin(c1, fe, "15m").loc[idx].rank(pct=True)
    m2 = condition_margin(c2, fe, "15m").loc[idx].rank(pct=True)
    g_all = group_margin({"all": [c1, c2]}, fe, "15m", idx)
    g_any = group_margin({"any": [c1, c2]}, fe, "15m", idx)
    both = pd.concat([m1, m2], axis=1)
    pd.testing.assert_series_equal(g_all, both.min(axis=1, skipna=False),
                                   check_names=False)
    pd.testing.assert_series_equal(g_any, both.max(axis=1), check_names=False)
    assert g_all.dropna().between(0, 1).all()


def test_group_margin_handles_nested_groups():
    b = _bars(np.linspace(100, 120, 40))
    fe = _fe(b)
    idx = b.index[10:]
    c1 = {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 5}}
    c2 = {"lhs": {"ind": "rsi", "n": 14}, "op": "<", "rhs": 70}
    c_flat = {"lhs": {"src": "close"}, "op": ">", "rhs": 0}
    group = {"all": [{"any": [c1, c2]}, c_flat]}
    g = group_margin(group, fe, "15m", idx)

    m1 = condition_margin(c1, fe, "15m").loc[idx].rank(pct=True)
    m2 = condition_margin(c2, fe, "15m").loc[idx].rank(pct=True)
    inner_any = pd.concat([m1, m2], axis=1).max(axis=1)
    m_flat = condition_margin(c_flat, fe, "15m").loc[idx].rank(pct=True)
    expected = pd.concat([inner_any.rank(pct=True), m_flat], axis=1).min(axis=1, skipna=False)
    pd.testing.assert_series_equal(g, expected, check_names=False)


def test_all_group_propagates_warmup_nan_any_survives_it():
    b = _bars(np.linspace(100, 120, 40))
    fe = _fe(b)
    idx = b.index          # includes sma warmup rows
    c_warm = {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 20}}
    c_ok = {"lhs": {"src": "close"}, "op": ">", "rhs": 0}
    g_all = group_margin({"all": [c_warm, c_ok]}, fe, "15m", idx)
    g_any = group_margin({"any": [c_warm, c_ok]}, fe, "15m", idx)
    assert g_all.iloc[:19].isna().all() and g_all.iloc[19:].notna().all()
    assert g_any.notna().all()


def test_not_group_margin_is_the_rank_complement_and_obeys_de_morgan():
    """De Morgan over a real frame, which is what forces `not` to be 1 - m.

    group_margin works in rank-percentile space: a member is .rank(pct=True),
    `all` takes the min and `any` the max. The complement of a margin m is
    therefore 1 - m and nothing else, because that is the only reading under
    which not(all(a, b)) and any(not a, not b) are the same number:

        1 - min(ra, rb) == max(1 - ra, 1 - rb)

    Any other choice would let two logically identical specs produce two
    different factors, and the ICIR lens would score them apart.

    The window starts past every member's warm-up on purpose. `all` reduces
    with skipna=False and `any` with skipna=True, so the two sides of the law
    disagree wherever a member is NaN; that asymmetry predates `not` and is
    not what this test is about.
    """
    b = _bars(np.linspace(100, 120, 40))
    fe = _fe(b)
    idx = b.index[20:]
    a = {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 5}}
    c = {"lhs": {"ind": "rsi", "n": 14}, "op": "<", "rhs": 70}

    negated_all = group_margin({"not": {"all": [a, c]}}, fe, "15m", idx)
    # `not` takes a group, never a bare leaf (N3-D6), so a single negated
    # condition is spelled {"not": {"all": [<leaf>]}}; a one-member `all` is
    # that member's rank, so these two ARE `not a` and `not c`.
    not_a = group_margin({"not": {"all": [a]}}, fe, "15m", idx)
    not_c = group_margin({"not": {"all": [c]}}, fe, "15m", idx)
    assert negated_all.notna().all() and not_a.notna().all()

    pd.testing.assert_series_equal(
        negated_all, pd.concat([not_a, not_c], axis=1).max(axis=1),
        check_names=False)
    # and the complement really is the complement of the group it negates,
    # rather than the law holding vacuously over two constant series
    plain_all = group_margin({"all": [a, c]}, fe, "15m", idx)
    pd.testing.assert_series_equal(negated_all, 1.0 - plain_all,
                                   check_names=False)
    assert plain_all.nunique() > 1


def test_spec_margin_ranks_a_not_group_top_level_and_nested():
    """A validated spec carrying `not` produces a finite ranked factor.

    margins.py recognized only all/any, so a top-level `not` reached
    condition_margin and raised AttributeError on cond["lhs"], and a nested
    one raised KeyError.

    What that costs is an availability failure and not a wrong number, which
    is worth stating precisely because it used to be the other way round. The
    IC lens calls this through `strategy_operation` (engine/ic.py:249), and
    that door swallows nothing: the AttributeError arrives as
    StrategyRuntimeError("strategy_raised", "ic_factor raised AttributeError")
    naming the play and the symbol, no `except` anywhere under nakagai/engine
    catches it, and `run_portfolio` refuses the whole replay. Measured on this
    branch by reverting margins.py alone and replaying one negated spec.

    So without this walker every backtest of a negated spec dies, loudly. The
    pre-Phase-1 engine substituted empty IC fields for any exception here, and
    the same crash wrote null correlations over zero observations instead,
    which read exactly like a legitimate abstention. That silent reading is
    gone with the engine that produced it; see chrvsd/nakagai#411.
    """
    b = _bars(np.linspace(100, 120, 40))
    fe = _fe(b)
    idx = b.index[20:]
    a = {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 5}}
    c = {"lhs": {"ind": "rsi", "n": 14}, "op": "<", "rhs": 70}
    top = {"version": 2, "name": "t", "timeframe": "15m",
           "long": {"not": {"all": [a, c]}}}
    nested = {"version": 2, "name": "t", "timeframe": "15m",
              "long": {"all": [{"not": {"any": [a, c]}}, a]},
              "short": {"any": [{"not": {"all": [c]}}, a]}}
    for spec in (top, nested):
        assert validate_spec(spec) == []
        m = spec_margin(spec, fe, idx)
        assert isinstance(m, pd.Series) and len(m) == len(idx)
        assert np.isfinite(m.to_numpy()).all()


def test_spec_margin_long_only_and_sides():
    b = _bars(np.linspace(100, 120, 40))
    fe = _fe(b)
    idx = b.index[10:]
    cond = {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 5}}
    long_only = {"version": 2, "name": "t", "timeframe": "15m",
                 "long": {"all": [cond]}}
    short_only = {"version": 2, "name": "t", "timeframe": "15m",
                  "short": {"all": [cond]}}
    ml = spec_margin(long_only, fe, idx)
    ms = spec_margin(short_only, fe, idx)
    pd.testing.assert_series_equal(ml, -ms, check_names=False)
    assert ml.dropna().between(0, 1).all()


def test_spec_margin_empty_spec_is_empty():
    b = _bars(np.linspace(100, 120, 40))
    out = spec_margin({"version": 2, "name": "t", "timeframe": "15m"},
                      _fe(b), b.index)
    assert isinstance(out, pd.Series) and out.empty


def test_bars_since_shares_the_one_walker_and_its_alignment():
    # There used to be two walkers keying their memos identically, so a
    # cross-timeframe node reachable both inside a bars_since cond and as a
    # direct condition poisoned whichever cache filled second. One walker means
    # one memo and one alignment: the daily close stays the previous session's
    # on both routes rather than leaking Jan 6's into Jan 6's own rows.
    b15 = _bars([100.0] * 26, start="2026-01-06 14:30")
    b1d = _bars([95.0, 96.0], start="2026-01-05 00:00", freq="1D")
    fe = _fe(b15, b1d=b1d)
    ref = {"lhs": {"src": "close"}, "op": ">", "rhs": {"src": "close", "tf": "1d"}}
    bars_since_cond = {"lhs": {"prim": "bars_since", "cond": ref}, "op": ">", "rhs": 0}
    group = {"all": [bars_since_cond, ref]}
    group_margin(group, fe, "15m", b15.index)
    assert (fe.series({"src": "close", "tf": "1d"}, "15m") == 95.0).all()
