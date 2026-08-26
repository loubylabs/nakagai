from datetime import time

import numpy as np
import pandas as pd
import pytest

from nakagai.data.schema import DEFAULT_TIMEFRAMES as TFS
from nakagai.strategies.base import Direction, MarketContext
from nakagai.strategies.rules import RuleStrategy
from nakagai.strategies.rules.frame_eval import FrameEval
from nakagai.strategies.rules.vocabulary import core_vocabulary
from nakagai.strategies.rules.windows import WindowSpec

RISK = {"stop": {"kind": "atr", "n": 14, "mult": 2.0}, "target": {"kind": "rr", "rr": 2.0}}


def _bars(closes, start="2026-01-05 14:30", freq="15min"):
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5,
                         "close": c, "volume": 1000.0})


def _frames(b15, b1h=None, b1d=None) -> dict:
    return {"15m": b15, "1h": b1h if b1h is not None else b15,
            "1d": b1d if b1d is not None else b15}


def _fe(b15, b1h=None, b1d=None) -> FrameEval:
    return FrameEval(_frames(b15, b1h, b1d), TFS)


def _ctx(b15, b1h=None, b1d=None, *, vocabulary=None) -> MarketContext:
    """A context shaped the way build_context builds one: cut frames, a walker
    over them, and a cursor on the last row of each."""
    frames = _frames(b15, b1h, b1d)
    return MarketContext("SPY", b15.index[-1] + pd.Timedelta(minutes=15),
                         bars=frames, tfs=TFS,
                         fe=FrameEval(frames, TFS, vocabulary=vocabulary),
                         cursor={tf: len(f) - 1 for tf, f in frames.items()})


def _holds(group, fe, tf="15m") -> bool:
    return bool(fe.group_series(group, tf).iloc[-1])


def test_math_and_indicator_composition():
    b = _bars(np.linspace(100, 120, 60))
    node = {"op": "*", "args": [2.0, {"ind": "sma", "n": 5, "of": {"src": "close"}}]}
    out = _fe(b).series(node, "15m")
    assert out.iloc[-1] == pytest.approx(2 * b["close"].iloc[-5:].mean())


def test_division_by_zero_is_nan_and_condition_false():
    b = _bars([100.0] * 30)
    group = {"all": [{"lhs": {"op": "/", "args": [{"src": "close"}, 0]}, "op": ">", "rhs": 0}]}
    assert _holds(group, _fe(b)) is False


def test_cross_timeframe_alignment_no_lookahead():
    b15 = _bars(np.linspace(100, 110, 60))
    b1d = _bars([90.0, 95.0], start="2026-01-03 00:00", freq="1D")
    node = {"src": "close", "tf": "1d"}
    out = _fe(b15, b1d=b1d).series(node, "15m")
    assert out.iloc[-1] == 95.0
    assert len(out) == len(b15)


def test_series_indicator_with_tf_computes_on_native_frame():
    # sma(2) on 1d must average the last TWO DAILY closes (105), not two 15m
    # samples of the upsampled daily series (which would give 110).
    b15 = _bars(np.linspace(100, 110, 60))
    b1d = _bars([90.0, 100.0, 110.0], start="2026-01-02 00:00", freq="1D")
    node = {"ind": "sma", "n": 2, "tf": "1d"}
    out = _fe(b15, b1d=b1d).series(node, "15m")
    assert len(out) == len(b15)
    assert out.iloc[-1] == pytest.approx(105.0)


def test_series_indicator_with_tf_explicit_of_matches_implicit():
    b15 = _bars(np.linspace(100, 110, 60))
    b1d = _bars([90.0, 100.0, 110.0], start="2026-01-02 00:00", freq="1D")
    fe = _fe(b15, b1d=b1d)
    implicit = fe.series({"ind": "sma", "n": 2, "tf": "1d"}, "15m")
    explicit = fe.series({"ind": "sma", "n": 2, "tf": "1d", "of": {"src": "close"}},
                         "15m")
    assert implicit.iloc[-1] == pytest.approx(105.0)
    assert explicit.iloc[-1] == pytest.approx(105.0)


def test_crosses_above_uses_last_two_bars():
    closes = [100.0] * 30 + [99.0, 101.0]
    b = _bars(closes)
    group = {"all": [{"lhs": {"src": "close"}, "op": "crosses_above", "rhs": 100.0}]}
    assert _holds(group, _fe(b)) is True


def test_nan_warmup_condition_false():
    b = _bars([100.0, 101.0, 102.0])   # far less than sma 200 warmup
    group = {"all": [{"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 200}}]}
    assert _holds(group, _fe(b)) is False


def test_memo_reuses_identical_subtrees(monkeypatch):
    import nakagai.strategies.indicators as ind
    calls = {"n": 0}
    real = ind.sma
    monkeypatch.setattr(ind, "sma", lambda s, n: calls.__setitem__("n", calls["n"] + 1) or real(s, n))
    b = _bars(np.linspace(100, 120, 60))
    fe = _fe(b)
    node = {"ind": "sma", "n": 20}
    fe.series(node, "15m")
    fe.series(node, "15m")
    assert calls["n"] == 1


def test_rule_strategy_emits_signal_and_validates():
    spec = {"version": 2, "name": "cross", "timeframe": "15m",
            "long": {"all": [{"lhs": {"src": "close"}, "op": "crosses_above", "rhs": 100.0}]},
            "risk": RISK}
    b = _bars([100.0] * 30 + [99.0, 101.0])
    sigs = RuleStrategy({"spec": spec}).on_bar(_ctx(b))
    assert len(sigs) == 1 and sigs[0].direction == Direction.LONG
    with pytest.raises(ValueError):
        RuleStrategy({"spec": {"version": 1, "name": "old"}})
    assert RuleStrategy({}).on_bar(_ctx(b)) == ()      # inert without a spec


def test_window_aggregate_in_condition_end_to_end():
    # first two 15m bars set the 30m opening window; the tape then holds below
    # it and only breaks out on the final bar, so the crossing lands on the
    # bar RuleStrategy actually evaluates.
    ramp = list(np.linspace(100, 100.3, 26))
    b = _bars(ramp + [100.3] * 5 + [103.0])
    spec = {"version": 2, "name": "orb", "timeframe": "15m",
            "long": {"all": [{"lhs": {"src": "close"}, "op": "crosses_above",
                              "rhs": {"ind": "highest", "of": {"src": "high"},
                                      "window": "ny_open_30"}}]},
            "risk": RISK}
    vocabulary = core_vocabulary().with_windows(WindowSpec(
        "ny_open_30", "America/New_York", time(9, 30), time(10),
        "xnys_session", "standard"))
    sigs = RuleStrategy({"spec": spec}, vocabulary=vocabulary).on_bar(
        _ctx(b, vocabulary=vocabulary))
    assert len(sigs) == 1
