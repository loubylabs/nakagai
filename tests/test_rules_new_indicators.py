"""The four Playbook indicators as rule-IR grammar: validate, eval, describe."""

import numpy as np
import pandas as pd

from nakagai.data.schema import DEFAULT_TIMEFRAMES as TFS
from nakagai.strategies import indicators as ind
from nakagai.strategies.rules import describe_spec, validate_spec
from nakagai.strategies.rules.frame_eval import FrameEval


def _spec(cond):
    return {"version": 2, "name": "t", "timeframe": "1h", "long": {"all": [cond]}}


def test_new_bar_indicators_validate():
    conds = [
        {"lhs": {"src": "close"}, "op": "crosses_above",
         "rhs": {"ind": "keltner", "n": 20, "mult": 1.5, "field": "upper"}},
        {"lhs": {"ind": "cci", "n": 20}, "op": "crosses_above", "rhs": -100},
        {"lhs": {"ind": "mfi", "n": 14}, "op": "crosses_above", "rhs": 20},
        {"lhs": {"ind": "wpr", "n": 14}, "op": "crosses_above", "rhs": -50},
    ]
    for cond in conds:
        assert validate_spec(_spec(cond)) == [], cond


def test_new_indicator_arg_errors():
    bad_field = {"lhs": {"ind": "keltner", "field": "nope"}, "op": ">", "rhs": 0}
    assert any("keltner.field" in e for e in validate_spec(_spec(bad_field)))
    with_of = {"lhs": {"ind": "mfi", "of": {"src": "close"}}, "op": ">", "rhs": 0}
    assert any("takes no `of`" in e for e in validate_spec(_spec(with_of)))
    oob = {"lhs": {"ind": "cci", "n": 1000}, "op": ">", "rhs": 0}
    assert any("cci.n" in e for e in validate_spec(_spec(oob)))


def _fe():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2026-01-05 14:30", periods=120, freq="15min", tz="UTC")
    c = pd.Series(100 + rng.normal(0, 1, 120).cumsum(), index=idx)
    bars = pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5,
                         "close": c, "volume": 1000.0}, index=idx)
    return FrameEval(
        "SPY", {("SPY", tf): bars for tf in ("15m", "1h", "1d")}, TFS,
    ), bars


def test_eval_matches_direct_indicator_calls():
    fe, bars = _fe()
    out = fe.series({"ind": "wpr", "n": 14}, "15m")
    pd.testing.assert_series_equal(out, ind.wpr(bars, 14), check_names=False)
    kc = fe.series({"ind": "keltner", "n": 20, "mult": 1.5, "field": "upper"},
                   "15m")
    pd.testing.assert_series_equal(kc, ind.keltner(bars, 20, 1.5)["upper"],
                                   check_names=False)


def test_describe_spec_renders_new_indicators():
    cond = {"lhs": {"ind": "mfi", "n": 14}, "op": "crosses_above", "rhs": 20}
    assert "mfi(14)" in describe_spec(_spec(cond))
