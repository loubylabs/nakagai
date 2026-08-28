"""RuleStrategy integration with pair-keyed expression evaluation."""

from types import SimpleNamespace

import pandas as pd

from nakagai.data.schema import DEFAULT_TIMEFRAMES as TFS
from nakagai.strategies.rules import RuleStrategy
from nakagai.strategies.rules.frame_eval import FrameEval


def _bars(values):
    index = pd.date_range(
        "2026-01-05 14:30", periods=len(values), freq="15min", tz="UTC")
    close = pd.Series(values, index=index, dtype="float64")
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000.0,
        },
        index=index,
    )


def test_group_at_starts_omitted_symbol_scope_at_the_traded_symbol():
    aapl = _bars([1, 2, 3])
    spy = _bars([9, 8, 7])
    evaluator = FrameEval(
        "AAPL",
        {("AAPL", "15m"): aapl, ("SPY", "15m"): spy},
        TFS,
    )
    strategy = RuleStrategy({"spec": {
        "version": 2,
        "name": "driving-default",
        "timeframe": "15m",
        "long": {"all": [{
            "lhs": {"src": "close"},
            "op": "<",
            "rhs": {"src": "close", "sym": "SPY"},
        }]},
        "risk": {
            "stop": {"kind": "atr", "n": 14, "mult": 2.0},
            "target": {"kind": "rr", "rr": 2.0},
        },
    }})
    context = SimpleNamespace(
        fe=evaluator,
        tfs=TFS,
        cursor={TFS.driving: len(aapl) - 1},
    )

    assert strategy._group_at(context, strategy.spec["long"]) is True
