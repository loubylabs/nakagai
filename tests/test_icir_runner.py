"""run_one rows carry per-window IC fields for rule specs, and only for them."""

import numpy as np
import pandas as pd

from nakagai.data.cache import MemoryBars
from nakagai.data.schema import TimeframeSet
from nakagai.engine.runner import run_one
from nakagai.engine.windows import Window
from nakagai.strategies.base import Strategy
from nakagai.strategies.rules.strategy import RuleStrategy

TFS = TimeframeSet(driving="15m", deltas={"15m": pd.Timedelta(minutes=15)})

SPEC = {"version": 2, "name": "icir-test", "timeframe": "15m",
        "long": {"all": [{"lhs": {"src": "close"}, "op": ">",
                          "rhs": {"ind": "sma", "n": 5}}]}}


def _frame(closes, start="2026-01-05 14:30"):
    idx = pd.date_range(start, periods=len(closes), freq="15min", tz="UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5,
                         "close": c, "volume": 1000.0})


class _Inert(Strategy):
    name = "inert"
    DEFAULT_PARAMS = {}

    def on_bar(self, ctx):
        return []


def _registry():
    return {"rules": RuleStrategy, "inert": _Inert}


def _window(frame):
    start = frame.index[200]
    end = frame.index[-1] + pd.Timedelta(minutes=15)
    return Window(frame.index[0], start, start, end)


def _cache():
    closes = 100 + 10 * np.sin(np.linspace(0, 6 * np.pi, 400))
    frame = _frame(closes)
    return MemoryBars({("SPY", "15m"): frame}), frame


def test_rule_spec_row_carries_ic_fields():
    cache, frame = _cache()
    row = run_one(cache, "rules", {"spec": SPEC}, "SPY", _window(frame),
                  tfs=TFS, registry=_registry)
    assert row["ic_1"] is not None and row["ic_n_1"] == 199
    assert set(row) >= {"ic_1", "ic_5", "ic_20", "ic_n_1", "ic_n_5", "ic_n_20"}


def test_non_rule_strategy_rows_have_none():
    cache, frame = _cache()
    row = run_one(cache, "inert", {}, "SPY", _window(frame),
                  tfs=TFS, registry=_registry)
    assert row["ic_1"] is None and row["ic_n_1"] == 0


def test_icir_false_skips_computation():
    cache, frame = _cache()
    row = run_one(cache, "rules", {"spec": SPEC}, "SPY", _window(frame),
                  tfs=TFS, registry=_registry, icir=False)
    assert row["ic_1"] is None and row["ic_n_1"] == 0


def test_icir_failure_does_not_kill_the_run_row(monkeypatch):
    # The lens is informational; a bug in it must never take down a
    # production run row. window_icir raising must degrade to the empty
    # fields, not propagate.
    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("nakagai.engine.runner.window_icir", _boom)
    cache, frame = _cache()
    row = run_one(cache, "rules", {"spec": SPEC}, "SPY", _window(frame),
                  tfs=TFS, registry=_registry)
    assert row["ic_1"] is None and row["ic_n_1"] == 0
    assert row["symbol"] == "SPY" and "sharpe" in row
