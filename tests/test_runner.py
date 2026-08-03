import numpy as np
import pandas as pd

from nakagai.data.cache import BarCache
from nakagai.data.schema import TimeframeSet
from nakagai.engine.runner import run_grid
from nakagai.engine.windows import Window
from nakagai.strategies.rules import RuleStrategy, Term, core_vocabulary


TFS = TimeframeSet(driving="15m", deltas={"15m": pd.Timedelta(minutes=15)})
SPEC = {"version": 2, "name": "worker-vocabulary", "timeframe": "15m",
        "long": {"all": [{"lhs": {"ind": "double_close"},
                            "op": ">", "rhs": 0}]}}


def worker_vocabulary():
    return core_vocabulary().with_terms(
        Term("double_close", "series", {}, {}, lambda s, _args: s * 2)
    )


def worker_registry():
    return {"rules": RuleStrategy}


def test_spawned_worker_rebuilds_the_injected_vocabulary(tmp_path):
    cache = BarCache(tmp_path / "cache")
    idx = pd.date_range("2026-01-05 14:30", periods=80, freq="15min", tz="UTC")
    close = pd.Series(100 + np.sin(np.linspace(0, 4, len(idx))), index=idx)
    bars = pd.DataFrame({"open": close, "high": close + 0.5,
                         "low": close - 0.5, "close": close,
                         "volume": 1000.0}, index=idx)
    cache.upsert("SPY", "15m", bars)
    window = Window(idx[0], idx[20], idx[20], idx[-1] + TFS.step)

    rows = run_grid(
        str(tmp_path / "cache"), ["rules"], ["SPY"], [window], workers=2,
        out=str(tmp_path / "runs.parquet"), params_by_strategy={"rules": {"spec": SPEC}},
        tfs=TFS, registry=worker_registry, vocabulary_factory=worker_vocabulary,
    )

    assert len(rows) == 1
    assert rows.iloc[0]["strategy"] == "rules"
