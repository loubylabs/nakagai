import numpy as np
import pandas as pd
import pytest

from nakagai.data.cache import BarCache
from nakagai.data.schema import TimeframeSet
from nakagai.engine.provenance import ARITHMETIC_VERSION, FILL_MODE
from nakagai.engine.runner import run_grid, run_one
from nakagai.engine.windows import Window
from nakagai.strategies.base import Strategy
from nakagai.strategies.rules import RuleStrategy, Term, core_vocabulary


TFS = TimeframeSet(driving="15m", deltas={"15m": pd.Timedelta(minutes=15)})
SPEC = {"version": 2, "name": "worker-vocabulary", "timeframe": "15m",
        "long": {"all": [{"lhs": {"ind": "double_close"},
                            "op": ">", "rhs": 0}]}}


def worker_vocabulary():
    return core_vocabulary().with_terms(
        Term("double_close", "series", {}, {}, lambda s, _args: s * 2)
    )


class _Inert(Strategy):
    name = "inert"
    DEFAULT_PARAMS = {}

    def on_bar(self, ctx):
        return []


def worker_registry():
    return {"rules": RuleStrategy, "inert": _Inert}


def _cache(tmp_path):
    cache = BarCache(tmp_path / "cache")
    idx = pd.date_range("2026-01-05 14:30", periods=80, freq="15min", tz="UTC")
    close = pd.Series(100 + np.sin(np.linspace(0, 4, len(idx))), index=idx)
    bars = pd.DataFrame({"open": close, "high": close + 0.5,
                         "low": close - 0.5, "close": close,
                         "volume": 1000.0}, index=idx)
    cache.upsert("SPY", "15m", bars)
    return cache, Window(idx[0], idx[20], idx[20], idx[-1] + TFS.step)


def test_spawned_worker_rebuilds_the_injected_vocabulary(tmp_path):
    _, window = _cache(tmp_path)

    rows = run_grid(
        str(tmp_path / "cache"), ["rules"], ["SPY"], [window], workers=2,
        out=str(tmp_path / "runs.parquet"), params_by_strategy={"rules": {"spec": SPEC}},
        tfs=TFS, registry=worker_registry, vocabulary_factory=worker_vocabulary,
    )

    assert len(rows) == 1
    assert rows.iloc[0]["strategy"] == "rules"
    assert rows.iloc[0]["arithmetic_version"] == ARITHMETIC_VERSION == "1"
    assert rows.iloc[0]["fill_mode"] == FILL_MODE == "pessimistic"

    persisted = pd.read_parquet(tmp_path / "runs.parquet")
    assert persisted.iloc[0]["arithmetic_version"] == "1"
    assert persisted.iloc[0]["fill_mode"] == "pessimistic"


def test_a_vocabulary_factory_for_a_non_rule_strategy_is_refused(tmp_path):
    """A factory the runner cannot honor is an error, never a silent drop.

    A plain strategy gets core_vocabulary() from Engine by design, because its
    context may still host composite RuleSpec members. Accepting the factory
    and ignoring it would let a caller believe a whole grid ran on an injected
    vocabulary when part of it ran on core, which is the exact silent no-op
    this seam exists to prevent.
    """
    cache, window = _cache(tmp_path)
    with pytest.raises(ValueError, match="not a RuleStrategy"):
        run_one(str(tmp_path / "cache"), "inert", {}, "SPY", window, tfs=TFS,
                registry=worker_registry, vocabulary_factory=worker_vocabulary)
    # Without a factory the same strategy still runs: the refusal is about the
    # caller's unmet expectation, not about plain strategies.
    row = run_one(str(tmp_path / "cache"), "inert", {}, "SPY", window, tfs=TFS,
                  registry=worker_registry)
    assert row["strategy"] == "inert"
    assert row["arithmetic_version"] == "1"
    assert row["fill_mode"] == "pessimistic"


def test_run_one_injects_the_vocabulary_without_minting_a_subclass(tmp_path, monkeypatch):
    """bound() belongs to load_catalog, where the class IS the product.

    Per job it mints a throwaway subclass that differs in nothing but a factory
    reference, thousands of them across a grid, when RuleStrategy.__init__
    already accepts a vocabulary.
    """
    def _refuse(_factory):
        raise AssertionError("run_one must not mint a bound subclass per job")

    monkeypatch.setattr(RuleStrategy, "bound", classmethod(
        lambda cls, factory: _refuse(factory)))
    _, window = _cache(tmp_path)
    row = run_one(str(tmp_path / "cache"), "rules", {"spec": SPEC}, "SPY",
                  window, tfs=TFS, registry=worker_registry,
                  vocabulary_factory=worker_vocabulary)
    assert row["strategy"] == "rules" and row["trades"]


def test_grid_append_preserves_legacy_rows_as_unversioned(tmp_path):
    _, window = _cache(tmp_path)
    out = tmp_path / "runs.parquet"
    pd.DataFrame([{"run_id": "legacy", "strategy": "inert"}]).to_parquet(out)

    rows = run_grid(
        str(tmp_path / "cache"), ["inert"], ["SPY"], [window], workers=1,
        out=str(out), tfs=TFS, registry=worker_registry,
    )

    assert rows.iloc[0]["arithmetic_version"] == "1"
    assert rows.iloc[0]["fill_mode"] == "pessimistic"
    persisted = pd.read_parquet(out).set_index("run_id")
    assert pd.isna(persisted.loc["legacy", "arithmetic_version"])
    assert pd.isna(persisted.loc["legacy", "fill_mode"])
    assert persisted.loc[rows.iloc[0]["run_id"], "arithmetic_version"] == "1"
    assert persisted.loc[rows.iloc[0]["run_id"], "fill_mode"] == "pessimistic"
