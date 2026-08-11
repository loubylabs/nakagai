import pandas as pd

from nakagai.data.cache import BarCache
from nakagai.engine.runner import run_grid
from nakagai.engine.windows import Window
from nakagai.engine.portfolio_types import Signal
from nakagai.strategies.base import Direction, Strategy


class AlwaysLong(Strategy):
    """Fires one LONG on the very first bar, then stays quiet."""

    name = "alwayslong"
    DEFAULT_PARAMS = {}

    def on_bar(self, ctx):
        if len(ctx.bars["15m"]) != 1:
            return []
        c = float(ctx.bars["15m"]["close"].iloc[-1])
        return [Signal(ctx.symbol, Direction.LONG, c, stop=c - 1.0, target=c + 100.0,
                       confidence=1.0, setup_tags=("test",), rationale="fixture")]


def _registry():
    return {"alwayslong": AlwaysLong}


def seed(root, make_bars):
    cache = BarCache(root)
    cache.upsert("SPY", "15m", make_bars(30, "15m", start="2026-03-02 14:30"))
    return cache


def test_trades_parquet_written_and_linked(tmp_path, make_bars):
    seed(tmp_path / "cache", make_bars)
    out = tmp_path / "runs.parquet"
    w = Window(pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-01", tz="UTC"),
               pd.Timestamp("2026-03-01", tz="UTC"), pd.Timestamp("2026-04-01", tz="UTC"))
    rows = run_grid(str(tmp_path / "cache"), ["alwayslong"], ["SPY"], [w], workers=1, out=str(out),
                    registry=_registry)
    assert "trades" not in rows.columns  # popped before the runs frame is built
    trades_path = tmp_path / "trades.parquet"
    assert trades_path.exists()
    trades = pd.read_parquet(trades_path)
    assert len(trades) >= 1
    assert set(trades["run_id"]) <= set(rows["run_id"])
    assert list(trades.columns) == ["run_id", "symbol", "direction", "qty", "entry_ts", "entry",
                                    "exit_ts", "exit", "stop", "target", "pnl", "r_multiple",
                                    "setup_tags", "exit_reason", "fees", "mae", "mfe"]
    assert trades["direction"].iloc[0] == "long"
    assert trades["setup_tags"].iloc[0] == "test"


def test_the_excursion_survives_the_round_trip(tmp_path, make_bars):
    """MAE/MFE that stops at the dataclass is write-only. The product question
    it answers, where should the stop have been, is asked of a whole catalog of
    stored trades, not of one in-memory replay."""
    seed(tmp_path / "cache", make_bars)
    w = Window(pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-01", tz="UTC"),
               pd.Timestamp("2026-03-01", tz="UTC"), pd.Timestamp("2026-04-01", tz="UTC"))
    run_grid(str(tmp_path / "cache"), ["alwayslong"], ["SPY"], [w], workers=1,
             out=str(tmp_path / "runs.parquet"), registry=_registry)
    trades = pd.read_parquet(tmp_path / "trades.parquet")
    assert (trades["mae"] >= 0).all() and (trades["mfe"] >= 0).all()
    assert trades["mfe"].iloc[0] > 0, "a long that ran to a +100 target moved for us"


def test_trades_append_semantics(tmp_path, make_bars):
    seed(tmp_path / "cache", make_bars)
    out = tmp_path / "runs.parquet"
    w = Window(pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-01", tz="UTC"),
               pd.Timestamp("2026-03-01", tz="UTC"), pd.Timestamp("2026-04-01", tz="UTC"))
    run_grid(str(tmp_path / "cache"), ["alwayslong"], ["SPY"], [w], workers=1, out=str(out),
            registry=_registry)
    n1 = len(pd.read_parquet(tmp_path / "trades.parquet"))
    run_grid(str(tmp_path / "cache"), ["alwayslong"], ["SPY"], [w], workers=1, out=str(out),
            registry=_registry)
    assert len(pd.read_parquet(tmp_path / "trades.parquet")) == 2 * n1
