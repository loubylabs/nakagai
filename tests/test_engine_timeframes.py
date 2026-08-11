"""The replay cadence follows TimeframeSet.driving, not a hardcoded 15m."""

import pandas as pd

from nakagai.data.cache import BarCache
from nakagai.data.schema import TimeframeSet
from nakagai.engine.engine import Engine
from nakagai.engine.portfolio_types import Signal
from nakagai.strategies.base import Direction, MarketContext, Strategy

H1 = TimeframeSet(driving="1h", higher=(),
                  deltas={"1h": pd.Timedelta(hours=1)})


class FirstBarLong(Strategy):
    name = "firstbarlong"

    def on_bar(self, ctx: MarketContext) -> tuple[Signal, ...]:
        if len(ctx.driving_bars) == 1:
            close = float(ctx.driving_bars["close"].iloc[-1])
            return [Signal(symbol=ctx.symbol, direction=Direction.LONG,
                           entry_ref=close, stop=close - 5.0,
                           target=close + 100.0, confidence=0.5,
                           setup_tags=("t",), rationale="first bar long")]
        return ()


def _hourly_bars(n: int = 8) -> pd.DataFrame:
    idx = pd.date_range("2026-01-05 14:00", periods=n, freq="1h", tz="UTC")
    base = pd.Series(range(n), index=idx, dtype="float64") + 100.0
    return pd.DataFrame({"open": base, "high": base + 1.0, "low": base - 1.0,
                         "close": base + 0.5, "volume": 1000.0}, index=idx)


def test_engine_replays_on_1h_driving_timeframe(tmp_path):
    cache = BarCache(tmp_path / "cache")
    bars = _hourly_bars()
    cache.upsert("SPY", "1h", bars)
    engine = Engine(FirstBarLong(), cache, "SPY",
                    bars.index[0], bars.index[-1] + pd.Timedelta(hours=1),
                    tfs=H1)
    result = engine.run()
    curve = result.equity_curve
    # marks land at each 1h bar close, one hour apart
    steps = curve.index.to_series().diff().dropna().unique()
    assert list(steps) == [pd.Timedelta(hours=1)]
    # the first-bar signal fills at the SECOND bar's open (next-bar-open rule)
    assert len(result.trades) == 1
    assert result.trades[0].entry_ts == bars.index[1]
