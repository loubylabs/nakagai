import pandas as pd
import pytest

from nakagai.data.cache import BarCache
from nakagai.engine.engine import Engine
from nakagai.strategies.base import Direction, Signal, Strategy


def put_bars(cache, rows, start="2026-06-01 13:30"):
    idx = pd.date_range(start, periods=len(rows), freq="15min", tz="UTC", name="ts")
    df = pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows], "volume": 1000.0},
        index=idx,
    )
    cache.upsert("SPY", "15m", df)
    return df


class OneShot(Strategy):
    """Emits one fixed signal on the first bar it sees, then goes quiet."""

    name = "oneshot"
    DEFAULT_PARAMS = {}

    def __init__(self, signal, params=None):
        super().__init__(params)
        self._signal = signal
        self._fired = False

    def on_bar(self, ctx):
        if self._fired:
            return []
        self._fired = True
        return [self._signal]


def run(cache, sig, rows, **kw):
    df = put_bars(cache, rows)
    eng = Engine(OneShot(sig), cache, "SPY", df.index[0], df.index[-1] + pd.Timedelta(minutes=15), **kw)
    return eng.run()


def test_long_target_hit(tmp_path):
    cache = BarCache(tmp_path)
    sig = Signal("SPY", Direction.LONG, None, stop=98.0, target=104.0, confidence=1.0, setup_tags=("t",), rationale="")
    rows = [
        (100, 100.5, 99.5, 100),   # bar0: signal fires here
        (100, 100.5, 99.5, 100),   # bar1: entry fills at open 100 (+.01 slip)
        (100, 105.0, 99.9, 104.5), # bar2: target 104 hit
    ]
    res = run(cache, sig, rows)
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.entry == pytest.approx(100.01)
    assert t.exit == pytest.approx(103.99)  # target - slippage
    assert t.exit_reason == "target"
    assert t.qty == 49  # floor(100 / (100.01-98))
    assert t.pnl == pytest.approx((103.99 - 100.01) * 49)


def test_stop_first_when_both_in_bar(tmp_path):
    cache = BarCache(tmp_path)
    sig = Signal("SPY", Direction.LONG, None, stop=98.0, target=104.0, confidence=1.0, setup_tags=("t",), rationale="")
    rows = [
        (100, 100.5, 99.5, 100),
        (100, 100.5, 99.5, 100),
        (100, 105.0, 97.0, 104.0),  # bar spans BOTH stop and target -> stop wins
    ]
    res = run(cache, sig, rows)
    assert res.trades[0].exit_reason == "stop"
    assert res.trades[0].exit == pytest.approx(97.99)  # stop - slippage


def test_short_symmetric(tmp_path):
    cache = BarCache(tmp_path)
    sig = Signal("SPY", Direction.SHORT, None, stop=102.0, target=96.0, confidence=1.0, setup_tags=("t",), rationale="")
    rows = [
        (100, 100.5, 99.5, 100),
        (100, 100.5, 99.5, 100),   # entry 100 - .01 slip = 99.99
        (99, 99.5, 95.5, 96.0),    # target 96 hit
    ]
    res = run(cache, sig, rows)
    t = res.trades[0]
    assert t.direction == Direction.SHORT
    assert t.entry == pytest.approx(99.99)
    assert t.exit == pytest.approx(96.01)
    assert t.pnl == pytest.approx((99.99 - 96.01) * t.qty)
    assert t.pnl > 0


def test_unsettled_rejection_counted(tmp_path):
    cache = BarCache(tmp_path)
    sig = Signal("SPY", Direction.LONG, None, stop=99.9, target=104.0, confidence=1.0, setup_tags=("t",), rationale="")
    # tight stop -> huge qty -> cost far exceeds equity0 -> ledger reject
    rows = [(100, 100.5, 99.95, 100)] * 3
    res = run(cache, sig, rows, equity0=1000.0)
    assert res.trades == []
    assert res.rejected_unsettled == 1


def test_open_position_closed_at_window_end(tmp_path):
    cache = BarCache(tmp_path)
    sig = Signal("SPY", Direction.LONG, None, stop=90.0, target=200.0, confidence=1.0, setup_tags=("t",), rationale="")
    rows = [(100, 100.5, 99.5, 100)] * 4  # never hits stop or target
    res = run(cache, sig, rows)
    assert len(res.trades) == 1
    assert res.trades[0].exit_reason == "eod_window"
