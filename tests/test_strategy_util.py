import pandas as pd

from nakagai.strategies.base import Direction, MarketContext
from nakagai.strategies.util import first_bar_of_session, fresh_bar, rr_signal


def _ctx(now, bars_15m=None, bars_1h=None):
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                         index=pd.DatetimeIndex([], tz="UTC"))
    return MarketContext(symbol="SPY", now=now,
                         bars={"15m": bars_15m if bars_15m is not None else empty,
                               "1h": bars_1h if bars_1h is not None else empty,
                               "1d": empty})


def _bars(ts_list, close=100.0):
    idx = pd.DatetimeIndex(ts_list, tz="UTC")
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1000.0}, index=idx)


def test_fresh_bar_true_only_first_15m_after_hour_close():
    bars_1h = _bars([pd.Timestamp("2026-01-05 15:00", tz="UTC")])
    assert fresh_bar(_ctx(pd.Timestamp("2026-01-05 16:00", tz="UTC"), bars_1h=bars_1h), "1h")
    assert not fresh_bar(_ctx(pd.Timestamp("2026-01-05 16:15", tz="UTC"), bars_1h=bars_1h), "1h")


def test_first_bar_of_session_detects_date_change():
    b = _bars([pd.Timestamp("2026-01-05 20:45", tz="UTC"),
               pd.Timestamp("2026-01-06 14:30", tz="UTC")])
    assert first_bar_of_session(_ctx(b.index[-1], bars_15m=b))


def test_rr_signal_rejects_stop_on_wrong_side():
    b = _bars([pd.Timestamp("2026-01-05 15:00", tz="UTC")], close=100.0)
    ctx = _ctx(b.index[-1], bars_15m=b)
    assert rr_signal(ctx, Direction.LONG, stop=101.0, rr=2.0, tags=("t",), rationale="x") is None
    sig = rr_signal(ctx, Direction.LONG, stop=98.0, rr=2.0, tags=("t",), rationale="x")
    assert sig is not None and sig.target == 104.0
