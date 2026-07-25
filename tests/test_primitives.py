import numpy as np
import pandas as pd

from nakagai.strategies.base import MarketContext


def _session_bars():
    """Two NY sessions of 15m bars, deterministic prices."""
    idx = []
    for day in ("2026-01-05", "2026-01-06"):
        idx.extend(pd.date_range(f"{day} 14:30", f"{day} 20:45", freq="15min", tz="UTC"))
    idx = pd.DatetimeIndex(idx)
    close = np.linspace(100, 110, len(idx))
    return pd.DataFrame({"open": close - 0.1, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": 1000.0}, index=idx)


def _ctx(bars):
    return MarketContext(symbol="SPY", now=bars.index[-1] + pd.Timedelta(minutes=15),
                         bars={"15m": bars, "1h": bars, "1d": bars})


def test_opening_range_high_is_per_session_and_constant_after_window():
    from nakagai.strategies.rules.primitives import PRIMITIVES
    bars = _session_bars()
    orh = PRIMITIVES["opening_range_high"]["fn"](_ctx(bars), bars, minutes=30)
    day2 = orh[bars.index.tz_convert("America/New_York").date == pd.Timestamp("2026-01-06").date()]
    first_two = bars.loc[day2.index[:2], "high"].max()
    assert (day2.dropna() == first_two).all()          # constant within the session
    assert pd.isna(orh.iloc[0])                        # NaN until the window completes


def test_prev_session_close_and_gap_pct():
    from nakagai.strategies.rules.primitives import PRIMITIVES
    bars = _session_bars()
    ctx = _ctx(bars)
    day1_close = bars[bars.index.date == pd.Timestamp("2026-01-05").date()]["close"].iloc[-1]
    psc = PRIMITIVES["prev_session_close"]["fn"](ctx, bars)
    assert psc.iloc[-1] == day1_close
    day2_open = bars[bars.index.date == pd.Timestamp("2026-01-06").date()]["open"].iloc[0]
    gap = PRIMITIVES["gap_pct"]["fn"](ctx, bars)
    assert abs(gap.iloc[-1] - 100 * (day2_open - day1_close) / day1_close) < 1e-9


def test_swing_high_returns_last_confirmed_swing_level():
    from nakagai.strategies.rules.primitives import PRIMITIVES
    bars = _session_bars()
    bars.loc[bars.index[20], "high"] = 200.0           # inject an obvious swing
    sh = PRIMITIVES["swing_high"]["fn"](_ctx(bars), bars, k=3)
    assert sh.iloc[-1] == 200.0


def test_time_primitives():
    from nakagai.strategies.rules.primitives import PRIMITIVES
    bars = _session_bars()
    ctx = _ctx(bars)
    dow = PRIMITIVES["day_of_week"]["fn"](ctx, bars)
    assert dow.iloc[0] == 0 and dow.iloc[-1] == 1      # Mon, Tue
    mins = PRIMITIVES["minutes_into_session"]["fn"](ctx, bars)
    assert mins.iloc[0] == 0.0 and mins.iloc[2] == 30.0


def test_bars_since_counts_bars_since_condition_true():
    from nakagai.strategies.rules.primitives import bars_since
    bars = _session_bars()
    mask = pd.Series(False, index=bars.index)
    mask.iloc[-4] = True
    out = bars_since(_ctx(bars), bars, cond={"placeholder": True},
                     eval_fn=lambda cond, bars_: mask)
    assert out.iloc[-1] == 3.0
    assert pd.isna(out.iloc[0])                        # never true yet -> NaN


def test_fvg_nearest_returns_float_or_nan():
    from nakagai.strategies.rules.primitives import PRIMITIVES
    bars = _session_bars()
    v = PRIMITIVES["fvg_nearest"]["fn"](_ctx(bars), bars, direction="long", field="top")
    assert isinstance(v, float)                        # NaN when no unfilled FVG exists


def test_day_of_week_reads_the_utc_calendar_day_on_daily_frames():
    # Session-aligned daily bars are labeled midnight UTC of their session date;
    # converting midnight UTC to NY lands on the prior evening, which used to
    # shift every weekday back by one (a Monday bar read as Sunday).
    from nakagai.strategies.rules.primitives import PRIMITIVES
    idx = pd.date_range("2026-01-05", periods=5, freq="B", tz="UTC")  # Mon..Fri
    bars = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=idx)
    dow = PRIMITIVES["day_of_week"]["fn"](None, bars)
    assert list(dow) == [0.0, 1.0, 2.0, 3.0, 4.0]
