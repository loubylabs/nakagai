import numpy as np
import pandas as pd
import pytest

from nakagai.data.schema import DEFAULT_TIMEFRAMES, EXCHANGE_TZ, session_open
from nakagai.strategies.base import MarketContext
from nakagai.strategies.rules import (
    PastFiresRefused, past_fires, unhistoried_reads,
)
from nakagai.strategies.rules.frame_eval import FrameEval
from nakagai.strategies.util import first_bar_of_session

DAYS = pd.date_range("2024-01-02", periods=60, freq="B", tz="UTC", name="ts")

LONG_ABOVE = {"version": 2, "timeframe": "1d",
              "long": {"all": [{"lhs": {"src": "close"}, "op": ">", "rhs": 105.0}]}}
ALWAYS_LONG = {"version": 2, "timeframe": "1d",
               "long": {"all": [{"lhs": {"src": "close"}, "op": ">", "rhs": 0.0}]}}


def _bars(close, index=DAYS, volume=1_000.0):
    close = np.asarray(close, dtype="float64")
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": volume}, index=index)


def _frames(aaa, spy=None):
    spy = np.full(len(DAYS), 400.0) if spy is None else spy
    return {("AAA", "1d"): _bars(aaa), ("SPY", "1d"): _bars(spy)}


def _volume_rule(threshold):
    return {"all": [{"lhs": {"src": "volume"}, "op": ">", "rhs": threshold}]}


def test_fires_are_the_bars_the_entry_holds():
    frames = _frames(100.0 + np.arange(60))
    out = past_fires(LONG_ABOVE, frames, "AAA", benchmark="SPY", horizon=5, warmup=0)
    assert out.long.fire_bars == tuple(range(6, 60))
    assert out.short is None and out.bars == 60
    assert (out.symbol, out.timeframe, out.horizon) == ("AAA", "1d", 5)
    assert out.first == DAYS[0].isoformat() and out.last == DAYS[-1].isoformat()


def test_moves_are_against_the_benchmark_over_the_horizon():
    frames = _frames(np.full(60, 100.0), 400.0 * 1.01 ** np.arange(60))
    out = past_fires(ALWAYS_LONG, frames, "AAA", benchmark="SPY", horizon=5, warmup=0)
    assert out.long.fire_moves[0] == pytest.approx(-(1.01 ** 5 - 1))
    assert len(out.long.fire_bars) == 60
    assert len(out.long.fire_moves) == 60 - 5      # fires without a horizon are unscored
    assert len(out.long.baseline_moves) == 60 - 5


def test_a_missing_benchmark_bar_drops_the_move_but_not_the_fire():
    frames = _frames(100.0 + np.arange(60))
    frames[("SPY", "1d")] = frames[("SPY", "1d")].drop(DAYS[10])
    out = past_fires(ALWAYS_LONG, frames, "AAA", benchmark="SPY", horizon=5, warmup=0)
    assert 10 in out.long.fire_bars
    # Row 10 has no benchmark start, and row 5 has no benchmark end.
    assert len(out.long.fire_moves) == 60 - 5 - 2
    assert len(out.long.baseline_moves) == 60 - 5 - 2


def test_a_short_side_scores_the_negated_move_and_yields_to_long():
    volume = np.full(60, 1_000.0)
    volume[10], volume[20] = 2_000.0, 1_300.0
    frames = _frames(100.0 + np.arange(60), 400.0 * 1.003 ** np.arange(60))
    frames[("AAA", "1d")]["volume"] = volume
    spec = {"version": 2, "timeframe": "1d",
            "long": _volume_rule(1_500.0), "short": _volume_rule(1_200.0)}
    long_only = {"version": 2, "timeframe": "1d", "long": _volume_rule(1_500.0)}
    out = past_fires(spec, frames, "AAA", benchmark="SPY", horizon=5, warmup=0)
    out_long_only = past_fires(long_only, frames, "AAA", benchmark="SPY",
                               horizon=5, warmup=0)
    assert out.long.fire_bars == (10,)
    assert out.short.fire_bars == (20,)
    assert 10 not in out.short.fire_bars and 20 in out.short.fire_bars
    assert out.short.fire_moves == (-out_long_only.long.baseline_moves[20],)
    assert out.short.baseline_moves == tuple(
        -m for m in out_long_only.long.baseline_moves)
    assert out_long_only.short is None


def test_warmup_rows_are_neither_fires_nor_baseline():
    frames = _frames(100.0 + np.arange(60))
    out = past_fires(ALWAYS_LONG, frames, "AAA", benchmark="SPY", horizon=5, warmup=20)
    assert min(out.long.fire_bars) == 20 and out.bars == 40
    assert out.first == DAYS[20].isoformat()
    assert len(out.long.baseline_moves) == 60 - 20 - 5


def test_a_reference_symbol_is_read_on_the_traded_index():
    # Intraday, because FrameEval refuses any reference on a session-aligned
    # spec frame.
    hours = pd.date_range("2024-01-02 14:00", periods=60, freq="1h", tz="UTC",
                          name="ts")
    frames = {("AAA", "1h"): _bars(100.0 + np.arange(60), index=hours),
              ("SPY", "1h"): _bars(np.full(60, 400.0), index=hours)}
    qqq = _bars(np.full(60, 300.0), index=hours).drop(hours[30])
    # A reference bar the traded symbol never printed is not read.
    extra = _bars([300.0], index=pd.DatetimeIndex(
        [hours[30] + pd.Timedelta(minutes=30)], name="ts"))
    frames[("QQQ", "1h")] = pd.concat([qqq, extra]).sort_index()
    spec = {"version": 2, "timeframe": "1h", "long": {"all": [
        {"lhs": {"src": "close", "sym": "QQQ"}, "op": ">", "rhs": 200.0}]}}
    out = past_fires(spec, frames, "AAA", benchmark="SPY", horizon=5, warmup=0)
    assert out.long.fire_bars == tuple(i for i in range(60) if i != 30)


def test_news_and_fact_reads_are_refused(count_where_vocab):
    frames = _frames(100.0 + np.arange(60))
    spec = {**LONG_ABOVE, "long": {"all": [
        {"lhs": {"fact": "market_cap"}, "op": ">", "rhs": 1.0}]}}
    assert unhistoried_reads(spec, ()) == ("fact:market_cap",)
    with pytest.raises(PastFiresRefused) as e:
        past_fires(spec, frames, "AAA", benchmark="SPY", horizon=5, warmup=0)
    assert e.value.code == "unhistoried"

    counted = {**LONG_ABOVE, "long": {"all": [
        {"lhs": {"prim": "count_where", "when": LONG_ABOVE["long"]["all"][0],
                 "n": 5},
         "op": ">", "rhs": 1.0},
        {"lhs": {"ind": "sma", "n": 5}, "op": ">", "rhs": 1.0}]}}
    assert unhistoried_reads(counted, ()) == ()
    assert unhistoried_reads(counted, {"count_where", "sma"}) == (
        "count_where", "sma")
    with pytest.raises(PastFiresRefused) as e:
        past_fires(counted, frames, "AAA", benchmark="SPY", horizon=5, warmup=0,
                   vocabulary=count_where_vocab, refused_terms={"count_where"})
    assert e.value.code == "unhistoried"
    # Not refused, the same rule evaluates.
    out = past_fires(counted, frames, "AAA", benchmark="SPY", horizon=5,
                     warmup=0, vocabulary=count_where_vocab)
    assert out.long.fire_bars


def test_a_composite_is_refused():
    frames = _frames(100.0 + np.arange(60))
    composite = {"version": 2, "timeframe": "1d",
                 "blocks": [{"play": "a"}, {"play": "b"}]}
    with pytest.raises(PastFiresRefused, match="composite") as e:
        past_fires(composite, frames, "AAA", benchmark="SPY", horizon=5, warmup=0)
    assert e.value.code == "composite"


def test_too_short_a_history_is_refused():
    frames = {key: frame.iloc[:10] for key, frame in
              _frames(100.0 + np.arange(60)).items()}
    with pytest.raises(PastFiresRefused) as e:
        past_fires(LONG_ABOVE, frames, "AAA", benchmark="SPY", horizon=5, warmup=5)
    assert e.value.code == "insufficient_history"
    # Two more bars than warmup and horizon need is enough.
    frames = {key: frame.iloc[:12] for key, frame in
              _frames(100.0 + np.arange(60)).items()}
    out = past_fires(LONG_ABOVE, frames, "AAA", benchmark="SPY", horizon=5, warmup=5)
    assert out.bars == 7


def _fifteen_over(days: pd.DatetimeIndex) -> pd.DataFrame:
    """26 regular-session 15m bars per session, opening on each day's 09:30
    New York wherever daylight saving puts it in UTC."""
    stamps = []
    for day in days:
        opens = session_open(pd.Timestamp(f"{day.date()} 12:00", tz=EXCHANGE_TZ))
        stamps.extend(opens + pd.Timedelta(minutes=15 * k) for k in range(26))
    index = pd.DatetimeIndex(stamps, name="ts")
    return _bars(np.full(len(index), 100.0), index=index)


def test_fire_bars_match_the_live_gate():
    close = 100.0 + 10.0 * (np.arange(60) % 3 == 0)    # fires every third day
    spec = {"version": 2, "timeframe": "1d", "long": {"all": [
        {"lhs": {"src": "close"}, "op": ">", "rhs": 105.0}]}}
    daily = _bars(close)
    fifteen = _fifteen_over(DAYS)
    frames = {("AAA", "1d"): daily, ("AAA", "15m"): fifteen,
              ("SPY", "1d"): _bars(np.full(60, 400.0))}
    out = past_fires(spec, frames, "AAA", benchmark="SPY", horizon=5, warmup=0)
    assert out.long.fire_bars == tuple(range(0, 60, 3))

    fe = FrameEval("AAA", {k: v for k, v in frames.items() if k[0] == "AAA"},
                   DEFAULT_TIMEFRAMES)
    held = fe.driving_group(spec["long"], "1d")
    gated = [
        j for j in range(len(fifteen))
        if held.iloc[j] and first_bar_of_session(MarketContext(
            symbol="AAA", now=fifteen.index[j],
            bars={"15m": fifteen.iloc[:j + 1]}))
    ]
    session = fifteen.index.tz_convert(EXCHANGE_TZ).date
    # The last session has no next session inside the 15m frame.
    scored = [d for d in out.long.fire_bars if d + 1 < len(DAYS)]
    assert len(gated) == len(scored)
    for d, j in zip(scored, gated):
        assert session[j] == DAYS[d + 1].date()
