"""Hard rule 3's core half: every existing catalog spec and fixture produces
byte-identical trades before and after node 03.

The frame is built once, seeded, session-shaped, and pinned here rather than
generated per test run: "deterministic and committed" is what a fixed seed in
the test file means, and a frame randomized per run would make this test's own
result irreproducible.

A comparison that compares nothing is the failure mode this guards against: a
spec producing zero trades before and after passes an equality assertion while
proving nothing. Every one of the nine specs below is asserted to produce at
least one trade on this frame; none needed a named exemption.

The digest is over EXACT values. Prices go in through repr(), which round-trips
a float bit-for-bit, and the sha256 is not truncated, because the requirement
is byte-identical trades: a rounded price and a shortened digest both admit a
change this test exists to refuse.
"""

import dataclasses
import hashlib
import json
import tempfile
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nakagai.data.cache import BarCache
from nakagai.data.resample import resample_bars
from nakagai.data.schema import DEFAULT_TIMEFRAMES as TFS
from nakagai.engine.engine import Engine, Trade
from nakagai.strategies.catalog import load_catalog
from nakagai.strategies.rules import RuleStrategy, core_vocabulary

CATALOG_SPECS = (Path(__file__).resolve().parents[1]
                 / "nakagai" / "strategies" / "catalog" / "specs")
FIXTURE_SPECS = Path(__file__).resolve().parent / "fixtures" / "rules"

SESSIONS = 260  # clears bollinger_breakout's sma(200) on daily bars


def _index(days, bars_per_session, start_hm=(9, 30), step_minutes=15):
    stamps = [d + pd.Timedelta(hours=start_hm[0], minutes=start_hm[1])
              + i * pd.Timedelta(minutes=step_minutes)
              for d in days for i in range(bars_per_session)]
    idx = pd.DatetimeIndex(stamps).tz_convert("UTC")
    idx.name = "ts"
    return idx


def _ohlcv(idx, seed, trend_amp=3.0, vol_spike_every=40, displacement_every=14,
           cycles=6):
    """Session-shaped bars with volume dispersion (clears rvol > 3), a trend
    that reverses (fires bars_since / turnaround-style users), and periodic
    displacement moves sharp enough to leave real 3-candle FVGs and order
    blocks (both need a genuine gap between non-adjacent candles, which a
    smooth random walk essentially never produces on its own)."""
    rng = np.random.default_rng(seed)
    n = len(idx)
    trend = np.sin(np.linspace(0, cycles * np.pi, n)) * trend_amp
    steps = rng.normal(0, 0.15, n)
    disp_rows = np.arange(displacement_every // 2, n, displacement_every)
    for row in disp_rows:
        direction = 1.0 if (row // displacement_every) % 2 == 0 else -1.0
        for k in range(min(3, n - row)):
            steps[row + k] += direction * rng.uniform(0.8, 1.4)
    close = 100 + trend + np.cumsum(steps)
    open_ = np.concatenate([[close[0] - 0.05], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.2, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.2, n))
    base_vol = 800 + rng.integers(0, 400, n).astype(float)
    spike_rows = rng.choice(n, size=max(1, n // vol_spike_every), replace=False)
    base_vol[spike_rows] *= rng.uniform(4, 8, size=len(spike_rows))
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": base_vol}, index=idx)


def build_cache():
    """The frame, in one place, so the derivation harness and the tests cannot
    build different ones. Returns (BarCache, start, end)."""
    c = BarCache(Path(tempfile.mkdtemp()))
    days = pd.bdate_range("2026-01-05", periods=SESSIONS, tz="America/New_York")
    m15 = _ohlcv(_index(days, 26, step_minutes=15), seed=7)
    h1 = _ohlcv(_index(days, 7, step_minutes=60), seed=11, trend_amp=4.0)
    d1 = _ohlcv(_index(days, 1, start_hm=(9, 30), step_minutes=0), seed=13,
                trend_amp=25.0, vol_spike_every=8, displacement_every=6,
                cycles=2)
    c.upsert("SPY", "15m", m15)
    c.upsert("SPY", "1h", h1)
    c.upsert("SPY", "1d", d1)
    c.upsert("SPY", "4h", resample_bars(h1, "4h"))
    return c, m15.index[0], m15.index[-1] + TFS.step


@pytest.fixture(scope="module")
def cache():
    return build_cache()


def _cell(v):
    """Deterministic, lossless, and total over Trade's field types.

    repr() on floats rather than str(), because str() rounds and a digest
    that rounds cannot see a change below the rounding. Enums by name, so
    the digest does not depend on member order. Sequences elementwise, for
    setup_tags.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, Enum):
        return v.name
    if isinstance(v, (tuple, list)):
        return [_cell(x) for x in v]
    return str(v)


def _trade_digest(trades):
    """Every field on Trade, discovered rather than listed.

    An earlier draft hand-listed seven fields, which left stop, target, qty,
    symbol, r_multiple, setup_tags, fees, mae and mfe uncompared: a trade
    whose target moved would have produced an identical digest, so hard rule
    3's "byte-identical trades" would have been proven over a subset while
    claiming the whole. Reading dataclasses.fields() also means a field added
    to Trade later joins the digest with nobody remembering to come back here.
    """
    names = [f.name for f in dataclasses.fields(Trade)]
    rows = [{n: _cell(getattr(t, n)) for n in names} for t in trades]
    blob = json.dumps(rows, sort_keys=True)
    return len(rows), hashlib.sha256(blob.encode()).hexdigest()


def _perturb(v):
    """A different value of the same type, for every type Trade holds.

    Total by construction rather than by a chain of isinstance checks that
    silently returns the input for an unhandled type: a _perturb that handed
    back an unchanged value would make the field's assertion below fail for
    the wrong reason, and the reason matters, since the whole point is to
    distinguish "not in the digest" from "not actually changed".
    """
    if isinstance(v, bool):
        return not v
    if isinstance(v, Enum):
        others = [m for m in type(v) if m is not v]
        assert others, f"{type(v).__name__} has one member; cannot perturb"
        return others[0]
    if isinstance(v, float):
        return v + 1.5
    if isinstance(v, pd.Timestamp):
        return v + pd.Timedelta(minutes=1)
    if isinstance(v, int):
        return v + 1
    if isinstance(v, str):
        return v + "_x"
    if isinstance(v, tuple):
        return v + ("x",)
    raise AssertionError(f"_perturb has no case for {type(v).__name__}; add one")


def _specs():
    out = {}
    for name, cls in load_catalog(CATALOG_SPECS, core_vocabulary).items():
        out[f"catalog:{name}"] = cls({})
    for path in sorted(FIXTURE_SPECS.glob("*.json")):
        spec = json.loads(path.read_text())
        out[f"fixture:{path.stem}"] = RuleStrategy({"spec": spec})
    return out


def _replay(name, cache):
    """One spec's trades. The single replay path, shared by the golden table
    and by the per-field digest guard, so the two cannot disagree about how a
    spec is run."""
    bar_cache, start, end = cache
    return Engine(_specs()[name], bar_cache, "SPY", start, end).run().trades


def test_the_digest_notices_a_change_in_every_trade_field(cache):
    """The digest's discriminating power, field by field.

    This is the guard that would have caught the seven-field draft. It fails
    with every offending field named, so a future narrowing of _trade_digest
    cannot pass quietly.

    Every field is checked before anything is asserted, rather than asserting
    inside the loop: an in-loop assert stops at the FIRST field that misses
    and reports only that one, which reads as a single oversight where the
    real fault is a whole class of fields left uncompared. Watched failing by
    hand-listing the seven-field draft again, and it reported all nine:
    ['symbol', 'qty', 'stop', 'target', 'r_multiple', 'setup_tags', 'fees',
    'mae', 'mfe'].
    """
    assert GOLDEN, "GOLDEN is empty: derive it before this test means anything"
    trades = _replay(sorted(GOLDEN)[0], cache)
    assert trades, "need at least one real trade to perturb"
    base = trades[0]
    d0 = _trade_digest([base])[1]
    missed = []
    for f in dataclasses.fields(Trade):
        before = getattr(base, f.name)
        bumped = dataclasses.replace(base, **{f.name: _perturb(before)})
        if _trade_digest([bumped])[1] == d0:
            missed.append(f.name)
    assert not missed, (
        f"these fields do not reach the digest, so a change to them is "
        f"invisible: {missed}")


# DERIVED, NOT TRANSCRIBED. Printed by running this module's build_cache() and
# _trade_digest() over _specs(), then re-printed with the node 03 production
# diff stashed out of nakagai/ and confirmed identical. That second run is what
# makes the table a measurement of "this node moved no trade" rather than a
# stamp of "this node produced these numbers".
#
# Trade counts survive independently: they match the counts recorded before the
# digest was widened to every Trade field, and the digest does not affect what
# the engine does. If a COUNT moves, that is a real finding.
GOLDEN = {
    "catalog:macd_trend": (123, "1b61d3909e02791c4dbb1482e086bdc494b85716fe449af5dfaf6a9142021cd0"),
    "catalog:rsi_reversion": (11, "6cc3472eac6433035a7c3faeae12582d24ea1b3804d5c6709637cb66e1c95cd3"),
    "catalog:sma_cross": (66, "f2d7eeb8d53e12fe8543406622d1cb9b021a14ca78986e58afee7aa17bd46bf6"),
    "fixture:bollinger_breakout": (1, "42d438c1c25614d86a3eb639bd0f7966c17e4926ff98f479eee9bcbbdf182cd5"),
    "fixture:discount_pullback": (33, "c862cb01391d4e3c8254d7ba94a13179d06886b94c0fa1507722881683b8986e"),
    "fixture:ifvg_reversal": (92, "27317685494d80d5bdeb092d318c0960d4b6ff75c45390de997ec0b5103af903"),
    "fixture:ob_bounce": (79, "b47ab8ca093f04cec8491597a997188c4954c8de6c1a30a896f0194a7aa1f4e6"),
    "fixture:orb": (228, "537a8be31cdcd09353432d631de9a8675eb70ca84aa467e495b48dbff18e2602"),
    "fixture:sma_cross": (66, "f2d7eeb8d53e12fe8543406622d1cb9b021a14ca78986e58afee7aa17bd46bf6"),
}


def test_the_frame_covers_every_spec_in_the_golden_table():
    """Also the floor that keeps an underived GOLDEN from reading green.

    test_every_spec_produces_the_golden_trades is parametrized over
    sorted(GOLDEN), so an empty table collects ZERO cases and pytest reports
    a pass having run nothing. This test is not parametrized, so it is the
    one that fails when the table has not been derived yet.
    """
    assert len(GOLDEN) == 9, f"GOLDEN has {len(GOLDEN)} entries, expected 9"
    assert set(_specs()) == set(GOLDEN)


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_every_spec_produces_the_golden_trades(name, cache):
    bar_cache, start, end = cache
    strat = _specs()[name]
    result = Engine(strat, bar_cache, "SPY", start, end).run()
    n, digest = _trade_digest(result.trades)
    want_n, want_digest = GOLDEN[name]
    assert n >= 1, f"{name}: zero trades proves nothing"
    assert n == want_n, f"{name}: trade count moved from {want_n} to {n}"
    assert digest == want_digest, f"{name}: trades changed (n={n})"
