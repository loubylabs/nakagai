"""MAE/MFE per trade, and the golden that proves adding it moved no trade.

P0's one risk: tracking the excursion means reading bar.high and bar.low while
a position is open, which is the bar loop. Reading MORE from a bar is not the
same as changing fill arithmetic, but the charter's "no re-prove needed" claim
rests on that distinction, so it is proven here rather than asserted. If the
golden below ever moves, this stops being a P0 item and becomes part of the P3
arithmetic epoch, where a re-prove is budgeted.
"""

import numpy as np
import pandas as pd
import pytest

from nakagai.data.cache import BarCache
from nakagai.engine.engine import Engine
from nakagai.engine.portfolio_types import Signal
from nakagai.strategies.base import Direction, Strategy


class Repeater(Strategy):
    """Signals on every bar it is flat, so one replay produces many trades."""

    name = "repeater"
    DEFAULT_PARAMS = {}

    def __init__(self, direction=Direction.LONG, stop_pct=0.02, target_pct=0.03,
                 params=None):
        super().__init__(params)
        self.direction = direction
        self.stop_pct, self.target_pct = stop_pct, target_pct

    def on_bar(self, ctx):
        price = float(ctx.bars["15m"]["close"].iloc[-1])
        if self.direction == Direction.LONG:
            stop, target = price * (1 - self.stop_pct), price * (1 + self.target_pct)
        else:
            stop, target = price * (1 + self.stop_pct), price * (1 - self.target_pct)
        return [Signal(ctx.symbol, self.direction, price, stop=stop, target=target,
                       confidence=1.0, setup_tags=("rep",), rationale="repeater")]


def _walk(n=900, seed=7, base=100.0):
    """A deterministic random walk. Fixed seed: the golden below is literal."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.4, n).cumsum()
    close = base + steps
    idx = pd.date_range("2026-06-01 13:30", periods=n, freq="15min", tz="UTC",
                        name="ts")
    return pd.DataFrame(
        {"open": close - 0.05, "high": close + 0.35, "low": close - 0.35,
         "close": close, "volume": 1_000.0}, index=idx)


@pytest.fixture
def replay(tmp_path):
    def _run(direction=Direction.LONG):
        cache = BarCache(tmp_path / str(direction))
        df = _walk()
        cache.upsert("SPY", "15m", df)
        eng = Engine(Repeater(direction), cache, "SPY", df.index[0],
                     df.index[-1] + pd.Timedelta(minutes=15))
        return eng.run()

    return _run


# -- the golden --------------------------------------------------------------
# Captured from the engine as it stood at 8825849, BEFORE the excursion
# tracking below existed. Every field a trade carried then, to full float
# precision, for both directions. Regenerate ONLY if a change is meant to move
# trades, and if it is, this item does not belong in P0.

def _fingerprint(trades) -> str:
    import hashlib

    body = "\n".join(
        f"{t.symbol}|{t.direction.value}|{t.qty}|{t.entry_ts.isoformat()}|"
        f"{t.entry!r}|{t.exit_ts.isoformat()}|{t.exit!r}|{t.stop!r}|"
        f"{t.target!r}|{t.pnl!r}|{t.r_multiple!r}|{t.setup_tags}|"
        f"{t.exit_reason}|{t.fees!r}"
        for t in trades)
    return hashlib.sha256(body.encode()).hexdigest()


GOLDEN = {
    Direction.LONG: {
        "n_trades": 14,
        "rejected": 642,
        "final_equity": 9205.888716881465,
        "fingerprint":
            "07f276438c29b81f301cc2a1bdaebf9faedd5d897ca33983e8657d5b2773bb60",
    },
    Direction.SHORT: {
        "n_trades": 14,
        "rejected": 517,
        "final_equity": 10263.662053017755,
        "fingerprint":
            "18310b4fc408b93edb9baf6d293ed26bdaec7bbcac8e358fc7697c74b8635565",
    },
}

# The first trade of the long replay, spelled out, so this golden is readable
# and not only checkable. If the fingerprint moves, diff against this first.
GOLDEN_FIRST_LONG = {
    "qty": 48,
    "entry": 100.07999727537403,
    "exit": 97.92451186914512,
    "stop": 98.00048222011614,
    "target": 103.00050682318329,
    "pnl": -103.46329949898745,
    "r_multiple": -1.0365327246748872,
    "exit_reason": "stop",
}


def test_the_replay_produces_enough_trades_to_be_worth_pinning(replay):
    """A golden over one trade would prove almost nothing."""
    assert len(replay().trades) >= 10


@pytest.mark.parametrize("direction", [Direction.LONG, Direction.SHORT])
def test_adding_the_excursion_moved_no_trade(replay, direction):
    """THE test this item's 'no re-prove needed' claim rests on.

    Reading bar.high and bar.low while a position is open is not the same as
    changing fill arithmetic, but it happens inside the bar loop, so the claim
    is proven rather than asserted. Trade count, settlement rejections, final
    equity and every field of every trade, against values captured before the
    excursion code existed."""
    res = replay(direction)
    want = GOLDEN[direction]
    assert len(res.trades) == want["n_trades"]
    assert res.rejected_unsettled == want["rejected"]
    assert float(res.equity_curve.iloc[-1]) == pytest.approx(
        want["final_equity"], abs=1e-9)
    # Every field of every trade, to full float precision. The three
    # assertions above localize a break; this one is the one that cannot be
    # satisfied by a trade set that merely aggregates the same way.
    assert _fingerprint(res.trades) == want["fingerprint"]


def test_the_first_long_trade_is_unchanged_field_by_field(replay):
    """The fingerprint says 'something moved'; this says what."""
    t = replay(Direction.LONG).trades[0]
    got = {"qty": t.qty, "entry": t.entry, "exit": t.exit, "stop": t.stop,
           "target": t.target, "pnl": t.pnl, "r_multiple": t.r_multiple,
           "exit_reason": t.exit_reason}
    assert got == GOLDEN_FIRST_LONG


def test_the_excursion_never_flatters_a_losing_trade(replay):
    """MFE is what the trade had available, MAE what it gave up. A stopped-out
    trade must show an MAE at least as deep as the stop it hit: it reached that
    level by definition, which is why it closed."""
    for t in replay().trades:
        if t.exit_reason != "stop":
            continue
        stop_distance_r = 1.0
        assert t.mae >= stop_distance_r - 1e-9, (
            f"a stop-out cannot have an MAE shallower than 1R, got {t.mae}")


def test_both_excursions_are_signed_the_same_way(replay):
    """Both are magnitudes, never negative. A negative MAE would read as a
    trade that was never once underwater by any amount, which is a different
    claim from 'it was underwater by zero'."""
    for t in replay(Direction.LONG).trades + replay(Direction.SHORT).trades:
        assert t.mae >= 0.0 and t.mfe >= 0.0


def test_a_winning_trade_has_an_mfe_that_reached_its_target(replay):
    for t in replay().trades:
        if t.exit_reason == "target":
            assert t.mfe >= t.r_multiple - 1e-9


def test_the_excursion_is_measured_in_r_not_in_dollars(replay):
    """R rather than price, so the figure is comparable across symbols and
    price levels, which is what makes 'where should the stop have been'
    answerable over a whole catalog rather than one ticker. Price is
    recoverable: mae * abs(entry - stop)."""
    t = next(t for t in replay().trades if t.mae > 0)
    risk_per_share = abs(t.entry - t.stop)
    assert t.mae < 100, "an R multiple, not a dollar excursion"
    assert risk_per_share > 0
