"""Golden fills: what the engine charges and where it fills.

Pillar 4 (Replay). Every number in the platform's evidence store is downstream
of these four functions, so they are pinned by hand-authored arithmetic rather
than by anything computed. A fill test whose expected value is itself computed
proves only that two copies of the same arithmetic agree.

Companion suite: the platform repo's tests/waterfall/test_stage4_replay.py runs
the same three bar shapes end to end. This file is the unit-level statement.
"""

import pandas as pd
import pytest

from nakagai.data.cache import MemoryBars
from nakagai.data.schema import TimeframeSet
from nakagai.engine.costs import FeeModel, SlippageModel
from nakagai.engine.engine import Engine
from nakagai.engine.portfolio import SettledLedger
from nakagai.strategies.base import Direction, Signal, Strategy

SYMBOL = "GAPX"
STOP, TARGET = 99.0, 104.0
ENTRY_OPEN = 100.2
FLAT = TimeframeSet(driving="15m", deltas={"15m": pd.Timedelta(minutes=15)})

# Four bars, identical except for bar 2, the event bar:
#   0 quiet   the signal fires at this close
#   1 quiet   the pending entry fills at this OPEN (100.2)
#   2 EVENT   the case under test
#   3 tail    so bar 2 has somewhere to close into
_HEAD = [(100.0, 100.4, 99.8, 100.2), (100.2, 100.6, 99.9, 100.4)]
_TAIL = (100.0, 100.3, 99.7, 100.1)


def _bars(event):
    idx = pd.date_range("2025-03-03 14:30", periods=4, freq="15min", tz="UTC")
    df = pd.DataFrame([*_HEAD, event, _TAIL],
                      columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 100_000.0
    return df.astype("float64")


class OneShotLong(Strategy):
    name = "oneshot_long"

    def on_bar(self, ctx):
        if len(ctx.driving_bars) != 1:
            return []
        return [Signal(symbol=ctx.symbol, direction=Direction.LONG, entry=None,
                       stop=STOP, target=TARGET, confidence=1.0,
                       setup_tags=("golden",), rationale="fill fixture")]


class OneShotShort(Strategy):
    name = "oneshot_short"

    def on_bar(self, ctx):
        if len(ctx.driving_bars) != 1:
            return []
        # Mirror image: stop above, target below.
        return [Signal(symbol=ctx.symbol, direction=Direction.SHORT, entry=None,
                       stop=TARGET, target=STOP, confidence=1.0,
                       setup_tags=("golden",), rationale="fill fixture")]


def _run(bars, strategy=None, **kw):
    engine = Engine(strategy or OneShotLong(), MemoryBars({(SYMBOL, "15m"): bars}),
                    SYMBOL, bars.index[0], bars.index[-1] + pd.Timedelta(minutes=15),
                    tfs=FLAT, **kw)
    return engine.run()


# ---------------------------------------------------------------------------
# The slippage model, on its own.

def test_slippage_floor_binds_below_a_hundred_dollars():
    """1bp of $96 is $0.0096, under the one-cent floor, so the floor wins.
    Slippage does not shrink to nothing on cheap stock: the tick is a cent."""
    assert SlippageModel().per_share(96.0) == pytest.approx(0.01)


def test_slippage_scales_above_the_floor():
    """1bp of $106 is $0.0106, over the floor, so the proportional term wins."""
    assert SlippageModel().per_share(106.0) == pytest.approx(0.0106)


def test_slippage_is_a_cent_at_a_hundred_dollars():
    """The crossover, and the old flat-cent constant. This is the one price at
    which H1 changes nothing, which is what makes every other price a
    deliberate move rather than an accident."""
    assert SlippageModel().per_share(100.0) == pytest.approx(0.01)


def test_a_five_hundred_dollar_stock_is_charged_five_cents():
    """The reason a flat cent had to go: it was 1bp here and 0.02bp there, so
    the same modeled friction was an order of magnitude harsher on cheap names
    and the stored numbers were not comparable across the universe."""
    assert SlippageModel().per_share(500.0) == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Fills.

def test_entry_fills_at_the_next_bar_open_plus_slippage():
    """Signals are acted on at the next bar's open, never at the close that
    produced them. 1bp of 100.2 is 0.01002, just over the floor."""
    trade = _run(_bars(_TAIL)).trades[0]
    assert trade.entry == pytest.approx(ENTRY_OPEN + 0.01002)


def test_a_gap_down_through_the_stop_fills_at_the_open():
    """Bar 2 opens at 96.0, below the 99.0 stop. The stop was never available:
    96.0 is the first price at which this position could have traded.

    Filling at 99.0, which is what the engine did before H1, books a loss the
    desk never had access to and models overnight gap risk as zero.
    """
    trade = _run(_bars((96.0, 96.4, 95.5, 96.2))).trades[0]
    assert trade.exit_reason == "stop"
    assert trade.exit == pytest.approx(96.0 - 0.01)   # floor binds at $96


def test_a_gap_up_through_the_target_fills_at_the_open():
    """Bar 2 opens at 106.0, above the 104.0 target. A resting sell limit fills
    at the open when the market gaps past it, so this direction of the same bug
    was understating wins."""
    trade = _run(_bars((106.0, 106.5, 105.8, 106.3))).trades[0]
    assert trade.exit_reason == "target"
    assert trade.exit == pytest.approx(106.0 - 0.0106)   # 1bp of 106 clears the floor


def test_a_bar_touching_both_levels_still_assumes_the_stop():
    """Bar 2 opens at 100.0, between the levels, and trades through both. The
    open is beyond neither, so the gap rule does not apply and OHLC cannot say
    which came first. Pessimism stands.

    This is the case the gap fix must NOT change. Letting rule 1 reach an
    intrabar bar would turn a conservative backtest into an optimistic one.
    """
    trade = _run(_bars((100.0, 104.5, 98.5, 100.5))).trades[0]
    assert trade.exit_reason == "stop"
    assert trade.exit == pytest.approx(STOP - 0.01)   # 1bp of 99 is 0.0099: floor binds


def test_a_short_gapping_up_through_its_stop_fills_at_the_open():
    """The mirror image. A short's stop is above it, and covering is a buy, so
    slippage is paid rather than given up."""
    trade = _run(_bars((106.0, 106.5, 105.8, 106.3)), strategy=OneShotShort()).trades[0]
    assert trade.exit_reason == "stop"
    assert trade.exit == pytest.approx(106.0 + 0.0106)


def test_a_short_gapping_down_through_its_target_fills_at_the_open():
    trade = _run(_bars((96.0, 96.4, 95.5, 96.2)), strategy=OneShotShort()).trades[0]
    assert trade.exit_reason == "target"
    assert trade.exit == pytest.approx(96.0 + 0.01)


# ---------------------------------------------------------------------------
# Fees.

def test_fees_are_zero_by_default_and_recorded_anyway():
    """Correct for the broker traded through today. Recorded per trade so a fee
    change is re-provable rather than archaeological."""
    trade = _run(_bars(_TAIL)).trades[0]
    assert trade.fees == 0.0


def test_fees_are_charged_round_trip_and_deducted_from_pnl():
    """Entry and exit, hence 2x. A per-trade commission of $1 on a round trip
    is $2, and it comes out of pnl rather than sitting beside it, because every
    metric above this layer reads pnl."""
    priced = _run(_bars(_TAIL), fees=FeeModel(per_trade=1.0))
    free = _run(_bars(_TAIL))

    assert priced.trades[0].fees == pytest.approx(2.0)
    assert priced.trades[0].pnl == pytest.approx(free.trades[0].pnl - 2.0)


def test_per_share_fees_scale_with_size():
    trade = _run(_bars(_TAIL), fees=FeeModel(per_share=0.001)).trades[0]
    assert trade.fees == pytest.approx(2 * 0.001 * trade.qty)


# ---------------------------------------------------------------------------
# The ledger seam.

def test_pending_total_reports_unsettled_cash():
    """The public accessor _mark now uses. Reaching into _pending meant a
    change to settlement bookkeeping broke equity marking from a distance."""
    now = pd.Timestamp("2025-03-03 20:00", tz="UTC")
    ledger = SettledLedger(1_000.0)
    ledger.credit(250.0, now)

    assert ledger.pending_total() == pytest.approx(250.0)


def test_pending_total_empties_as_cash_settles():
    """settled() sweeps matured entries out of _pending, so callers must ask in
    that order. This pins the interaction the engine depends on."""
    monday = pd.Timestamp("2025-03-03 20:00", tz="UTC")
    ledger = SettledLedger(1_000.0)
    ledger.credit(250.0, monday)

    ledger.settled(monday + pd.Timedelta(days=1))

    assert ledger.pending_total() == pytest.approx(0.0)
