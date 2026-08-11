"""The exits block: what a rule spec's `manage` decides, and what it does.

Two halves, and they answer different questions. The first replays a real spec
through `run_portfolio`, so a time stop, a rule exit, a trailing stop, and a
breakeven each move a real position through a real ledger. The second calls
`RuleStrategy.manage` directly, because a decision is a VALUE: the boundary
between "what the strategy returned" and "what the account did with it" is
where a mutation used to hide, and only a direct call can see the returned
decision before anything applies it.
"""

import numpy as np
import pandas as pd

import pytest

from nakagai.engine import ExitReason
from nakagai.engine.portfolio_types import ManagementDecision, PositionView
from nakagai.strategies.base import MarketContext, validate_management_decision
from nakagai.strategies.rules import RuleStrategy
from nakagai.strategies.rules.frame_eval import FrameEval
from nakagai.strategies.rules.vocabulary import core_vocabulary
from tests.portfolio_fixtures import RULES_WARMUP_INTERVALS, rules_replay

RISK = {"stop": {"kind": "percent", "pct": 5.0}, "target": {"kind": "rr", "rr": 20.0}}
# A short's rr target walks DOWN from the reference, so rr 20 against a 5%
# stop lands it on zero, which is not a price any tape reaches. rr 15 keeps
# the target far enough below the fall (~25 against a low of 40) that these
# exits still fire on the stop, and keeps it a real level.
SHORT_RISK = {"stop": {"kind": "percent", "pct": 5.0},
              "target": {"kind": "rr", "rr": 15.0}}

# The replay's first twenty intervals are warmup, and nothing opening inside
# them is evaluated. A long fixture therefore starts under its own entry level
# for exactly that many bars, so the cross it cares about lands in the test
# range rather than being warmed away.
WARMUP = [99.0] * RULES_WARMUP_INTERVALS
BAR = pd.Timedelta(minutes=15)


def _spec(exits):
    return {"version": 2, "name": "t", "timeframe": "15m",
            "long": {"all": [{"lhs": {"src": "close"}, "op": "crosses_above", "rhs": 100.0}]},
            "exits": exits, "risk": RISK}


def _short_spec(exits):
    return {"version": 2, "name": "ts", "timeframe": "15m",
            "short": {"all": [{"lhs": {"src": "close"}, "op": "crosses_below", "rhs": 100.0}]},
            "exits": exits, "risk": SHORT_RISK}


def _run(closes, exits):
    return rules_replay(_spec(exits), WARMUP + list(closes))


def _run_short(closes, exits):
    return rules_replay(_short_spec(exits), list(closes))


def test_time_stop_closes_after_n_bars():
    """Four bars after the fill, to the interval, and by a management exit.

    The old engine could only be pinned to a range here because its clock was
    bar arithmetic. The replay's clock is the schedule, so the holding period
    is an exact count of scheduled intervals.
    """
    closes = [99.0, 101.0] + [101.0] * 40      # entry, then drift sideways forever
    result = _run(closes, {"time_stop": {"bars": 4}})

    (trade,) = result.trades
    assert trade.exit_reason is ExitReason.MANAGE
    assert (trade.exit_ts - trade.entry_ts) == 4 * BAR


def test_rule_exit_fires():
    """A condition group in the exits block closes the position on its own."""
    closes = [99.0, 101.0] + list(np.linspace(101, 120, 30))
    exits = {"exit": {"any": [{"lhs": {"src": "close"}, "op": ">", "rhs": 110.0}]}}

    result = _run(closes, exits)

    (trade,) = result.trades
    assert trade.exit_reason is ExitReason.MANAGE
    assert trade.exit < 120             # left before the top


def test_trailing_stop_ratchets_and_stops_out():
    """The ratchet reaches the ledger, which is what a returned decision is for.

    The recorded final stop is above the entry, and the trade is a winner
    stopped out rather than a loser. Nothing but the account applying the
    strategy's returned stop could produce that pair.

    Either stop reason is accepted, and the difference is not this test's
    subject: a stop that has ratcheted right up under a falling tape is
    routinely already beyond the next bar's OPEN, which books at that open as a
    gap rather than at the level. Which of the two a given fixture lands on is
    the gap rule's business and is pinned where the gap rule lives.
    """
    up = list(np.linspace(99, 115, 20))
    down = list(np.linspace(115, 104, 12))

    result = _run(up + down, {"trailing": {"kind": "percent", "pct": 3.0}})

    (trade,) = result.trades
    assert trade.exit_reason in (ExitReason.STOP, ExitReason.STOP_GAP)
    assert trade.final_stop > trade.entry
    assert trade.final_stop != trade.initial_stop
    assert trade.net_pnl > 0


def test_breakeven_moves_stop_to_entry():
    """A round trip that gave back everything still loses almost nothing."""
    up = list(np.linspace(99, 112, 16))         # > 1R in profit at 5% risk
    down = list(np.linspace(112, 90, 16))

    result = _run(up + down, {"breakeven_at": {"rr": 1.0}})

    (trade,) = result.trades
    assert trade.exit_reason is ExitReason.STOP
    assert abs(trade.exit - trade.entry) < 1.0  # stopped near breakeven, not at -5%


def test_a_spec_with_no_exits_block_still_protects_the_position():
    """Without an exits block the initial levels are the only ones there are."""
    closes = [99.0, 101.0] + list(np.linspace(101, 90, 20))

    result = _run(closes, {})

    (trade,) = result.trades
    assert trade.exit_reason in (ExitReason.STOP, ExitReason.STOP_GAP,
                                 ExitReason.END_OF_WINDOW)
    assert trade.final_stop == trade.initial_stop


def test_short_atr_trailing_stop_ratchets_down_and_stops_out():
    """The mirror: a short's trailing stop only ever moves DOWN.

    Flat first, to build fourteen bars of ATR history without crossing the
    entry level. A steady fall crosses below 100 and keeps falling, so the ATR
    trailing stop ratchets below the entry price, and a hard rally then pushes
    a bar's high back above that ratcheted level.
    """
    flat = [105.0] * 20
    fall_to_entry = list(np.linspace(105, 90, 15))
    fall_further = list(np.linspace(90, 40, 30))
    rally = list(np.linspace(40, 90, 10))

    result = _run_short(flat + fall_to_entry + fall_further + rally,
                        {"trailing": {"kind": "atr", "n": 14, "mult": 2.0}})

    (trade,) = result.trades
    assert trade.direction == "short"
    assert trade.exit_reason in (ExitReason.STOP, ExitReason.STOP_GAP)
    assert trade.final_stop < trade.entry            # never above
    assert trade.net_pnl > 0


# ------------------------------- management is a value, not a mutation

def _view(**overrides):
    fields = {"direction": "long", "qty": 10,
              "entry_ts": pd.Timestamp("2026-01-05 14:30", tz="UTC"),
              "entry": 100.0, "initial_stop": 95.0, "initial_target": 200.0,
              "live_stop": 95.0, "live_target": 200.0}
    return PositionView(**{**fields, **overrides})


def _manage_ctx(closes, now_offset=1):
    idx = pd.date_range("2026-01-05 14:30", periods=len(closes), freq="15min",
                        tz="UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    bars = pd.DataFrame({"open": c, "high": c + 0.2, "low": c - 0.2,
                         "close": c, "volume": 1000.0}, index=idx)
    return MarketContext(symbol="SPY", now=idx[-1] + pd.Timedelta(minutes=15),
                         bars={"15m": bars, "1h": bars, "1d": bars})


def test_a_ratchet_comes_back_as_a_decision_and_leaves_the_view_alone():
    strategy = RuleStrategy({"spec": _spec({"trailing": {"kind": "percent",
                                                         "pct": 3.0}})})
    position = _view()
    ctx = _manage_ctx([110.0] * 20)
    decision = strategy.manage(position, ctx)
    assert decision.action == "hold"
    assert decision.stop == pytest.approx(110.0 * 0.97)
    assert position.live_stop == 95.0     # the view never moved


def test_a_ratchet_that_would_loosen_the_live_stop_is_not_returned():
    """3% below 110 is 106.7, which is BELOW a live stop already at 108. A
    trailing stop that gave that back would widen the risk it exists to cut."""
    strategy = RuleStrategy({"spec": _spec({"trailing": {"kind": "percent",
                                                         "pct": 3.0}})})
    decision = strategy.manage(_view(live_stop=108.0), _manage_ctx([110.0] * 20))
    assert decision.stop is None


def test_a_spec_with_no_exits_block_holds():
    strategy = RuleStrategy({"spec": _spec({})})
    assert strategy.manage(_view(), _manage_ctx([110.0] * 20)) == ManagementDecision(
        action="hold", stop=None, target=None)


def test_a_rule_exit_comes_back_as_an_exit_decision():
    exits = {"exit": {"any": [{"lhs": {"src": "close"}, "op": ">", "rhs": 105.0}]}}
    strategy = RuleStrategy({"spec": _spec(exits)})
    ctx = _manage_ctx([110.0] * 20)
    ctx.fe = FrameEval(ctx.bars, vocabulary=core_vocabulary())
    ctx.cursor = {"15m": len(ctx.bars["15m"]) - 1}
    assert strategy.manage(_view(), ctx).action == "exit"


def _flat_ctx(price=110.0, n=20):
    """Zero-range bars: a halt, an illiquid tape, or a flat fixture. ATR over
    this window is exactly 0.0, which is what drives the trailing distance to
    nothing."""
    idx = pd.date_range("2026-01-05 14:30", periods=n, freq="15min", tz="UTC")
    bars = pd.DataFrame({"open": price, "high": price, "low": price,
                         "close": price, "volume": 1000.0}, index=idx)
    return MarketContext(symbol="SPY", now=idx[-1] + pd.Timedelta(minutes=15),
                         bars={"15m": bars, "1h": bars, "1d": bars})


def test_a_zero_range_window_does_not_ratchet_the_stop_onto_the_close():
    """A zero ATR puts the trailing candidate exactly on the deciding close,
    which is the price rather than a level protecting it. The boundary refuses
    such a stop, so the producer must not hand it one."""
    strategy = RuleStrategy({"spec": _spec({"trailing": {"kind": "atr", "n": 14,
                                                         "mult": 2.0}})})
    ctx = _flat_ctx()
    position = _view()
    decision = strategy.manage(position, ctx)
    assert decision.stop is None
    assert validate_management_decision(decision, position=position,
                                        deciding_close=110.0) is decision


def test_a_short_zero_range_window_holds_too():
    strategy = RuleStrategy({"spec": _short_spec({"trailing": {"kind": "atr",
                                                              "n": 14,
                                                              "mult": 2.0}})})
    position = _view(direction="short", initial_stop=115.0, initial_target=90.0,
                     live_stop=115.0, live_target=90.0)
    decision = strategy.manage(position, _flat_ctx())
    assert decision.stop is None
