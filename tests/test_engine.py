"""A real RuleSpec replayed through `run_portfolio`: the singleton, end to end.

Every other portfolio file drives the loop with a scripted strategy, which is
the right instrument for a question about the chronology: a script says exactly
what comes back and when. It cannot answer a different question, and this file
asks that one. A real definition, resolved from a real registry, building a
real `RuleStrategy` through a real factory, producing signals from real
indicator values over hand-authored closes.

The behaviours below are the approved ones the retired singleton engine was
pinned on: where an entry fills, which level an ambiguous bar takes, that a
short mirrors a long, that an open position is closed at the window's end, that
a fill gapping past its own stop is refused, and that a train bar warms an
indicator without ever trading. They are asserted here against the one public
door, which is what "a singleton is the smallest valid portfolio" has to mean
in practice.

Every price is a literal. The bars open and close at the same number, so a fill
at the next open is the close a reader can point at, and every stop, target,
quantity, and PnL below is derived by hand from the spec's own risk block.
"""

import pytest

from nakagai.engine import ExitReason, RejectionReason
from tests.portfolio_fixtures import (
    RULES_WARMUP_INTERVALS,
    BarPlan,
    replay_account,
    rules_interval,
    rules_replay,
)

# Two percent below the entry for the stop and twice that distance above it for
# the target. Chosen so both levels are exact decimals at an entry of 101.0:
# the stop is 98.98, the protective distance is 2.02, and the target is 105.04.
RISK = {"stop": {"kind": "percent", "pct": 2.0},
        "target": {"kind": "rr", "rr": 2.0}}
ENTRY = 101.0
STOP = 98.98
TARGET = 105.04
# A tenth of a percent of 100,000 is 100 of risk, which over 2.02 of protective
# distance is 49 shares.
QTY = 49

LONG_SPEC = {
    "version": 2, "name": "cross_long", "timeframe": "15m",
    "long": {"all": [{"lhs": {"src": "close"}, "op": "crosses_above",
                      "rhs": 100.0}]},
    "risk": RISK,
}

SHORT_SPEC = {
    "version": 2, "name": "cross_short", "timeframe": "15m",
    "short": {"all": [{"lhs": {"src": "close"}, "op": "crosses_below",
                       "rhs": 100.0}]},
    "risk": RISK,
}

# The cross lands on interval 21, the first close after the warmup range, so
# the entry fills at interval 22's open and interval 23 is the first bar that
# can carry an exit.
SIGNAL_BAR = RULES_WARMUP_INTERVALS + 1
ENTRY_BAR = SIGNAL_BAR + 1
EVENT_BAR = SIGNAL_BAR + 2


def long_closes(event: float, tail: float | None = None) -> list[float]:
    """Flat under the level, one cross, one quiet entry bar, then the event."""
    rest = event if tail is None else tail
    return [99.0] * (SIGNAL_BAR) + [ENTRY, ENTRY, event] + [rest] * 4


def short_closes(event: float) -> list[float]:
    """The mirror: flat above the level, one cross down, then the event."""
    return [101.0] * (SIGNAL_BAR) + [99.0, 99.0, event] + [event] * 4


# ----------------------------------------------------------------- entries


def test_a_signal_fills_at_the_next_scheduled_open_and_sizes_from_its_risk():
    """The entry is the whole singleton contract in one trade.

    The spec crosses 100.0 at interval 21's close and the account fills at
    interval 22's open, never at the close that produced the signal. The
    quantity is the floored risk budget over the protective distance the spec's
    own risk block computed, and both levels are recorded as the signal named
    them.
    """
    result = rules_replay(LONG_SPEC, long_closes(101.0))

    (trade,) = result.trades
    assert (trade.play_id, trade.symbol, trade.strategy) == (
        "play-r", "SPY", "cross_long")
    assert trade.direction == "long"
    assert (trade.signal_ts, trade.entry_ts) == (
        rules_interval(ENTRY_BAR), rules_interval(ENTRY_BAR))
    assert (trade.entry, trade.qty) == (ENTRY, QTY)
    assert trade.initial_stop == STOP
    assert trade.initial_target == pytest.approx(TARGET)


def test_a_bar_reaching_the_target_books_the_target_and_two_r():
    """The favourable case, with every number derived rather than measured.

    Interval 23 closes at 105.0 and its high clears 105.04, so the target is
    reached inside the bar and the exit prints the LEVEL rather than the
    extreme. Forty-nine shares over 4.04 of gain is 197.96, and dividing by the
    49 shares' 98.98 of risk is exactly two R.
    """
    result = rules_replay(LONG_SPEC, long_closes(105.0))

    (trade,) = result.trades
    assert trade.exit_reason is ExitReason.TARGET
    assert trade.exit == pytest.approx(TARGET)
    assert trade.gross_pnl == pytest.approx((TARGET - ENTRY) * QTY)
    assert trade.r_multiple == pytest.approx(2.0)
    assert result.metrics.all_trades.n_wins == 1


def test_a_bar_spanning_both_levels_takes_the_stop():
    """Pessimism on an ambiguous bar, on a real strategy's own geometry.

    Interval 23's range covers 98.98 and 105.04 at once. OHLC cannot say which
    came first, so the stop is assumed and the trade loses one R. This is the
    case the gap rule must NOT reach: the bar opens between the two levels.
    """
    result = rules_replay(
        LONG_SPEC, long_closes(101.0),
        bars=(BarPlan(symbol="SPY", at=rules_interval(EVENT_BAR), open=101.0,
                      high=106.0, low=98.0, close=101.0),))

    (trade,) = result.trades
    assert trade.exit_reason is ExitReason.STOP
    assert trade.exit == STOP
    assert trade.r_multiple == pytest.approx(-1.0)
    assert result.metrics.all_trades.n_wins == 0


def test_a_short_mirrors_the_long_on_every_side():
    """The stop sits above and the target below, and profit is the fall.

    Crossing below 100.0 at an entry of 99.0 puts the stop at 100.98 and the
    target 2 R below at 95.04. Interval 23 opens at 96.0, above the target, and
    trades down through it, so the cover is intrabar at the level itself rather
    than a gap at the open.
    """
    result = rules_replay(
        SHORT_SPEC, short_closes(96.0),
        bars=(BarPlan(symbol="SPY", at=rules_interval(EVENT_BAR), open=96.0,
                      high=96.2, low=94.9, close=96.0),))

    (trade,) = result.trades
    assert trade.direction == "short"
    assert (trade.entry, trade.initial_stop) == (99.0, pytest.approx(100.98))
    assert trade.initial_target == pytest.approx(95.04)
    assert trade.exit_reason is ExitReason.TARGET
    assert trade.gross_pnl > 0.0
    assert result.metrics.short_trades.n_trades == 1
    assert result.metrics.long_trades.n_trades == 0


def test_an_open_position_is_closed_at_the_window_end():
    """Nothing crosses a reporting-window boundary.

    The tape never reaches either level, so the position survives to the last
    scheduled close and is liquidated there rather than carried out of the
    result.
    """
    result = rules_replay(LONG_SPEC, long_closes(101.0))

    (trade,) = result.trades
    assert trade.exit_reason is ExitReason.END_OF_WINDOW
    assert trade.exit_ts == result.request.window.test_end
    assert result.equity[-1].open_positions == 0


def test_a_fill_gapping_past_its_own_stop_takes_no_position():
    """The print that broke the level came before the position existed.

    The signal is decided at 101.0 with its stop at 98.98, and the entry bar
    opens at 95.0. Entering there would open a position its own stop already
    sits inside, which the protective contract has no meaning for. It is
    recorded as a refusal with its own reason rather than dropped, and the
    signal still counts on the slice. The tape stays below the level after the
    gap, so the one refusal is the only thing this replay has to say.
    """
    result = rules_replay(LONG_SPEC, [99.0] * SIGNAL_BAR + [ENTRY] + [95.0] * 6)

    assert result.trades == ()
    (refused,) = result.rejections
    assert refused.reason is RejectionReason.INVALID_PROTECTIVE_GEOMETRY
    assert result.slices[0].signals == 1


# --------------------------------------------------------- the warmup range


def test_a_cross_inside_the_train_range_never_becomes_a_signal():
    """Train data is warmup only, and the loop opens at `test_start`.

    The tape crosses 100.0 at interval 5, deep inside the train range, and
    never crosses again. No interval opening before `test_start` is replayed,
    so nothing is evaluated there: no signal, no pending intent, and no trade
    enters the test range.
    """
    closes = [99.0] * 5 + [101.0] * (RULES_WARMUP_INTERVALS + 10)

    result = rules_replay(LONG_SPEC, closes)

    assert result.trades == ()
    assert result.rejections == ()
    assert result.slices[0].signals == 0


def test_the_train_range_is_visible_history_for_an_indicator():
    """Warmup is not invisible, it is untraded.

    A fourteen-period ATR needs fourteen closed bars before it has a value at
    all, and the cross here happens at the FIRST close of the test range. The
    stop it produced is a real number, which is only possible if the strategy
    could read the twenty bars behind `test_start`.
    """
    spec = {**LONG_SPEC, "name": "atr_long",
            "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
                     "target": {"kind": "rr", "rr": 2.0}}}
    closes = [95.0 + 0.1 * step for step in range(RULES_WARMUP_INTERVALS)] + [
        101.0] * 8

    result = rules_replay(spec, closes)

    (trade,) = result.trades
    assert 0.0 < trade.initial_stop < trade.entry
    assert trade.initial_target > trade.entry


# -------------------------------------------------------------- the account


def test_a_risk_budget_below_one_share_takes_no_position():
    """Sizing floors, so a budget that cannot buy a share buys none.

    A thousand-dollar account at a tenth of a percent is one dollar of risk,
    which over 2.02 of protective distance rounds to zero shares. That is a
    signal the account declined rather than an error, so nothing is booked.
    """
    result = rules_replay(LONG_SPEC, long_closes(101.0),
                          account=replay_account(1_000.0))

    assert result.trades == ()
    assert result.slices[0].signals == 1
    assert result.metrics.all_trades.n_trades == 0
