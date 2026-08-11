"""The strict strategy boundary.

Three properties are load-bearing and each one has failed silently before:

- what `on_bar` returns is a real ordered sequence and EVERY item survives,
  because a portfolio cannot audit contention over signals that disappeared;
- `manage` reads an immutable `PositionView` and answers with a
  `ManagementDecision`, so no strategy can reach into engine-owned state;
- a strategy that raises never becomes an empty list. A refusal and a quiet
  bar must not be the same observation.
"""

import dataclasses

import pandas as pd
import pytest

from nakagai.engine.portfolio_types import (
    ManagementDecision,
    PositionView,
    Signal,
    StrategyOutputError,
    StrategyRuntimeError,
)
from nakagai.strategies.base import (
    HOLD,
    MarketContext,
    Strategy,
    call_manage,
    call_on_bar,
    strategy_operation,
    validate_management_decision,
    validate_signal_sequence,
)

DECIDING_CLOSE = 100.0
ENTRY_TS = pd.Timestamp("2026-06-01 14:30", tz="UTC")


def long_signal(symbol="SPY", *, tag="setup", **overrides) -> Signal:
    fields = {
        "symbol": symbol, "direction": "long", "entry_ref": DECIDING_CLOSE,
        "stop": 98.0, "target": 104.0, "confidence": 0.5,
        "setup_tags": (tag,), "rationale": "why",
    }
    return Signal(**{**fields, **overrides})


def short_signal(symbol="SPY", **overrides) -> Signal:
    mirrored = {"direction": "short", "stop": 104.0, "target": 98.0}
    return long_signal(symbol, **{**mirrored, **overrides})


def long_position_view(**overrides) -> PositionView:
    fields = {
        "direction": "long", "qty": 10, "entry_ts": ENTRY_TS, "entry": 100.0,
        "initial_stop": 98.0, "initial_target": 104.0,
        "live_stop": 98.0, "live_target": 104.0,
    }
    return PositionView(**{**fields, **overrides})


def short_position_view(**overrides) -> PositionView:
    fields = {
        "direction": "short", "qty": 10, "entry_ts": ENTRY_TS, "entry": 100.0,
        "initial_stop": 104.0, "initial_target": 98.0,
        "live_stop": 104.0, "live_target": 98.0,
    }
    return PositionView(**{**fields, **overrides})


def market_context(symbol="SPY", close=DECIDING_CLOSE) -> MarketContext:
    idx = pd.date_range("2026-06-01 14:30", periods=2, freq="15min", tz="UTC")
    bars = pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0,
         "close": close, "volume": 1_000.0},
        index=idx,
    )
    return MarketContext(symbol=symbol, now=idx[-1] + pd.Timedelta(minutes=15),
                         bars={"15m": bars})


class Returns(Strategy):
    """Hands back whatever it was built with, however wrong that is."""

    name = "returns"

    def __init__(self, value, decision=HOLD, params=None):
        super().__init__(params)
        self._value, self._decision = value, decision

    def on_bar(self, ctx):
        return self._value

    def manage(self, position, ctx):
        return self._decision


class Raises(Strategy):
    """Fails the way real strategy code fails: from inside its own body."""

    name = "raises"

    def on_bar(self, ctx):
        return [1 / 0]

    def manage(self, position, ctx):
        raise KeyError("atr")


class Mutates(Strategy):
    """Tries to ratchet by assignment, the way the old contract allowed."""

    name = "mutates"

    def on_bar(self, ctx):
        return []

    def manage(self, position, ctx):
        position.live_stop = 99.0
        return HOLD


# ------------------------------------------------------- the returned sequence


@pytest.mark.parametrize("bad", ["signal", {"symbol": "SPY"}, iter(()),
                                 None, long_signal(), 7])
def test_on_bar_requires_a_real_sequence(bad):
    with pytest.raises(StrategyOutputError):
        validate_signal_sequence(bad, symbol="SPY", deciding_close=DECIDING_CLOSE)


def test_an_empty_sequence_is_a_valid_quiet_bar():
    assert validate_signal_sequence([], symbol="SPY",
                                    deciding_close=DECIDING_CLOSE) == ()
    assert validate_signal_sequence((), symbol="SPY",
                                    deciding_close=DECIDING_CLOSE) == ()


def test_ordered_multi_signal_output_keeps_every_signal():
    got = validate_signal_sequence(
        [long_signal("SPY", tag="first"), long_signal("SPY", tag="second")],
        symbol="SPY",
        deciding_close=DECIDING_CLOSE,
    )
    assert [signal.setup_tags for signal in got] == [("first",), ("second",)]


def test_a_valid_sequence_returns_a_tuple_of_the_same_signals():
    signals = [long_signal(), short_signal()]
    got = validate_signal_sequence(signals, symbol="SPY",
                                   deciding_close=DECIDING_CLOSE)
    assert isinstance(got, tuple)
    assert [signal is item for signal, item in zip(got, signals)] == [True, True]


def test_a_non_signal_item_is_refused_even_beside_valid_signals():
    with pytest.raises(StrategyOutputError) as caught:
        validate_signal_sequence([long_signal(), {"symbol": "SPY"}],
                                 symbol="SPY", deciding_close=DECIDING_CLOSE)
    assert caught.value.details["field"] == "signals[1]"


# ------------------------------------------------------------- per-signal rules


@pytest.mark.parametrize("overrides", [
    pytest.param({"symbol": "QQQ"}, id="wrong_symbol"),
    pytest.param({"symbol": "spy"}, id="uncanonical_symbol"),
    pytest.param({"symbol": ""}, id="blank_symbol"),
    pytest.param({"direction": "flat"}, id="unsupported_direction"),
    pytest.param({"direction": "LONG"}, id="uppercase_direction"),
    pytest.param({"direction": 1}, id="numeric_direction"),
    pytest.param({"entry_ref": float("nan")}, id="nan_entry_ref"),
    pytest.param({"entry_ref": float("inf")}, id="infinite_entry_ref"),
    pytest.param({"entry_ref": 0.0}, id="zero_entry_ref"),
    pytest.param({"entry_ref": 99.0}, id="entry_ref_off_the_deciding_close"),
    pytest.param({"stop": -1.0}, id="negative_stop"),
    pytest.param({"stop": float("nan")}, id="nan_stop"),
    pytest.param({"stop": True}, id="boolean_stop"),
    pytest.param({"target": 0.0}, id="zero_target"),
    pytest.param({"target": float("-inf")}, id="infinite_target"),
    pytest.param({"confidence": 0.0}, id="zero_confidence"),
    pytest.param({"confidence": 1.5}, id="confidence_above_one"),
    pytest.param({"confidence": -0.5}, id="negative_confidence"),
    pytest.param({"confidence": float("nan")}, id="nan_confidence"),
    pytest.param({"confidence": True}, id="boolean_confidence"),
    pytest.param({"setup_tags": ()}, id="empty_tags"),
    pytest.param({"setup_tags": ("",)}, id="blank_tag"),
    pytest.param({"setup_tags": ("  ",)}, id="whitespace_tag"),
    pytest.param({"setup_tags": ["tag"]}, id="tags_are_not_a_tuple"),
    pytest.param({"setup_tags": (7,)}, id="non_string_tag"),
    pytest.param({"rationale": ""}, id="blank_rationale"),
    pytest.param({"rationale": "   "}, id="whitespace_rationale"),
    pytest.param({"rationale": None}, id="missing_rationale"),
    pytest.param({"stop": 101.0}, id="long_stop_above_the_reference"),
    pytest.param({"target": 99.0}, id="long_target_below_the_reference"),
    pytest.param({"stop": 100.0}, id="long_stop_on_the_reference"),
    pytest.param({"target": 100.0}, id="long_target_on_the_reference"),
])
def test_a_signal_outside_the_contract_is_refused(overrides):
    with pytest.raises(StrategyOutputError):
        validate_signal_sequence([long_signal(**overrides)], symbol="SPY",
                                 deciding_close=DECIDING_CLOSE)


@pytest.mark.parametrize("overrides", [
    pytest.param({"stop": 99.0}, id="short_stop_below_the_reference"),
    pytest.param({"target": 101.0}, id="short_target_above_the_reference"),
    pytest.param({"stop": 100.0}, id="short_stop_on_the_reference"),
    pytest.param({"target": 100.0}, id="short_target_on_the_reference"),
])
def test_short_geometry_is_the_mirror_of_long_geometry(overrides):
    with pytest.raises(StrategyOutputError):
        validate_signal_sequence([short_signal(**overrides)], symbol="SPY",
                                 deciding_close=DECIDING_CLOSE)


def test_valid_long_and_short_geometry_passes():
    got = validate_signal_sequence([long_signal(), short_signal()],
                                   symbol="SPY", deciding_close=DECIDING_CLOSE)
    assert [signal.direction for signal in got] == ["long", "short"]


def test_confidence_of_exactly_one_is_allowed():
    got = validate_signal_sequence([long_signal(confidence=1.0)], symbol="SPY",
                                   deciding_close=DECIDING_CLOSE)
    assert got[0].confidence == 1.0


def test_the_signal_is_frozen():
    signal = long_signal()
    with pytest.raises(dataclasses.FrozenInstanceError):
        signal.stop = 1.0


# --------------------------------------------------------- management decisions


def test_management_is_a_frozen_decision_and_cannot_loosen_a_stop():
    position = long_position_view(live_stop=99.0)
    with pytest.raises(StrategyOutputError):
        validate_management_decision(
            ManagementDecision(action="hold", stop=98.0, target=None),
            position=position,
            deciding_close=101.0,
        )


def test_a_short_stop_cannot_loosen_upward():
    position = short_position_view(live_stop=101.0)
    with pytest.raises(StrategyOutputError):
        validate_management_decision(
            ManagementDecision(action="hold", stop=102.0, target=None),
            position=position,
            deciding_close=99.0,
        )


def test_a_tightening_stop_is_accepted_in_both_directions():
    long_decision = validate_management_decision(
        ManagementDecision(action="hold", stop=99.5, target=None),
        position=long_position_view(live_stop=99.0), deciding_close=101.0,
    )
    short_decision = validate_management_decision(
        ManagementDecision(action="hold", stop=100.5, target=None),
        position=short_position_view(live_stop=101.0), deciding_close=99.0,
    )
    assert (long_decision.stop, short_decision.stop) == (99.5, 100.5)


def test_an_unchanged_stop_is_not_a_loosening():
    decision = validate_management_decision(
        ManagementDecision(action="hold", stop=99.0, target=None),
        position=long_position_view(live_stop=99.0), deciding_close=101.0,
    )
    assert decision.stop == 99.0


def test_null_levels_keep_the_live_levels():
    position = long_position_view(live_stop=99.0, live_target=103.0)
    decision = validate_management_decision(HOLD, position=position,
                                            deciding_close=101.0)
    assert (decision.stop, decision.target) == (None, None)


@pytest.mark.parametrize("position,close", [
    pytest.param(long_position_view(live_stop=99.0, live_target=103.0), 97.0,
                 id="long_closed_below_its_stop"),
    pytest.param(long_position_view(live_stop=99.0, live_target=103.0), 105.0,
                 id="long_closed_above_its_target"),
    pytest.param(short_position_view(live_stop=101.0, live_target=97.0), 103.0,
                 id="short_closed_above_its_stop"),
    pytest.param(short_position_view(live_stop=101.0, live_target=97.0), 95.0,
                 id="short_closed_beyond_its_target"),
])
def test_holding_a_position_whose_close_left_its_levels_is_not_the_strategys_fault(
        position, close):
    """Where the close sits against a level the decision did not touch is the
    engine's own state. A stock hold claimed nothing about it, so refusing
    here would abort an ordinary losing trade and blame the strategy."""
    assert validate_management_decision(HOLD, position=position,
                                        deciding_close=close) is HOLD


@pytest.mark.parametrize("position,decision,close,crossed", [
    pytest.param(long_position_view(live_stop=99.0, live_target=104.0),
                 ManagementDecision(action="hold", stop=104.4, target=None),
                 105.0, "stop", id="long_stop_pushed_past_a_stale_target"),
    pytest.param(long_position_view(live_stop=99.0, live_target=104.0),
                 ManagementDecision(action="hold", stop=None, target=98.5),
                 98.0, "target", id="long_target_pulled_through_a_stale_stop"),
    pytest.param(short_position_view(live_stop=101.0, live_target=97.0),
                 ManagementDecision(action="hold", stop=96.5, target=None),
                 96.0, "stop", id="short_stop_pulled_past_a_stale_target"),
    pytest.param(short_position_view(live_stop=101.0, live_target=97.0),
                 ManagementDecision(action="hold", stop=None, target=101.5),
                 102.0, "target", id="short_target_pushed_through_a_stale_stop"),
])
def test_a_replacement_cannot_cross_an_untouched_counterpart_level(
        position, decision, close, crossed):
    """Each of these passes the close check on its own and still inverts the
    bracket. An inverted bracket is not a tighter stop: `_check_exit` tests
    `open <= stop` then `open >= target`, so once they cross, those two
    branches cover the whole real line and the position exits at the next
    open whatever the price does."""
    with pytest.raises(StrategyOutputError) as caught:
        validate_management_decision(decision, position=position,
                                     deciding_close=close)
    assert caught.value.code == "crossed_protective_levels"
    assert caught.value.details["field"] == crossed


def test_a_replacement_that_stops_short_of_its_counterpart_is_accepted():
    """The guard rail is crossing, not proximity: a stop may ratchet right up
    under an untouched target."""
    decision = validate_management_decision(
        ManagementDecision(action="hold", stop=103.9, target=None),
        position=long_position_view(live_stop=99.0, live_target=104.0),
        deciding_close=105.0,
    )
    assert decision.stop == 103.9


def test_the_two_geometry_rules_refuse_under_different_codes():
    """They answer different questions, so a caller can tell them apart and a
    reader cannot collapse them back into one pivot check."""
    position = long_position_view(live_stop=99.0, live_target=104.0)
    with pytest.raises(StrategyOutputError) as unprotective:
        validate_management_decision(
            ManagementDecision(action="hold", stop=102.0, target=None),
            position=position, deciding_close=101.0)
    with pytest.raises(StrategyOutputError) as crossed:
        validate_management_decision(
            ManagementDecision(action="hold", stop=104.4, target=None),
            position=position, deciding_close=105.0)
    assert unprotective.value.code == "unprotective_replacement"
    assert crossed.value.code == "crossed_protective_levels"


def test_a_replacement_is_judged_against_the_close_not_an_untouched_level():
    """A long whose bar rallied through its target and closed above it can
    still ratchet its stop: the stop is a real protective level at that
    close, and the untouched target is the engine's business."""
    decision = validate_management_decision(
        ManagementDecision(action="hold", stop=101.85, target=None),
        position=long_position_view(live_stop=99.0, live_target=104.0),
        deciding_close=105.0,
    )
    assert decision.stop == 101.85


@pytest.mark.parametrize("bad", [
    pytest.param("exit", id="a_string"),
    pytest.param(None, id="none"),
    pytest.param(True, id="a_boolean"),
    pytest.param(("hold", None, None), id="a_tuple"),
    pytest.param({"action": "hold"}, id="a_mapping"),
])
def test_manage_must_return_a_management_decision(bad):
    with pytest.raises(StrategyOutputError):
        validate_management_decision(bad, position=long_position_view(),
                                     deciding_close=101.0)


@pytest.mark.parametrize("decision,close", [
    pytest.param(ManagementDecision(action="hold", stop=None, target=99.0),
                 101.0, id="long_target_below_the_close"),
    pytest.param(ManagementDecision(action="hold", stop=102.0, target=None),
                 101.0, id="long_stop_above_the_close"),
    pytest.param(ManagementDecision(action="hold", stop=101.0, target=None),
                 101.0, id="long_stop_on_the_close"),
    pytest.param(ManagementDecision(action="exit", stop=None, target=101.0),
                 101.0, id="long_target_on_the_close"),
])
def test_a_replacement_must_protect_the_deciding_close(decision, close):
    with pytest.raises(StrategyOutputError):
        validate_management_decision(decision,
                                     position=long_position_view(live_stop=99.0),
                                     deciding_close=close)


@pytest.mark.parametrize("decision,close", [
    pytest.param(ManagementDecision(action="hold", stop=None, target=103.0),
                 99.0, id="short_target_above_the_close"),
    pytest.param(ManagementDecision(action="hold", stop=98.0, target=None),
                 99.0, id="short_stop_below_the_close"),
    pytest.param(ManagementDecision(action="hold", stop=99.0, target=None),
                 99.0, id="short_stop_on_the_close"),
])
def test_a_short_replacement_is_the_mirror(decision, close):
    with pytest.raises(StrategyOutputError):
        validate_management_decision(
            decision, position=short_position_view(live_stop=101.0),
            deciding_close=close)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0, True])
def test_a_nonfinite_or_nonpositive_replacement_cannot_even_be_built(bad):
    """The decision refuses the value at construction, so `manage` raises
    inside its own body. `call_manage` still reports a strategy OUTPUT error,
    because the strategy produced a value core refuses, not a runtime fault."""
    strategy = Returns([], decision=None)

    def manage(position, ctx):
        return ManagementDecision(action="hold", stop=bad, target=None)

    strategy.manage = manage
    with pytest.raises(StrategyOutputError):
        call_manage(strategy, long_position_view(), market_context(),
                    deciding_close=101.0)


def test_management_mutation_is_an_output_error_not_a_runtime_error():
    with pytest.raises(StrategyOutputError) as caught:
        call_manage(Mutates(), long_position_view(), market_context(),
                    deciding_close=101.0)
    assert caught.value.details["operation"] == "manage"


# ------------------------------------------------------------- the error boundary


def test_a_raising_on_bar_never_becomes_an_empty_list():
    with pytest.raises(StrategyRuntimeError) as caught:
        call_on_bar(Raises(), market_context(), deciding_close=DECIDING_CLOSE)
    assert caught.value.details["operation"] == "on_bar"
    assert caught.value.details["strategy"] == "raises"


def test_a_raising_manage_never_becomes_a_hold():
    with pytest.raises(StrategyRuntimeError) as caught:
        call_manage(Raises(), long_position_view(), market_context(),
                    deciding_close=DECIDING_CLOSE)
    assert caught.value.details["operation"] == "manage"


def test_an_invalid_return_is_an_output_error_rather_than_a_runtime_error():
    with pytest.raises(StrategyOutputError):
        call_on_bar(Returns("signal"), market_context(),
                    deciding_close=DECIDING_CLOSE)


def test_a_valid_strategy_call_returns_every_signal_in_order():
    strategy = Returns([long_signal(tag="first"), long_signal(tag="second")])
    got = call_on_bar(strategy, market_context(), deciding_close=DECIDING_CLOSE)
    assert [signal.setup_tags for signal in got] == [("first",), ("second",)]


def test_the_default_manage_holds_without_touching_the_position():
    position = long_position_view(live_stop=99.0)
    decision = call_manage(Returns([]), position, market_context(),
                           deciding_close=101.0)
    assert decision == ManagementDecision(action="hold", stop=None, target=None)
    assert position.live_stop == 99.0


def test_strategy_operation_carries_stable_details_and_no_traceback():
    with pytest.raises(StrategyRuntimeError) as caught:
        with strategy_operation("dependencies", strategy="sma_cross", symbol="SPY"):
            raise ValueError("boom")
    details = caught.value.details
    assert details == {"operation": "dependencies", "error": "ValueError",
                       "strategy": "sma_cross", "symbol": "SPY"}
    assert isinstance(caught.value.__cause__, ValueError)


def test_strategy_operation_lets_a_typed_strategy_error_through_unchanged():
    original = StrategyOutputError("invalid_value", "bad", {"field": "stop"})
    with pytest.raises(StrategyOutputError) as caught:
        with strategy_operation("on_bar", strategy="x", symbol="SPY"):
            raise original
    assert caught.value is original


def test_a_constructor_failure_is_a_runtime_error():
    with pytest.raises(StrategyRuntimeError) as caught:
        with strategy_operation("construct", strategy="rules", symbol="SPY"):
            raise ValueError("spec invalid")
    assert caught.value.details["operation"] == "construct"
