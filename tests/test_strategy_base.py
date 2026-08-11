import dataclasses

import pandas as pd
import pytest

from nakagai.engine.portfolio_types import ManagementDecision, PositionView, Signal
from nakagai.strategies.base import Direction, Strategy


class Dummy(Strategy):
    name = "dummy"
    DEFAULT_PARAMS = {"a": 1, "b": 2}

    def on_bar(self, ctx):
        return []


def _position() -> PositionView:
    return PositionView(direction="long", qty=3,
                        entry_ts=pd.Timestamp("2026-06-01 14:30", tz="UTC"),
                        entry=100.0, initial_stop=98.0, initial_target=104.0,
                        live_stop=98.0, live_target=104.0)


def test_params_merge_defaults():
    assert Dummy().params == {"a": 1, "b": 2}
    assert Dummy({"b": 9}).params == {"a": 1, "b": 9}


def test_manage_defaults_to_hold():
    assert Dummy().manage(_position(), object()) == ManagementDecision(
        action="hold", stop=None, target=None)


def test_the_default_manage_cannot_touch_the_position():
    position = _position()
    Dummy().manage(position, object())
    with pytest.raises(dataclasses.FrozenInstanceError):
        position.live_stop = 1.0


def test_signal_is_frozen():
    s = Signal("SPY", Direction.LONG, 100.0, 99.0, 105.0, 0.7, ("sweep",), "why")
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.stop = 1.0
