from nakagai.strategies.base import Direction, PositionAction, Signal, Strategy


class Dummy(Strategy):
    name = "dummy"
    DEFAULT_PARAMS = {"a": 1, "b": 2}

    def on_bar(self, ctx):
        return []


def test_params_merge_defaults():
    assert Dummy().params == {"a": 1, "b": 2}
    assert Dummy({"b": 9}).params == {"a": 1, "b": 9}


def test_manage_defaults_to_hold():
    assert Dummy().manage(object(), object()) == PositionAction.HOLD


def test_signal_is_frozen():
    s = Signal("SPY", Direction.LONG, None, 99.0, 105.0, 0.7, ("sweep",), "why")
    try:
        s.stop = 1.0
        raise AssertionError("Signal must be immutable")
    except AttributeError:
        pass
