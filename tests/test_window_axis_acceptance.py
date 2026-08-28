"""Public acceptance proof for one named window across every RuleSpec seam."""

import json
from datetime import time

import pandas as pd

from nakagai.strategies.rules import (
    WindowSpec,
    canonical_spec,
    compile_pine,
    core_vocabulary,
    describe_spec,
    validate_spec,
)
from nakagai.strategies.rules.frame_eval import FrameEval


LOW_IEX_DISCLOSURE = "US-equity extended-hours IEX data can be sparse."
LONDON = WindowSpec(
    "london",
    "Europe/London",
    time(8),
    time(16, 30),
    "weekday",
    "low_iex",
)
HIGH = {"ind": "highest", "of": {"src": "high"}, "window": "london"}
LOW = {"ind": "lowest", "of": {"src": "low"}, "window": "london"}
SPEC = {
    "version": 2,
    "name": "london-range",
    "timeframe": "15m",
    "long": {"all": [{"lhs": HIGH, "op": ">", "rhs": LOW}]},
}


def test_public_window_axis_handles_london_high_and_low_without_bespoke_terms():
    vocabulary = core_vocabulary().with_windows(LONDON)
    bespoke = {"london_high", "london_low"}

    assert bespoke.isdisjoint(term.name for term in vocabulary.all_terms())
    assert validate_spec(SPEC, vocabulary) == []

    canonical = canonical_spec(SPEC, vocabulary)
    condition = canonical["long"]["all"][0]
    assert condition["lhs"] == HIGH
    assert condition["rhs"] == LOW
    assert bespoke.isdisjoint(json.dumps(canonical).split('"'))

    index = pd.DatetimeIndex([
        "2026-01-05 08:00",
        "2026-01-05 12:00",
        "2026-01-05 16:15",
        "2026-01-05 16:30",
    ], tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [3.0, 4.0, 5.0, 6.0],
            "high": [4.0, 9.0, 7.0, 999.0],
            "low": [2.0, 1.0, 3.0, -999.0],
            "close": [3.5, 5.0, 6.0, 7.0],
            "volume": [1000.0] * 4,
        },
        index=index,
    )
    evaluator = FrameEval(
        "SPY", {("SPY", "15m"): bars}, vocabulary=vocabulary)
    london_high = evaluator.series(HIGH, "15m")
    london_low = evaluator.series(LOW, "15m")
    assert london_high.iloc[:-1].isna().all()
    assert london_low.iloc[:-1].isna().all()
    assert london_high.iloc[-1] == 9.0
    assert london_low.iloc[-1] == 1.0

    description = describe_spec(SPEC, vocabulary)
    assert "highest(of=high) over london" in description
    assert "lowest(of=low) over london" in description
    assert description.count(LOW_IEX_DISCLOSURE) == 2

    pine = compile_pine(SPEC, vocabulary)
    warning = (
        "Window 'london' uses US-equity extended-hours IEX data, which can "
        "be sparse."
    )
    for artifact in (pine.indicator, pine.strategy):
        header = " ".join(
            line.lstrip("/").strip().removeprefix("- ")
            for line in artifact.splitlines()
            if line.startswith("//")
        )
        assert header.count(warning) == 1
        assert '"Europe/London"' in artifact
        assert not any(name in artifact for name in bespoke)
