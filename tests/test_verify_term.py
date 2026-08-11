# tests/test_verify_term.py
"""The per-term causality gate, and its honesty about what it cannot test."""

import numpy as np
import pandas as pd
import pytest

from nakagai.strategies.rules.vocabulary import Term, core_vocabulary
from nakagai.strategies.rules.verify import (
    CONDITION_ARG, TermVerdict, exemption_reason,
)


def test_an_end_anchored_term_is_exempt_and_says_why():
    v = core_vocabulary()
    reason = exemption_reason(v.primitives["fvg_nearest"])
    assert reason is not None
    assert "end_anchored" in reason


def test_a_condition_taking_term_is_exempt_and_says_why():
    v = core_vocabulary()
    reason = exemption_reason(v.primitives["bars_since"])
    assert reason is not None
    assert "condition" in reason


def test_an_ordinary_term_is_not_exempt():
    v = core_vocabulary()
    assert exemption_reason(v.indicators["sma"]) is None
    assert exemption_reason(v.indicators["macd"]) is None
    assert exemption_reason(v.primitives["gap_pct"]) is None


def test_exactly_these_core_terms_are_exempt():
    """The exempt set is a declared constant, not whatever the code happens to skip.

    A term becoming silently exempt is how a real look-ahead bug would hide, so
    the set is asserted rather than counted.
    """
    exempt = {t.name for t in core_vocabulary().all_terms()
              if exemption_reason(t) is not None}
    assert exempt == {"fvg_nearest", "order_block", "bars_since"}
