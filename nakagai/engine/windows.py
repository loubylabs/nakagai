"""Rolling evaluation windows: the spec's 'X large cycles'.

Pillar 5 (Protocol) of the platform's docs/internal/PILLARS.md.

READ THIS BEFORE QUOTING THE PROTOCOL ANYWHERE. What the pipeline runs is
**fixed-parameter rolling out-of-sample evaluation**, not walk-forward
optimization. These windows carry a train span, and nothing fits on it: a
replay uses the spec's default parameters on every window. That is honest,
there is no in-sample leakage to worry about, but it is not "tune 4 months,
validate 1 month" and no document may say so. Decision recorded in PILLARS.md
Pillar 5, 2026-07-24.

The train span stays in the Window because it is the span a fit step WOULD use,
and recording it now means the evidence store already has the column when that
step is built. It is reserved, not used.

This module is the SHAPE of the evaluation protocol and nothing else. Core's
own replay takes a `ReplayWindow` on its request and never builds one of these;
the platform lays out a launch's windows with `walk_forward` and pins the
13/4/1 constants against its own documentation.
"""

from dataclasses import dataclass

import pandas as pd

# The platform's standardized evaluation window: backtests and the web's
# quick-test flows all use this shape so their numbers compare like-for-like.
# Change it here and every surface moves together. Mirrored in PRODUCT.md and
# docs/internal/STRATEGY_LAB.md; the platform's tests/test_docs_sync.py fails
# if those drift from these constants.
VERIFY_MONTHS = 13
VERIFY_TRAIN, VERIFY_TEST = 4, 1


@dataclass(frozen=True)
class Window:
    # RESERVED, NOT FIT ON. See the module docstring. Recorded on every
    # evidence row so a future fit step inherits the schema, and so that
    # "which span would this have been tuned over" stays answerable.
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    # The only span replayed.
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward(start: pd.Timestamp, end: pd.Timestamp,
                 train_months: int = 4, test_months: int = 1, step_months: int = 1) -> list[Window]:
    out: list[Window] = []
    cursor = start
    while True:
        train_end = cursor + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        out.append(Window(cursor, train_end, train_end, test_end))
        cursor = cursor + pd.DateOffset(months=step_months)
    return out
