"""Rolling walk-forward windows: the spec's 'X large cycles'."""

from dataclasses import dataclass

import pandas as pd

# The platform's standardized evaluation window: backtests and the web's
# quick-test flows all use this shape so their numbers compare like-for-like.
# Change it here and every surface moves together.
VERIFY_MONTHS = 13
VERIFY_TRAIN, VERIFY_TEST = 4, 1


@dataclass(frozen=True)
class Window:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
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
