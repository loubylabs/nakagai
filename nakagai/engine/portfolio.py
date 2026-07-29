"""T+1 settled-funds ledger for cash-account realism. v1 skips exchange holidays."""

import datetime as dt

import pandas as pd

from nakagai.data.schema import EXCHANGE_TZ


def _next_weekday(d: dt.date) -> dt.date:
    d += dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


class SettledLedger:
    def __init__(self, cash: float):
        self._settled = float(cash)
        self._pending: list[tuple[dt.date, float]] = []  # (settle_date_NY, amount)

    def settled(self, now: pd.Timestamp) -> float:
        today = now.tz_convert(EXCHANGE_TZ).date()
        still = []
        for settle_date, amount in self._pending:
            if settle_date <= today:
                self._settled += amount
            else:
                still.append((settle_date, amount))
        self._pending = still
        return self._settled

    def reserve(self, amount: float, now: pd.Timestamp) -> bool:
        if self.settled(now) < amount:
            return False
        self._settled -= amount
        return True

    def pending_total(self) -> float:
        """Cash credited but not yet settled.

        Equity marking needs this, and reaching into _pending to get it meant a
        change to settlement bookkeeping broke the engine from a distance. Call
        settled() first: it sweeps matured entries out of _pending, so asking
        in the other order double-counts anything that has just settled.
        """
        return sum(amount for _, amount in self._pending)

    def credit(self, amount: float, now: pd.Timestamp):
        self._pending.append((_next_weekday(now.tz_convert(EXCHANGE_TZ).date()), float(amount)))
