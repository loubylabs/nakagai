import pandas as pd

from nakagai.engine.portfolio import SettledLedger

MON = pd.Timestamp("2026-06-01 15:00", tz="UTC")
TUE = pd.Timestamp("2026-06-02 15:00", tz="UTC")
FRI = pd.Timestamp("2026-06-05 15:00", tz="UTC")
NEXT_MON = pd.Timestamp("2026-06-08 15:00", tz="UTC")


def test_reserve_within_settled_cash():
    led = SettledLedger(1000.0)
    assert led.reserve(400.0, MON) is True
    assert led.settled(MON) == 600.0


def test_reserve_rejects_over_settled():
    led = SettledLedger(1000.0)
    assert led.reserve(1500.0, MON) is False
    assert led.settled(MON) == 1000.0  # untouched


def test_sale_proceeds_settle_next_weekday():
    led = SettledLedger(0.0)
    led.credit(500.0, MON)
    assert led.settled(MON) == 0.0       # same day: unsettled
    assert led.settled(TUE) == 500.0     # T+1


def test_friday_sale_settles_monday():
    led = SettledLedger(0.0)
    led.credit(500.0, FRI)
    assert led.settled(FRI + pd.Timedelta(hours=1)) == 0.0
    assert led.settled(NEXT_MON) == 500.0
