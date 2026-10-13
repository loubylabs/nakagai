"""Derived timeframes: bars computed from cached bars rather than fetched.

4h is the first of them, aggregated from the cached 1h bars and anchored at
midnight America/New_York.

Anchoring is the whole point. Alpaca's intraday bars are bucketed on the UTC
clock, so its 4Hour bars close at 08:00 / 12:00 / 16:00 ET under EDT and at
07:00 / 11:00 / 15:00 ET under EST, where the last bar of the regular session
closes a full hour BEFORE the bell. The boundaries move twice a year, so a
backtest spanning a DST change measures two different instruments and neither
of them lines up with the session on both sides of the change. Resampling the
1h cache against the Eastern wall clock closes buckets at 04:00 / 08:00 /
12:00 / 16:00 / 20:00 ET year round, so the last regular-session bucket closes
exactly on the 16:00 bell in winter and in summer alike.

Deriving locally also costs no provider requests, since 1h is already pulled
every cycle. The decisive reason, though, is that the derived bars are
materialized into the same parquet cache the backtester reads, by the same
function the live path calls: backtest and live agree by construction, because
nothing downstream can tell a derived parquet from a fetched one.

1w is the second, aggregated from the cached 1d bars into Monday-to-Friday
New York weeks. A weekly bar is labeled with its week's first session date at
New York midnight, the key BarCache gives a daily bar, so a holiday Monday
labels the week Tuesday. It is session-aligned like 1d (nakagai/data/schema),
which is why it carries no delta.
"""

from collections.abc import Mapping

import pandas as pd

from nakagai.data.schema import (
    BAR_COLUMNS, EXCHANGE_TZ, WEEK_ALIGNED, empty_bars, session_cutoff,
    validate_bars,
)

__all__ = ["DERIVED", "resample_bars"]

# Which timeframes are computed from another cached timeframe, and from which.
# One declarative place, so a caller asks rather than hardcoding the pair: the
# sync loop needs to know that 4h follows a 1h fetch and 1w a 1d fetch instead
# of being fetched themselves, and the platform needs the same fact to decide
# what a scan cycle owes.
DERIVED: Mapping[str, str] = {"4h": "1h", "1w": "1d"}

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum"}


def resample_bars(src: pd.DataFrame, timeframe: str, *,
                  now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Aggregate `src` bars of DERIVED[timeframe] into `timeframe` buckets.

    Buckets are half-open [t, t + delta) on the Eastern wall clock, labeled at
    their left edge in UTC like every other frame in the cache. open=first,
    high=max, low=min, close=last, volume=sum.

    Two rules keep the result honest, and each one is load-bearing:

    - A still-forming bucket is never published. A bucket is emitted only once
      the source holds a bar covering its final period, so a partial aggregate
      cannot land in the cache where every reader downstream takes it for a
      closed bar. It is self-healing: a bucket held back by a provider gap
      becomes emittable the moment the missing bar arrives, and the caller
      re-derives its trailing window each cycle, so the bucket is written once,
      complete.
    - Empty buckets are dropped rather than emitted. resample().agg() yields a
      row per period whether or not anything traded, and validate_bars would
      accept those NaN rows (with a volume of 0.0, since an empty sum is not
      NaN) as real overnight bars.

    Weeks follow the same two rules with a weekly notion of "complete": a week
    is published once the source holds its Friday bar, or any bar from a later
    week, or `now` falls in a later New York week. The third fact is what a
    holiday Friday needs. Without it the week waits for the next week's first
    daily bar, and a symbol with no print by the first scan of that week would
    pass the once-a-week gate (strategies/util.first_bar_of_week) before its
    week existed and never decide on it. Four weeks in 2026 end on a holiday
    Friday. A week completed by the clock alone may still lack a session the
    provider has not delivered; the next derive resumes from that week's own
    label and rewrites it whole when the bar lands.

    `now` is clock evidence and nothing else, so it is optional and keyword
    only. A 4h bucket's completeness is decided by the source alone, and the
    4h branch ignores it.
    """
    source_tf = DERIVED.get(timeframe)
    if source_tf is None:
        raise ValueError(f"{timeframe!r} is not a derived timeframe "
                         f"(derived: {sorted(DERIVED)})")
    if not len(src.index):
        return empty_bars()
    src = validate_bars(src)
    if timeframe in WEEK_ALIGNED:
        return _weeks(src, now)
    bucket = pd.Timedelta(timeframe)
    step = pd.Timedelta(source_tf)

    # Resampling the tz-aware index directly would bucket in ABSOLUTE time, so
    # the wall-clock alignment would shift by an hour across every DST change,
    # which is the exact failure this module exists to avoid. Convert to
    # Eastern, drop the offset, and bucket on naive wall-clock time instead.
    naive = src.copy()
    naive.index = src.index.tz_convert(EXCHANGE_TZ).tz_localize(None)
    grid = naive.resample(bucket, origin="start_day", label="left", closed="left")
    out = grid.agg(_AGG)
    out = out[grid.size().to_numpy() > 0]

    # Withhold the buckets the source cannot yet fill. Done in wall-clock space
    # deliberately: on the fall-back day the 00:00 bucket spans five real hours,
    # and label + bucket in absolute time would call it complete an hour early.
    covered_to = (src.index[-1] + step).tz_convert(EXCHANGE_TZ).tz_localize(None)
    out = out[out.index + bucket <= covered_to]

    # Re-localizing is safe BECAUSE the boundaries land on 00/04/08/12/16/20 ET,
    # never inside the spring-forward gap (02:00-03:00) or the fall-back
    # ambiguous hour (01:00-02:00). Anyone changing the bucket size to one that
    # can land there (1h, 2h, 3h) owes this line an `ambiguous`/`nonexistent`
    # policy; until then the default (raise) is the honest failure mode.
    out.index = out.index.tz_localize(EXCHANGE_TZ).tz_convert("UTC")
    return validate_bars(out)


def _weeks(src: pd.DataFrame, now: pd.Timestamp | None) -> pd.DataFrame:
    """1d bars into New York weeks; see resample_bars for when a week is out.

    A daily label's UTC calendar date is its session date under both label
    conventions (engine/context.closed_before rests on the same fact), so the
    session dates are read off the UTC index and never converted to New York,
    which would move a 00:00Z legacy label back a day.
    """
    sessions = src.index.tz_convert("UTC").tz_localize(None).normalize()
    weeks = sessions - pd.to_timedelta(sessions.weekday, unit="D")
    grid = src.assign(_session=sessions).groupby(weeks.to_numpy(), sort=True)
    out = grid.agg(open=("open", "first"), high=("high", "max"),
                   low=("low", "min"), close=("close", "last"),
                   volume=("volume", "sum"),
                   first=("_session", "min"), last=("_session", "max"))
    complete = ((out.index < out.index.max())
                | (out["last"].dt.weekday.to_numpy() >= 4))
    if now is not None:
        current = session_cutoff(now, "1w").tz_localize(None)
        complete = complete | (out.index < current)
    out = out[complete]
    # Midnight never falls in a daylight-saving gap or fold (both happen at
    # 02:00 local), so the re-localize needs no ambiguous/nonexistent policy.
    out.index = (pd.DatetimeIndex(out["first"]).tz_localize(EXCHANGE_TZ)
                 .tz_convert("UTC"))
    return validate_bars(out[BAR_COLUMNS])
