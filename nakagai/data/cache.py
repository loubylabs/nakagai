"""Local parquet cache. Backtests read ONLY from here, offline and reproducible."""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

# empty_bars is re-exported here because this module has always been its import
# site; it now lives in schema.py, beside BAR_COLUMNS, so the providers can share
# the one implementation instead of each hardcoding the column list.
from nakagai.data.schema import (
    EXCHANGE_TZ, WEEK_ALIGNED, empty_bars, validate_bars,
)
from nakagai.filelock import file_lock

__all__ = ["BASIS_BAND", "BarCache", "MemoryBars", "empty_bars"]

log = logging.getLogger(__name__)

# How far an overlap bar's open may move before the cached history is judged
# to be on another price basis. A bar's open does not change while the bar
# forms, so a moved open means the vendor re-adjusted the history (a split).
# The smallest common split, 5-for-4, moves it by 0.80; a stock dividend under
# 15% is inside the band and is not detected.
BASIS_BAND = 1.15


def _canonical_cache_frame(timeframe: str, df: pd.DataFrame) -> pd.DataFrame:
    """Use one New York-midnight key per daily session date."""
    if timeframe != "1d" or df.empty:
        return df
    canonical = df.copy()
    session_dates = canonical.index.tz_convert("UTC").tz_localize(None).normalize()
    canonical.index = session_dates.tz_localize(EXCHANGE_TZ).tz_convert("UTC")
    canonical.index.name = "ts"
    return canonical


def _period_keys(timeframe: str, index: pd.DatetimeIndex) -> pd.Index:
    """What makes two rows the same bar: the label, except for a week.

    A weekly label is the week's first SESSION, which moves when a missing
    Monday bar lands and a full derive relabels the week from Tuesday to
    Monday. Keyed on the label, the cache would keep both rows; keyed on the
    week, the newest copy wins, the same rule a re-derived bar already obeys.
    """
    if timeframe not in WEEK_ALIGNED:
        return index
    days = index.tz_convert("UTC").tz_localize(None).normalize()
    return days - pd.to_timedelta(days.weekday, unit="D")


def _keyed_opens(timeframe: str, df: pd.DataFrame) -> pd.Series:
    """Opens by period key, the last copy of a duplicate key winning."""
    opens = pd.Series(df["open"].to_numpy(dtype="float64"),
                      index=_period_keys(timeframe, df.index))
    return opens[~opens.index.duplicated(keep="last")]


def _moved_basis(timeframe: str, existing: pd.DataFrame,
                 incoming: pd.DataFrame) -> float | None:
    """The first overlap ratio incoming/existing open outside the band, or None.

    Only rows with a finite, positive open on both sides are compared, so a
    zero, missing or NaN open can never trigger a replace."""
    if existing.empty or incoming.empty:
        return None
    old = _keyed_opens(timeframe, existing)
    new = _keyed_opens(timeframe, incoming)
    common = old.index.intersection(new.index).sort_values()
    if not len(common):
        return None
    old, new = old.loc[common].to_numpy(), new.loc[common].to_numpy()
    usable = np.isfinite(old) & np.isfinite(new) & (old > 0) & (new > 0)
    ratios = new[usable] / old[usable]
    moved = ratios[(ratios <= 1 / BASIS_BAND) | (ratios >= BASIS_BAND)]
    return float(moved[0]) if len(moved) else None


class MemoryBars:
    """BarCache-shaped, dict-backed frames, keyed by (symbol, timeframe).

    A sweep backtests one symbol's bars against many specs; round-tripping the
    same frames through temp parquet once per run is pure overhead. load()
    mirrors BarCache.load's missing-file contract (empty schema frame)."""

    def __init__(self, frames: dict):
        self._frames = dict(frames)

    def load(self, symbol: str, timeframe: str) -> pd.DataFrame:
        df = self._frames.get((symbol, timeframe))
        return df if df is not None else empty_bars()


class BarCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # (symbol, timeframe) pairs whose file upsert replaced because the
        # price basis moved, and that no caller has refilled yet. The writers
        # in nakagai/data/sync.py refill and discard in the same call; a
        # caller with its own provider loop drains this itself.
        self.rebased: set[tuple[str, str]] = set()

    def path(self, symbol: str, timeframe: str) -> Path:
        return self.root / f"{symbol}_{timeframe}.parquet"

    def upsert(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        """Merge `df` into the cached pair, newest copy of a duplicate ts winning.
        For 1w the duplicate is the same WEEK, not the same label
        (_period_keys), so a week relabeled Tuesday to Monday stays one row.

        Locked and atomic for the reason nakagai/filelock.py exists: this is a
        read-concat-write, and an unlocked one silently keeps only whichever
        writer finished last. The scan loop, a backfill, and the proving farm
        can all touch one pair, and a lost bar reads downstream as "this play
        does not trade much" rather than as data loss. append_parquet is not
        reusable here because it concatenates with ignore_index=True, and this
        cache is indexed by ts and dedupes on that index.

        A moved price basis replaces instead of merging. Bars are split-adjusted
        at fetch time, so after a split the re-fetched overlap bar comes back
        with its open moved by the split ratio, and every cached row is on a
        basis the vendor no longer serves. When any overlap row's open moved
        outside BASIS_BAND, the incoming frame alone becomes the file: short,
        never mixed. The pair is added to `self.rebased` and one warning is
        logged; the return value is unchanged.
        """
        if "interpolated" in df.columns:
            df = df[df["interpolated"] != True]  # noqa: E712 (spec: fake gap-fill bars never enter the cache)
        df = _canonical_cache_frame(timeframe, validate_bars(df))
        path = self.path(symbol, timeframe)
        with file_lock(path):
            existing = _canonical_cache_frame(timeframe, self.load(symbol, timeframe))
            ratio = _moved_basis(timeframe, existing, df)
            if ratio is not None:
                log.warning(
                    "%s %s: price basis moved (overlap open ratio %.4g); "
                    "replacing the cached bars with the incoming frame",
                    symbol, timeframe, ratio)
                self.rebased.add((symbol, timeframe))
                merged = df
            else:
                merged = pd.concat([existing, df])
            merged = merged[
                ~_period_keys(timeframe, merged.index).duplicated(keep="last")
            ].sort_index()
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            try:
                merged.to_parquet(tmp)
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
        return len(df)

    def coverage(self, symbol: str, timeframe: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """(first_ts, last_ts) of the cached bars, or None if nothing is cached."""
        p = self.path(symbol, timeframe)
        if not p.exists():
            return None
        idx = pd.read_parquet(p, columns=[]).index
        if len(idx) == 0:
            return None
        return idx[0], idx[-1]

    def load(self, symbol: str, timeframe: str) -> pd.DataFrame:
        p = self.path(symbol, timeframe)
        if not p.exists():
            return empty_bars()
        df = pd.read_parquet(p)
        if len(df.index) >= 3:
            try:
                df.index.freq = pd.infer_freq(df.index)  # None if irregular; parquet drops freq
            except ValueError:
                pass  # leave freq as None
        return df
