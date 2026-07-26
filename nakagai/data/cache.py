"""Local parquet cache. Backtests read ONLY from here, offline and reproducible."""

import os
from pathlib import Path

import pandas as pd

from nakagai.data.schema import BAR_COLUMNS, validate_bars
from nakagai.filelock import file_lock


def empty_bars() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="ts")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in BAR_COLUMNS}, index=idx)


class MemoryBars:
    """BarCache-shaped, dict-backed frames, keyed by (symbol, timeframe).

    The permutation harness backtests hundreds of permuted copies per pair;
    round-tripping each copy through temp parquet was pure overhead. load()
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

    def path(self, symbol: str, timeframe: str) -> Path:
        return self.root / f"{symbol}_{timeframe}.parquet"

    def upsert(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        """Merge `df` into the cached pair, newest copy of a duplicate ts winning.

        Locked and atomic for the reason nakagai/filelock.py exists: this is a
        read-concat-write, and an unlocked one silently keeps only whichever
        writer finished last. The scan loop, a backfill, and the proving farm
        can all touch one pair, and a lost bar reads downstream as "this play
        does not trade much" rather than as data loss. append_parquet is not
        reusable here because it concatenates with ignore_index=True, and this
        cache is indexed by ts and dedupes on that index.
        """
        if "interpolated" in df.columns:
            df = df[df["interpolated"] != True]  # noqa: E712 (spec: fake gap-fill bars never enter the cache)
        df = validate_bars(df)
        path = self.path(symbol, timeframe)
        with file_lock(path):
            existing = self.load(symbol, timeframe)
            merged = pd.concat([existing, df])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
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
