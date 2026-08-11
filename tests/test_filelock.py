"""Concurrent parquet appends must not lose rows: a shared file is evidence."""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from nakagai.filelock import append_parquet, file_lock


def _append(args) -> None:
    path, worker = args
    append_parquet(Path(path), pd.DataFrame([{"worker": worker, "n": i} for i in range(10)]))


def test_append_parquet_creates_then_appends(tmp_path):
    p = tmp_path / "shared.parquet"
    append_parquet(p, pd.DataFrame([{"a": 1}]))
    append_parquet(p, pd.DataFrame([{"a": 2}]))
    assert pd.read_parquet(p)["a"].tolist() == [1, 2]


def test_concurrent_appends_across_processes_lose_nothing(tmp_path):
    """8 processes x 10 rows: without the lock, read-concat-write drops rows."""
    p = tmp_path / "shared.parquet"
    with ProcessPoolExecutor(max_workers=8) as pool:
        list(pool.map(_append, [(str(p), w) for w in range(8)]))

    df = pd.read_parquet(p)
    assert len(df) == 80, f"lost {80 - len(df)} rows to a write race"
    assert sorted(df["worker"].unique()) == list(range(8))
    assert df.groupby("worker").size().tolist() == [10] * 8


def test_append_parquet_leaves_no_temp_files(tmp_path):
    p = tmp_path / "shared.parquet"
    append_parquet(p, pd.DataFrame([{"a": 1}]))
    leftovers = [f.name for f in tmp_path.iterdir() if f.name.endswith(".tmp")]
    assert leftovers == []


def test_append_parquet_keeps_prior_data_when_the_write_fails(tmp_path, monkeypatch):
    p = tmp_path / "shared.parquet"
    append_parquet(p, pd.DataFrame([{"a": 1}]))

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError):
        append_parquet(p, pd.DataFrame([{"a": 2}]))
    monkeypatch.undo()
    assert pd.read_parquet(p)["a"].tolist() == [1]  # not truncated


def test_file_lock_times_out_rather_than_corrupting(tmp_path):
    target = tmp_path / "shared.parquet"
    pid = os.fork() if hasattr(os, "fork") else None
    if pid == 0:  # child: hold the lock past the parent's timeout
        with file_lock(target):
            import time
            time.sleep(2)
        os._exit(0)
    import time
    time.sleep(0.3)  # let the child acquire
    try:
        with pytest.raises(TimeoutError):
            with file_lock(target, timeout=0.3):
                pass
    finally:
        if pid:
            os.waitpid(pid, 0)
