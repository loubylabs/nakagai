"""Cross-process advisory file locking, and the read-modify-write it exists for.

A shared parquet file on disk is appended to by several processes at once: API
job threads, an MCP subprocess, a nightly CLI, a bar-cache upsert. A bare
read-concat-write loses whichever writer finishes first, and the files this
guards are evidence a guardrail reads, so a lost row is a policy problem rather
than only a data problem.
"""

import errno
import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


@contextmanager
def file_lock(target: Path, timeout: float = 120.0, poll: float = 0.05):
    """Hold an exclusive lock keyed on `target` (via a sidecar `.lock` file).

    Advisory and cooperative: only guards against processes that also call this.
    Blocks up to `timeout`, then raises TimeoutError rather than corrupting.
    The lock file is never unlinked; unlinking races with other waiters.
    """
    lock_path = target.parent / f".{target.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out after {timeout}s waiting for the lock on {target}"
                    ) from e
                time.sleep(poll)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def write_parquet_atomic(path: Path, df: pd.DataFrame, **to_parquet_kwargs) -> None:
    """Write `df` to `path` atomically: temp file alongside the target, then
    `os.replace()`. Caller holds `file_lock(path)`; this only prevents a
    crash mid-write from leaving a truncated file behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        df.to_parquet(tmp, **to_parquet_kwargs)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def append_parquet(path: Path, new: pd.DataFrame, timeout: float = 120.0) -> None:
    """Append `new` to the parquet at `path` under an exclusive lock.

    Writes to a temp file and os.replace()s it in, so a crash mid-write leaves
    the previous parquet intact rather than a truncated file that every
    subsequent read raises on.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path, timeout=timeout):
        combined = (
            pd.concat([pd.read_parquet(path), new], ignore_index=True)
            if path.exists()
            else new
        )
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            combined.to_parquet(tmp)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
