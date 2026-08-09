# Canonical daily cache session keys implementation plan

> **For implementers:** Follow this plan with test-driven development. Complete one task at a time, commit each task, and record the verification output in the SDD report.

**Goal:** Make every daily cache use one New York-midnight key per UTC session date, then release the behavior as core 0.4.1.

**Architecture:** Add one private frame normalizer at the `BarCache.upsert` merge boundary. It rewrites only `1d` indexes, normalizes existing and incoming frames before concatenation, and leaves the current lock, atomic replacement, and last-write-wins behavior intact. Release metadata and the lockfile then move together to 0.4.1.

**Tech stack:** Python 3.12, pandas, parquet, pytest, uv

**Design:** `docs/specs/2026-08-09-daily-cache-session-key-design.md`

---

## Global constraints

- Work only in `/Users/chrisdoan/git/nakagai-core-fix-304-daily-cache` on `fix/304-daily-cache`.
- Preserve unrelated work in every other checkout.
- Use `apply_patch` for source, test, and documentation edits.
- Do not add a read-side fallback, compatibility alias, provider-specific branch, or second normalization path.
- Do not alter intraday timestamps.
- Keep `docs/superpowers/` and `.superpowers/` ignored and untracked.
- Do not use an em dash in code, comments, docs, commits, or reports.
- Do not add a Codex co-author trailer.

## Task 1: Canonicalize daily sessions at the cache merge boundary

**Files:**

- Modify: `tests/test_cache.py`
- Modify: `nakagai/data/cache.py`

### Step 1: Add the daily-session regression tests

Add this helper near the imports in `tests/test_cache.py`:

```python
def _bars_at(stamps: list[str], closes: list[float], volumes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": closes,
            "high": [value + 1 for value in closes],
            "low": [value - 1 for value in closes],
            "close": closes,
            "volume": volumes,
        },
        index=pd.DatetimeIndex(stamps, name="ts"),
    )
```

Add five tests after `test_upsert_merges_and_overwrites_overlap`:

```python
def test_daily_upsert_collapses_summer_labels_to_new_york_midnight(tmp_path):
    cache = BarCache(tmp_path)
    mixed = _bars_at(
        ["2026-07-14T00:00:00Z", "2026-07-14T04:00:00Z"],
        [100.0, 200.0],
        [1_000.0, 2_000.0],
    )

    assert cache.upsert("SPY", "1d", mixed) == 2

    loaded = cache.load("SPY", "1d")
    expected = _bars_at(["2026-07-14T04:00:00Z"], [200.0], [2_000.0])
    pd.testing.assert_frame_equal(loaded, expected)


def test_daily_upsert_collapses_winter_labels_to_new_york_midnight(tmp_path):
    cache = BarCache(tmp_path)
    mixed = _bars_at(
        ["2026-01-14T00:00:00Z", "2026-01-14T05:00:00Z"],
        [100.0, 200.0],
        [1_000.0, 2_000.0],
    )

    cache.upsert("SPY", "1d", mixed)

    expected = _bars_at(["2026-01-14T05:00:00Z"], [200.0], [2_000.0])
    pd.testing.assert_frame_equal(cache.load("SPY", "1d"), expected)


def test_daily_upsert_keeps_incoming_row_for_same_session(tmp_path):
    cache = BarCache(tmp_path)
    cache.upsert(
        "SPY",
        "1d",
        _bars_at(["2026-07-14T04:00:00Z"], [100.0], [1_000.0]),
    )

    cache.upsert(
        "SPY",
        "1d",
        _bars_at(["2026-07-14T00:00:00Z"], [300.0], [3_000.0]),
    )

    expected = _bars_at(["2026-07-14T04:00:00Z"], [300.0], [3_000.0])
    pd.testing.assert_frame_equal(cache.load("SPY", "1d"), expected)


def test_daily_upsert_repairs_mixed_sessions_already_on_disk(tmp_path):
    cache = BarCache(tmp_path)
    cache.path("SPY", "1d").parent.mkdir(parents=True, exist_ok=True)
    _bars_at(
        [
            "2026-07-14T00:00:00Z",
            "2026-07-14T04:00:00Z",
            "2026-07-15T04:00:00Z",
        ],
        [100.0, 200.0, 300.0],
        [1_000.0, 2_000.0, 3_000.0],
    ).to_parquet(cache.path("SPY", "1d"))

    cache.upsert(
        "SPY",
        "1d",
        _bars_at(["2026-07-16T04:00:00Z"], [400.0], [4_000.0]),
    )

    expected = _bars_at(
        [
            "2026-07-14T04:00:00Z",
            "2026-07-15T04:00:00Z",
            "2026-07-16T04:00:00Z",
        ],
        [200.0, 300.0, 400.0],
        [2_000.0, 3_000.0, 4_000.0],
    )
    pd.testing.assert_frame_equal(cache.load("SPY", "1d"), expected)


def test_intraday_upsert_preserves_exact_timestamps(tmp_path):
    cache = BarCache(tmp_path)
    bars = _bars_at(
        ["2026-07-14T00:00:00Z", "2026-07-14T04:00:00Z"],
        [100.0, 200.0],
        [1_000.0, 2_000.0],
    )

    cache.upsert("SPY", "15m", bars)

    pd.testing.assert_frame_equal(cache.load("SPY", "15m"), bars)
```

### Step 2: Run the new tests and confirm the expected failures

Run:

```bash
uv run pytest tests/test_cache.py -k "daily_upsert or intraday_upsert_preserves" -v
```

Expected before implementation: the four daily tests fail because the cache either retains both labels or retains `00:00Z`. The intraday control passes.

If a daily test passes unexpectedly, stop and inspect whether the test is exercising the real `BarCache.upsert` path before continuing.

### Step 3: Add the one canonicalization helper

Change the schema import in `nakagai/data/cache.py` to:

```python
from nakagai.data.schema import EXCHANGE_TZ, empty_bars, validate_bars
```

Add this private helper above `MemoryBars`:

```python
def _canonical_cache_frame(timeframe: str, df: pd.DataFrame) -> pd.DataFrame:
    """Use one New York-midnight key per daily session date."""
    if timeframe != "1d" or df.empty:
        return df
    canonical = df.copy()
    session_dates = canonical.index.tz_convert("UTC").tz_localize(None).normalize()
    canonical.index = session_dates.tz_localize(EXCHANGE_TZ).tz_convert("UTC")
    canonical.index.name = "ts"
    return canonical
```

In `BarCache.upsert`, canonicalize the validated incoming frame before the path is resolved:

```python
df = _canonical_cache_frame(timeframe, validate_bars(df))
```

Inside the existing lock, canonicalize the existing frame before concatenation:

```python
existing = _canonical_cache_frame(timeframe, self.load(symbol, timeframe))
```

Do not modify `load`, `coverage`, `MemoryBars`, provider output, the lock scope, or the atomic write sequence.

### Step 4: Run focused cache verification

Run:

```bash
uv run pytest tests/test_cache.py -v
```

Expected: every cache test passes, including the lock, failed-write, and concurrent-upsert coverage.

### Step 5: Commit the behavior

Run:

```bash
git add nakagai/data/cache.py tests/test_cache.py
git commit -m "fix(data): canonicalize daily cache sessions"
```

## Task 2: Release core 0.4.1

**Files:**

- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `README.md`

The version constant and release prose are configuration and human-facing metadata. They do not need a permanent unit test. Verification must inspect the built wheel metadata and the refreshed lockfile.

### Step 1: Update the release metadata

Change the project version in `pyproject.toml` from `0.4.0` to `0.4.1`.

Add this release note immediately above the existing `0.4.0` entry in `README.md`:

```markdown
### 0.4.1

- Canonicalize daily cache rows to midnight New York by UTC session date, so
  mixed provider labels cannot retain duplicate rows for one market session.
```

Refresh the lockfile:

```bash
uv lock
```

### Step 2: Verify the lockfile and wheel version

Run:

```bash
rg -n 'name = "nakagai"|version = "0.4.1"' pyproject.toml uv.lock
uv build --wheel -o dist
uv run --no-project --with ./dist/nakagai-0.4.1-py3-none-any.whl python -c 'from importlib.metadata import version; assert version("nakagai") == "0.4.1"'
```

Expected: `pyproject.toml` and the root package entry in `uv.lock` report `0.4.1`; the wheel builds and its installed distribution metadata reports `0.4.1`.

### Step 3: Run the complete core gate

Run:

```bash
uv run pytest
```

Expected baseline: at least 1,005 tests pass with no failures. The known non-failing warning may remain.

### Step 4: Check the final diff for scope and prohibited artifacts

Run:

```bash
git status --short
git diff --check origin/main...HEAD
git diff --name-only origin/main...HEAD
git ls-files docs/superpowers .superpowers
rg -n $'\u2014|\u2013' nakagai/data/cache.py tests/test_cache.py README.md docs/specs/2026-08-09-daily-cache-session-key-design.md docs/plans/2026-08-09-daily-cache-session-key-plan.md
```

Expected: only the design, plan, cache implementation, tests, release note, project version, and lockfile are in scope. The ignored scratch query prints nothing. The punctuation query prints nothing.

### Step 5: Commit the release

Run:

```bash
git add pyproject.toml uv.lock README.md
git commit -m "release: prepare core 0.4.1"
```

## Plan-to-spec check

- The summer and winter tests prove both daylight-saving offsets.
- The survivor tests cover an existing mixed parquet and an incoming overwrite.
- The intraday control proves normalization is limited to `1d`.
- The implementation has one normalization seam inside the locked read, merge, and write path.
- Release verification checks the actual wheel metadata and the full core suite.
- Platform pinning, deployment, production backup and repair, issue closure, and cross-repository close-out belong to the later platform #304 plan after this core PR merges.
