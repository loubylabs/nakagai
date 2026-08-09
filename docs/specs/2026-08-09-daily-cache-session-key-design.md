# Canonical daily cache session keys

**Date:** 2026-08-09

**Status:** Approved for implementation

**Platform issue:** [chrvsd/nakagai#304](https://github.com/chrvsd/nakagai/issues/304)

**Repositories:** `loubylabs/nakagai`, then `chrvsd/nakagai`

## Goal

Guarantee that a daily cache contains at most one row per US market session,
regardless of whether a provider labels that session at midnight UTC or midnight
New York. Repair the known production caches without changing their current
Alpaca IEX price and volume basis.

## Root cause

`BarCache.upsert` deduplicates exact timestamps. The historical YFinance writer
labelled a daily row at `00:00Z`; the current Alpaca writer labels the same summer
session at `04:00Z` and the same winter session at `05:00Z`. Both timestamps have
the same UTC calendar date, so they identify the same market session while
remaining distinct index values.

The provider transition on 2026-07-25 explains the bounded mixed window in the
production cache. Current cold pulls are clean and use Alpaca for every fetched
timeframe. This change is recurrence protection plus repair of rows already on
disk.

## Canonical key

For `timeframe == "1d"`, the UTC calendar date carried by each label is the
session date. Canonicalize that date to midnight in `America/New_York`, then
convert it to UTC:

- A July session becomes `04:00Z`.
- A January session becomes `05:00Z`.
- A legacy `00:00Z` row and a current `04:00Z` or `05:00Z` row on the same UTC
  date become one key.

This deliberately does not derive the session date by converting the legacy
timestamp to New York first. Doing that would turn a `00:00Z` label into the
previous New York calendar date.

The normalization belongs inside `BarCache.upsert`, the one read, merge, and
write seam. `validate_bars` remains timeframe-agnostic. Providers remain free to
return their native valid labels. Readers inherit a canonical cache without a
second cleanup or interpretation path.

## Merge and survivor rule

Normalize both the existing frame and the incoming frame before concatenation.
Keep the existing merge order and `keep="last"` rule:

1. Within an already mixed existing frame, sorting places `00:00Z` before
   `04:00Z` or `05:00Z`. The later current-provider row survives.
2. An incoming row follows the existing frame and therefore replaces the cached
   row for the same session, preserving the current upsert contract.
3. Intraday timeframes retain their exact timestamps and current behavior.

The pair lock, temporary parquet write, and atomic `os.replace` remain the only
write path. No compatibility wrapper or read-side fallback is added.

## Core implementation

`nakagai/data/cache.py` gains one private daily-index normalization helper and
calls it for the existing and incoming frames inside the existing pair lock.

Tests in `tests/test_cache.py` must prove:

- Mixed `00:00Z` and `04:00Z` summer rows collapse to one session and the
  `04:00Z` row's OHLCV survives.
- Mixed `00:00Z` and `05:00Z` winter rows collapse to one session and the
  `05:00Z` row survives.
- A new incoming row still replaces an existing row for its session.
- A later upsert repairs duplicate sessions already present elsewhere in the
  existing daily parquet.
- A 15-minute cache preserves its exact timestamps.
- The existing lock and atomic-write tests stay green.

Core releases as `0.4.1`. `README.md` records the behavior change. The merge to
core `main` runs the full suite, builds the standalone wheel, and publishes the
new version through the existing trusted-publishing workflow.

## Platform integration

After the core PR merges, the platform pins the exact merged core SHA in
`pyproject.toml` and refreshes `uv.lock`. Platform CI runs both the platform
suite and the core-integration job against that revision. The pin merge deploys
the API through the existing Fly workflow.

No platform cache-normalization implementation is added. Core owns the one
answer; the platform only consumes the new release.

## Production repair

Production repair is authorized by the owner on 2026-08-09. It runs only after
the platform deployment is healthy on the new core revision.

1. Read every production `*_1d.parquet` and enumerate files with more than one
   row on a UTC calendar date.
2. Stop if the observed shape differs materially from issue #304 or if any file
   cannot be read.
3. Create a dated backup directory on the Fly volume and copy every affected
   parquet into it with metadata preserved.
4. Verify every backup is readable and matches its source byte size before any
   source file changes.
5. For each affected symbol, load its daily frame and pass that frame through
   the deployed `BarCache.upsert`. This uses the locked, atomic production path
   and the survivor rule above.
6. Re-read each repaired file and verify:
   - one row per UTC calendar date;
   - every label is midnight New York, expressed as `04:00Z` or `05:00Z`;
   - each formerly duplicated session matches the later New York-anchored row
     from the backup exactly across OHLCV;
   - coverage retains the same first and last session dates.
7. Run a forced scan only if ordinary scan health did not already exercise the
   repaired cache after deployment. Verify scan health and one daily-bar API
   read for a formerly affected symbol.

If any post-repair assertion fails, restore only the affected files from the
dated backup with an atomic replace, verify they are readable, and stop. Keep
the backup until the repaired cache has survived the next regular session.

## Delivery order

1. Merge and publish the core `0.4.1` PR.
2. Merge the platform pin PR and confirm CI plus Fly deployment.
3. Perform and verify the production repair.
4. Comment on and close platform issue #304 with the core PR, platform PR,
   deployment, backup path, affected symbols, and verification results.

## Non-goals

- Changing providers or switching from Alpaca IEX volume.
- Reconstructing consolidated historical volume.
- Adding a second daily-cache implementation in the platform.
- Rewriting intraday timestamps.
- Deleting the production backup during this change.

## Acceptance

- Core 0.4.1 is published from a green core `main`.
- The platform pins the merged core revision and its full suite is green.
- Daily upserts cannot retain two rows for one UTC session date.
- Production affected caches contain one New York-midnight row per session.
- Former duplicate sessions retain the current Alpaca row exactly.
- The repair backup exists and is readable.
- Platform issue #304 is closed with evidence.
