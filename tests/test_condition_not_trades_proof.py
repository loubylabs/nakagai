"""Hard rule 3's core half: every existing catalog spec and fixture produces
byte-identical trades before and after node 03.

The instrument is the one public replay door, `run_portfolio`. Engine Phase 1
retired the singleton `Engine` this proof used to drive, so the harness below
builds what that door now requires: an exchange schedule, its context bars, and
one frame per declared timeframe. Everything it builds is derived from two
constants and a seed, so "deterministic and committed" is what the fixed seeds
in this file mean, and a frame randomized per run would make this test's own
result irreproducible.

A comparison that compares nothing is the failure mode this guards against: a
spec producing zero trades before and after passes an equality assertion while
proving nothing. Every one of the nine specs below is asserted to produce at
least one trade on this frame; none needed a named exemption.

The comparison has two halves, because the fields answer to two authorities.
The fourteen fields that are not arithmetic go into a sha256 and are compared
EXACTLY: the timestamps, direction, quantity, ordinals, exit reason, tags and
the four identity digests Phase 1 added. Those are what a grammar regression
moves, since a spec that evaluates differently fires on a different BAR. The
twelve float fields are compared per trade and per field against a committed
reference with a relative tolerance, because the last digits of a computed
price are not reproducible across architectures and a full-precision table
would pin the machine that derived it. `_FLOAT_TOLERANCE` carries the
measurement that forced this, including the experiment that ruled the fixture
out as the cause.

The identity fields are safe on the exact side, which is measured rather than
assumed: `test_the_identities_are_derived_and_never_generated` shows each is a
digest over the request and the signal, so a moved identity always means a
moved input and never a flaky run.
"""

import dataclasses
import hashlib
import json
import math
from datetime import date, time, timedelta
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nakagai.engine.bars import PortfolioBars
from nakagai.engine.canonical import (
    definition_digest,
    expected_replay_id,
    schedule_digest,
    trade_id,
)
from nakagai.engine.portfolio_types import (
    PlayRequest,
    PortfolioTrade,
    ReplaySchedule,
    ReplayWindow,
    ScheduledContextBar,
)
from nakagai.engine.registry import (
    FrozenStrategyRegistry,
    dependencies_for,
    rules_definition,
    spec_base_digest,
)
from nakagai.engine.replay import run_portfolio
from nakagai.strategies.catalog import load_entries
from nakagai.strategies.rules import core_vocabulary
from nakagai.strategies.rules.windows import WindowSpec
from tests.portfolio_fixtures import (
    base_identity,
    base_request,
    daily_session_dates,
    frictionless_execution,
    replay_account,
    scheduled_labels,
    session_intervals,
)

CATALOG_SPECS = (Path(__file__).resolve().parents[1]
                 / "nakagai" / "strategies" / "catalog" / "specs")
FIXTURE_SPECS = Path(__file__).resolve().parent / "fixtures" / "rules"

SYMBOL = "SPY"
PLAY_ID = "play-p"
# The spec the two machinery guards below drive. Named rather than read off
# GOLDEN, because both ask whether the DIGEST can see a change and neither is
# about the table: reading `sorted(GOLDEN)[0]` made them raise IndexError on an
# underived table, which reads as a broken guard where the real fault is one
# missing derivation. `test_the_frame_covers_every_spec_in_the_golden_table` is
# the one guard that owns GOLDEN's emptiness, and it is not parametrized.
PROBE = "catalog:macd_trend"
TZ = "America/New_York"
SESSION_OPEN = "09:30"
SESSION_INTERVALS = 26  # 09:30 to 16:00 Eastern, on the 15-minute base clock

# 340 weekday sessions, warming up on the first 210. The warmup is set by the
# hungriest spec rather than by taste: bollinger_breakout reads sma(200) on
# DAILY bars, so it cannot produce a signal until 200 sessions have closed, and
# a shorter train range would leave it with zero trades and this file would
# prove nothing about it. The 130 sessions that remain are what every other
# spec trades over.
#
# The test range is sized to make every spec fire, and not beyond it. A
# regression in the grammar shows up on the FIRST bar it touches, so a longer
# run buys more trades rather than more discrimination, and the nine replays
# below already compare 434 of them field by field. Doubling the range triples
# this module's runtime, because every evaluation re-reads its own prefix.
SESSIONS = 340
TRAIN_SESSIONS = 210


def _orb_fixture_vocabulary():
    """The frozen ORB fixture's grammar, identical to its Pine fixture row."""
    return core_vocabulary().with_windows(WindowSpec(
        "ny_open_30",
        "America/New_York",
        time(9, 30),
        time(10),
        "xnys_session",
        "standard",
    ))


def _vocabulary_factory(name: str):
    return _orb_fixture_vocabulary if name == "fixture:orb" else core_vocabulary


# ------------------------------------------------------------ the schedule


def _session_open(session: date) -> pd.Timestamp:
    """The session's first base open, as an instant.

    Built from Eastern wall clock rather than from a UTC literal, because a
    year of sessions crosses both 2026 transitions and 09:30 in New York is
    14:30Z in January and 13:30Z in July. A schedule that fixed the UTC hour
    would put half its sessions an hour off their own opening bell.
    """
    return pd.Timestamp(f"{session.isoformat()} {SESSION_OPEN}",
                        tz=TZ).tz_convert("UTC")


def _local(session: date, hour: int) -> pd.Timestamp:
    """An Eastern wall-clock hour on `session`, as an instant."""
    day = session + timedelta(days=hour // 24)
    return pd.Timestamp(f"{day.isoformat()} {hour % 24:02d}:00",
                        tz=TZ).tz_convert("UTC")


def _covers(intervals, start, end):
    """The scheduled base intervals a context period covers.

    The same overlap `_require_session_coverage` walks: an interval is covered
    when it closes after the period opens and opens before the period ends.
    """
    return [row for row in intervals if row.close_ts > start and row.open_ts < end]


def _fresh_at(intervals, period_end, base_step=pd.Timedelta(minutes=15)):
    """The scheduled base close in `[period_end, period_end + one base bar)`.

    None where there is none, which is what a period ending on a session's last
    close would produce if the next session opened more than one base bar away.
    """
    for row in intervals:
        if row.close_ts < period_end:
            continue
        return row.close_ts if row.close_ts < period_end + base_step else None
    return None


def _hourly_bars(intervals, blocks):
    """One bar per UTC hour that covers at least one scheduled interval.

    A 1h label is a cached UTC left edge, so the grid is a UTC one: a session
    opening at 14:30Z has its first bar labeled 14:00Z, and it covers the two
    intervals that open inside that hour.
    """
    rows = []
    for session, block in blocks:
        edge = block[0].open_ts.floor("h")
        while edge < block[-1].close_ts:
            end = edge + pd.Timedelta(hours=1)
            if _covers(block, edge, end):
                rows.append(ScheduledContextBar(
                    timeframe="1h", session_date=session, label_ts=edge,
                    period_start=edge, period_end=end, available_at=end,
                    fresh_context_at=_fresh_at(intervals, end),
                    source="fetched_left_edge"))
            edge = end
    return rows


def _four_hour_bars(intervals, blocks):
    """The Eastern-midnight-anchored buckets a regular session falls inside.

    Two of the six, and which two is a property of the session rather than a
    choice: 08:00 to 12:00 Eastern holds the morning and 12:00 to 16:00 holds
    the afternoon, and the other four buckets cover no scheduled interval at
    all. Both edges are built from wall clock, so each period is four hours on
    the exchange's clock whatever the offset is that week.
    """
    rows = []
    for session, block in blocks:
        for hour in range(0, 24, 4):
            start, end = _local(session, hour), _local(session, hour + 4)
            if not _covers(block, start, end):
                continue
            rows.append(ScheduledContextBar(
                timeframe="4h", session_date=session, label_ts=start,
                period_start=start, period_end=end, available_at=end,
                fresh_context_at=_fresh_at(intervals, end),
                source="derived_1h_et_midnight"))
    return rows


def _daily_bars(blocks):
    """One bar a session, spanning it, released at the NEXT session's open.

    The last session has no next session and therefore no release, which is
    why the loop stops one short rather than emitting a bar the schedule
    validator would refuse.
    """
    rows = []
    for position, (session, block) in enumerate(blocks[:-1]):
        following = blocks[position + 1][1]
        rows.append(ScheduledContextBar(
            timeframe="1d", session_date=session,
            label_ts=_local(session, 0),
            period_start=block[0].open_ts, period_end=block[-1].close_ts,
            available_at=following[0].open_ts,
            fresh_context_at=following[0].close_ts,
            source="session_aligned"))
    return rows


def build_schedule(sessions: int | None = None) -> ReplaySchedule:
    """A full regular-session schedule and every context bar it supports.

    Context bars are emitted in `CONTEXT_TIMEFRAMES` order, which the schedule
    validator requires of the tuple as a whole rather than of each session.
    """
    blocks = [(session, session_intervals(session, _session_open(session),
                                          SESSION_INTERVALS))
              for session in daily_session_dates(SESSIONS if sessions is None
                                                 else sessions)]
    intervals = tuple(row for _, block in blocks for row in block)
    context = tuple([*_hourly_bars(intervals, blocks),
                     *_four_hour_bars(intervals, blocks),
                     *_daily_bars(blocks)])
    draft = ReplaySchedule(identity=base_identity(), base_intervals=intervals,
                           context_bars=context)
    return dataclasses.replace(draft,
                               identity=base_identity(schedule_digest(draft)))


# --------------------------------------------------------------- the frames


def _ohlcv(idx, seed, trend_amp=3.0, vol_spike_every=40, displacement_every=14,
           cycles=6):
    """Session-shaped bars with volume dispersion (clears rvol > 3), a trend
    that reverses (fires bars_since / turnaround-style users), and periodic
    displacement moves sharp enough to leave real 3-candle FVGs and order
    blocks (both need a genuine gap between non-adjacent candles, which a
    smooth random walk essentially never produces on its own).

    Every operation here is bit-identical on every platform, and that is a
    requirement rather than a nicety: the golden table below is a committed
    digest of full-precision floats, so an input that differs in its last bit
    between two machines makes the table a fingerprint of the machine that
    derived it rather than of the engine's behaviour.

Two operations here were not, and both are gone.

    `np.sin` goes to the platform's libm: measured on numpy 2.5.1, the same
    call over 8840 points hashes 6f814b7b on arm64 Darwin and 88014336 on
    x86_64 Linux. The trend is a triangular wave instead, which is +, -, * and
    floor on values a binary float represents exactly, so it rounds the same
    way everywhere. It reverses as often as the sine did and turns harder,
    which is if anything a better fixture for the turnaround-style users this
    frame exists to fire.

    `rng.uniform(lo, hi)` is `lo + range * d`, one rounding where the compiler
    contracts it into an FMA and two where it does not: 254 of 1893 draws
    differed by a single ulp. Scaling a raw `random()` through separate array
    kernels cannot contract, because each kernel rounds before the next reads
    it, and it hashes identically on both.

    `linspace`, `cumsum`, `normal`, `integers` and `choice` were measured
    portable and are used as they are. What remains after all of that is NOT in
    this function: 45 of the 5208 float values these frames produce still
    differ across the two architectures, by at most 2.9e-14, and that residue
    is the engine's own arithmetic. `_FLOAT_TOLERANCE` records the experiment."""
    rng = np.random.default_rng(seed)
    n = len(idx)
    # Two reversals a cycle, matching what `cycles * pi` of sine gave.
    phase = np.arange(n, dtype="float64") * (cycles / (2.0 * n))
    trend = (4.0 * np.abs(phase - np.floor(phase) - 0.5) - 1.0) * trend_amp
    steps = rng.normal(0, 0.15, n)
    disp_rows = np.arange(displacement_every // 2, n, displacement_every)
    spans = [min(3, n - row) for row in disp_rows]
    # `rng.uniform(lo, hi)` is `lo + range * d`, which is one rounding where
    # the compiler contracts it into an FMA and two where it does not: 254 of
    # these 1893 draws differed by a single ulp between arm64 and x86_64.
    # Scaling a raw `random()` through separate array kernels cannot contract,
    # because each kernel rounds before the next one reads it. Measured
    # identical on both.
    kicks = rng.random(int(sum(spans)))
    kicks = kicks * 0.6
    kicks = kicks + 0.8
    rows = np.concatenate([row + np.arange(span)
                           for row, span in zip(disp_rows, spans)])
    signs = np.concatenate(
        [np.full(span, 1.0 if (row // displacement_every) % 2 == 0 else -1.0)
         for row, span in zip(disp_rows, spans)])
    steps[rows] += signs * kicks
    close = 100 + trend + np.cumsum(steps)
    open_ = np.concatenate([[close[0] - 0.05], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.2, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.2, n))
    base_vol = 800 + rng.integers(0, 400, n).astype(float)
    spike_rows = rng.choice(n, size=max(1, n // vol_spike_every), replace=False)
    spikes = rng.random(len(spike_rows))
    spikes = spikes * 4.0
    spikes = spikes + 4.0
    base_vol[spike_rows] *= spikes
    index = pd.DatetimeIndex(list(idx), tz="UTC", name="ts")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": base_vol}, index=index,
                        dtype="float64")


# One seed a timeframe, and each frame is generated in its own right rather
# than resampled from another. The four are inputs, not a claim that one
# market produced all of them: what this file measures is that a spec reads
# the same numbers before and after node 03, and an arbitrary fixed frame
# answers that as well as a consistent one would.
_FRAME_SHAPES = {
    "15m": dict(seed=7),
    "1h": dict(seed=11, trend_amp=4.0),
    "4h": dict(seed=17, trend_amp=8.0, vol_spike_every=20,
               displacement_every=10, cycles=4),
    "1d": dict(seed=13, trend_amp=25.0, vol_spike_every=8,
               displacement_every=6, cycles=2),
}


def build_frames(schedule: ReplaySchedule, boundary: pd.Timestamp) -> dict:
    """One frame per timeframe, on exactly the labels the schedule declares."""
    return {
        (SYMBOL, timeframe): _ohlcv(
            scheduled_labels(schedule, timeframe, boundary), **shape)
        for timeframe, shape in _FRAME_SHAPES.items()
    }


# ---------------------------------------------------------------- the specs


def _specs():
    """The nine spec bodies this file replays, keyed by where they ship from.

    The shipped catalog goes through its own loader rather than through a
    glob, so a spec this proof covers is a spec the catalog actually serves:
    a file the loader would refuse is a file that never reaches a user, and
    reading the directory directly would quietly proof-cover it anyway.
    """
    out = {f"catalog:{name}": entry["spec"]
           for name, entry in load_entries(CATALOG_SPECS,
                                           core_vocabulary).items()}
    for path in sorted(FIXTURE_SPECS.glob("*.json")):
        out[f"fixture:{path.stem}"] = json.loads(path.read_text())
    return out


@pytest.fixture(scope="module")
def market():
    """The schedule and its frames, built once for all nine replays."""
    schedule = build_schedule()
    boundary = schedule.base_intervals[-1].close_ts
    return schedule, build_frames(schedule, boundary)


def _request(spec: dict, schedule: ReplaySchedule, vocabulary_factory):
    intervals = schedule.base_intervals
    split = TRAIN_SESSIONS * SESSION_INTERVALS
    base = spec_base_digest(spec, vocabulary_factory)
    return base_request(
        plays=(PlayRequest(play_id=PLAY_ID, strategy=spec["name"],
                           definition_digest=definition_digest(base, {}),
                           params={}, priority=100),),
        symbols=(SYMBOL,),
        window=ReplayWindow(
            train_start=intervals[0].open_ts,
            train_end=intervals[split].open_ts,
            test_start=intervals[split].open_ts,
            test_end=intervals[-1].close_ts,
        ),
        schedule_identity=schedule.identity,
        ic_tail_end=intervals[-1].close_ts,
        account=replay_account(),
        execution=frictionless_execution(),
    ), base


def replay(name: str, market):
    """One spec's replay, through the public door and nothing else.

    Shared by the golden table and by every guard below, so the two cannot
    disagree about how a spec is run.
    """
    schedule, frames = market
    spec = _specs()[name]
    vocabulary_factory = _vocabulary_factory(name)
    request, base = _request(spec, schedule, vocabulary_factory)
    registry = FrozenStrategyRegistry.from_definitions(
        (rules_definition(spec["name"], base, spec=spec,
                          vocabulary_factory=vocabulary_factory),))
    declared = dependencies_for(request, registry).timeframes
    bars = PortfolioBars({key: frame for key, frame in frames.items()
                          if key[1] in declared})
    return request, run_portfolio(request, bars, registry, schedule)


# --------------------------------------------------------------- the digest


# Two comparisons, because the fields answer to two different authorities.
#
# Everything that is not arithmetic is compared EXACTLY, through a digest: the
# timestamps, the direction, the quantity, the ordinals, the exit reason, the
# tags, and the identity digests. Those are what a grammar regression moves. A
# spec that evaluates differently fires on a DIFFERENT BAR, so its signal_ts
# moves, or its qty, or it stops existing; none of that is a rounding question
# and none of it is allowed a tolerance here.
#
# The floats are compared PER TRADE AND PER FIELD against a committed
# reference, with a relative tolerance. Every part of that sentence was paid
# for:
#
# - A tolerance rather than a digest, because `lo + range * d` is one rounding
#   where the compiler contracts it into an FMA and two where it does not.
#   Measured over these nine replays, 753 of 5208 float values differ between
#   arm64 Darwin and x86_64 Linux. A digest of full-precision floats therefore
#   pins the machine that derived it, which is how this proof passed locally
#   and failed in CI on all nine specs with every trade COUNT matching.
# - Not a digest of ROUNDED floats either. Rounding does not make near-equal
#   values equal: two values a single ulp apart can straddle the boundary and
#   print differently. Measured, twelve significant digits took the x86_64
#   failures from nine specs to four rather than to none.
# - PER TRADE AND PER FIELD, never a total. Per-field totals were tried and are
#   an aggregate floor: two trades whose entries move by +0.25 and -0.25 leave
#   every total untouched, so the proof would accept trades it exists to
#   refuse. A tolerance on a total is also not the tolerance it appears to be,
#   because it scales with the total: 1e-9 relative on ORB's entry sum of
#   ~12,795 is an absolute window of 1.3e-5, which one trade could absorb
#   entirely.
#
# The tolerance is forced rather than chosen, and that was established by
# trying to remove the need for it. `_ohlcv` was made bit-portable in two
# steps, and the residue was measured after each:
#
#   fixture as first written      753 of 5208 values differ, worst 3.4e-13
#   after the np.sin fix          (still every spec, via rng.uniform)
#   after the FMA-free draws       45 of 5208 values differ, worst 2.9e-14
#
# Forty-five differences survive a fixture with no non-portable operation left
# in it, so the source is the engine's own arithmetic and no amount of fixture
# care removes it. 1e-11 leaves about three hundred times headroom over that
# residue and seven orders of margin below any real change: a trade that moved
# moved to another bar, so its price differs in the second or third digit.
_FLOAT_TOLERANCE = 1e-11

FLOAT_REFERENCE = (Path(__file__).resolve().parent / "fixtures"
                   / "node03-trade-floats.json")


def _cell(v):
    """Exact and total over the field types the digest covers.

    Floats never reach this: `_float_fields()` routes them to the tolerant
    comparison and `_exact_fields()` is what this renders. A float arriving
    here is a partition bug rather than a value to render, and it says so.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        raise AssertionError(
            f"a float reached the exact digest: {v!r}. Floats are compared "
            f"with a tolerance by _float_mismatches, because their last digits "
            f"are not reproducible across architectures.")
    if isinstance(v, Enum):
        return v.name
    if isinstance(v, (tuple, list)):
        return [_cell(x) for x in v]
    return str(v)


def _float_fields():
    """The fields carrying arithmetic, discovered from the dataclass.

    Read off the annotations rather than a hand-list, so a field added to
    PortfolioTrade later joins the right side of the split with nobody
    remembering to come back here.
    """
    return tuple(f.name for f in dataclasses.fields(PortfolioTrade)
                 if f.type in ("float", float))


def _exact_fields():
    floats = set(_float_fields())
    return tuple(f.name for f in dataclasses.fields(PortfolioTrade)
                 if f.name not in floats)


def _trade_digest(trades):
    """Every NON-float field, discovered rather than listed.

    An earlier draft hand-listed seven fields, which left stop, target, qty,
    symbol, r_multiple, setup_tags, fees, mae and mfe uncompared: a trade
    whose target moved would have produced an identical digest, so hard rule
    3's claim would have been proven over a subset while claiming the whole.
    Reading dataclasses.fields() also means a field added to PortfolioTrade
    later joins the comparison with nobody remembering to come back here,
    which is how Phase 1's four new identity fields joined it.
    """
    names = _exact_fields()
    rows = [{n: _cell(getattr(t, n)) for n in names} for t in trades]
    blob = json.dumps(rows, sort_keys=True)
    return len(rows), hashlib.sha256(blob.encode()).hexdigest()


def _trade_floats(trades):
    """Every float field of every trade, in field order."""
    names = _float_fields()
    return [[getattr(t, name) for name in names] for t in trades]


def _float_mismatches(got, want):
    """Every field of every trade that moved by more than the tolerance.

    Returns them all rather than the first, so one run names the whole shape
    of a regression instead of sending its reader round the loop once per
    field.
    """
    if len(got) != len(want):
        return [f"trade count is {len(got)}, reference holds {len(want)}"]
    names = _float_fields()
    out = []
    for i, (grow, wrow) in enumerate(zip(got, want)):
        for name, g, w in zip(names, grow, wrow):
            if math.isclose(g, w, rel_tol=_FLOAT_TOLERANCE,
                            abs_tol=_FLOAT_TOLERANCE):
                continue
            rel = abs(g - w) / max(abs(g), abs(w), 1e-300)
            out.append(f"trade {i} {name}: {g!r} vs {w!r} (rel {rel:.2e})")
    return out


def _perturb(v):
    """A different value of the same type, for every type PortfolioTrade holds.

    Total by construction rather than by a chain of isinstance checks that
    silently returns the input for an unhandled type: a _perturb that handed
    back an unchanged value would make the field's assertion below fail for
    the wrong reason, and the reason matters, since the whole point is to
    distinguish "not in the digest" from "not actually changed".
    """
    if isinstance(v, bool):
        return not v
    if isinstance(v, Enum):
        others = [m for m in type(v) if m is not v]
        assert others, f"{type(v).__name__} has one member; cannot perturb"
        return others[0]
    if isinstance(v, float):
        return v + 1.5
    if isinstance(v, pd.Timestamp):
        return v + pd.Timedelta(minutes=1)
    if isinstance(v, int):
        return v + 1
    if isinstance(v, str):
        return v + "_x"
    if isinstance(v, tuple):
        return v + ("x",)
    raise AssertionError(f"_perturb has no case for {type(v).__name__}; add one")


def _with_field(trade: PortfolioTrade, name: str, value) -> PortfolioTrade:
    """A real PortfolioTrade carrying one changed field, built around __init__.

    `dataclasses.replace` cannot be used and the reason is not stylistic.
    `PortfolioTrade.__post_init__` refuses most of the perturbations above at
    construction: a direction outside {long, short}, an identity that is not a
    prefixed digest, an entry whose stop now sits on the wrong side of it. A
    replace-based guard would raise on those fields instead of measuring them,
    and a guard that raises tells you nothing about whether the digest would
    have noticed. Building the instance field by field is what makes the
    question askable at all, and the object is still exactly the type the
    digest reads.
    """
    clone = object.__new__(PortfolioTrade)
    for field in dataclasses.fields(PortfolioTrade):
        object.__setattr__(clone, field.name, getattr(trade, field.name))
    object.__setattr__(clone, name, value)
    return clone


def test_every_field_reaches_one_side_of_the_comparison(market):
    """The discriminating power of both halves, field by field.

    This is the guard that would have caught the seven-field draft, and it now
    also holds the SPLIT honest: a field that reached neither the digest nor
    the sums would be silently uncompared, and a field quietly moved from the
    exact side to the tolerant one would lose its exactness with nothing
    saying so. Every field is asserted to reach the side its type puts it on.

    Every field is checked before anything is asserted, rather than asserting
    inside the loop: an in-loop assert stops at the FIRST field that misses
    and reports only that one, which reads as a single oversight where the
    real fault is a whole class of fields left uncompared. Watched failing by
    hand-listing a subset again, and it reported every field left out.
    """
    _, result = replay(PROBE, market)
    assert result.trades, "need at least one real trade to perturb"
    base = result.trades[0]
    # Derived from the TRADE rather than from _float_fields(), which is the
    # whole point: _exact_fields() is defined as _float_fields()'s complement,
    # so asserting their union covers every field is true for any split at all,
    # including an empty one and one that swallows every field. An earlier
    # draft asserted exactly that and could not fail. This compares the
    # declared split against the types a real trade actually carries.
    observed = {name for name in (f.name for f in
                                  dataclasses.fields(PortfolioTrade))
                if isinstance(getattr(base, name), float)
                and not isinstance(getattr(base, name), bool)}
    assert set(_float_fields()) == observed, (
        f"the declared float split disagrees with what a real trade carries: "
        f"declared-not-float {sorted(set(_float_fields()) - observed)}, "
        f"float-not-declared {sorted(observed - set(_float_fields()))}")

    d0, f0 = _trade_digest([base])[1], _trade_floats([base])
    missed_exact, missed_float = [], []
    for name in _exact_fields():
        bumped = _with_field(base, name, _perturb(getattr(base, name)))
        if _trade_digest([bumped])[1] == d0:
            missed_exact.append(name)
    for name in _float_fields():
        bumped = _with_field(base, name, _perturb(getattr(base, name)))
        if not _float_mismatches(_trade_floats([bumped]), f0):
            missed_float.append(name)
    assert not missed_exact, (
        f"these fields do not reach the digest, so a change to them is "
        f"invisible: {missed_exact}")
    assert not missed_float, (
        f"these fields are not compared against the reference, so a change "
        f"to them is invisible: {missed_float}")


def test_the_identities_are_derived_and_never_generated(market):
    """Why the four Phase 1 identity fields are safe to digest.

    An identity that were minted per run would make this whole table flap: the
    digest covers every field, so a fresh `trade_id` on every replay would
    change it every time and the golden numbers would be unreproducible by
    construction. They are not minted. Every one is a digest over the request
    and the signal, so this asserts the formulas rather than trusting them, and
    a future change that generated an identity would fail HERE, naming the
    cause, instead of showing up as nine unexplained digest drifts.

    `replay_id` carries the vocabulary through `definition_digest`, which is
    the other half of the same point: it moves when the GRAMMAR moves. Node 03
    declares no new term and changes no term's schema, so it does not move, and
    the golden table can be compared across the node at all.
    """
    request, result = replay(PROBE, market)
    assert result.trades
    assert request.replay_id == expected_replay_id(request)
    for ordinal, trade in enumerate(result.trades):
        assert trade.replay_id == request.replay_id
        assert trade.trade_ordinal == ordinal
        assert trade.play_id == PLAY_ID
        assert trade.trade_id == trade_id(
            request.replay_id, trade.play_id, trade.symbol,
            trade.signal_ordinal)


# DERIVED, NOT TRANSCRIBED. Printed by running this module's harness over
# _specs() at core 992f55e with the node 03 production diff ABSENT, then
# re-printed on the node's own branch and confirmed identical. That second run
# is what makes the table a measurement of "this node moved no trade" rather
# than a stamp of "this node produced these numbers".
#
# It is also re-derived on a SECOND ARCHITECTURE. Every operation building the
# frame is bit-identical everywhere, which `_ohlcv` says took two fixes to get
# to; the exact half of the comparison therefore holds identically on x86_64
# Linux and arm64 Darwin, and the tolerant half absorbs what the ENGINE's
# arithmetic still differs by.
#
# The counts are not carried over from the table this file held before engine
# Phase 1. That engine replayed each spec on its own timeframe with no account
# between the signal and the trade; this one drives a 15-minute base clock,
# funds every fill out of one account, and closes the window at its end. The
# numbers moved because the replay model moved, and the baseline run above is
# what attributes that rather than assuming it.
GOLDEN = {
    "catalog:macd_trend": (65, "564180028e581a24a12df61791cd2ce3b82561b6983ecbb14ee569cef50d9114"),
    "catalog:rsi_reversion": (7, "ba3898c5b1a92fe6f559354fdf0c6c8c0096e4dc10f37e73df043f3e63c87e06"),
    "catalog:sma_cross": (40, "e40bab366bd80114bdd7d976335b295c67284d230d82c5a32150ab1ecf237103"),
    "fixture:bollinger_breakout": (4, "63366c3fe7501c81cd4195327311cc57697d78dfa939eedf48e11fbe62cd3e3b"),
    "fixture:discount_pullback": (23, "3ad81b3a64f670795d2ff9a85fc8b8f4222b564dd192b8a8abd4e5e089d681ab"),
    "fixture:ifvg_reversal": (65, "3dec72caaa3ea53b64b6f9be202df20b2280f853786f23272145301aa294ac43"),
    "fixture:ob_bounce": (55, "11ad81be9bfc3770ca11f516d3a81cfdf031376e840aa7e02c0c97f9b7404890"),
    "fixture:orb": (135, "71da11a9170c835f4b6fd0e29a22f0bef389f8e8356a196c3abe3ae6b6743e2f"),
    "fixture:sma_cross": (40, "e40bab366bd80114bdd7d976335b295c67284d230d82c5a32150ab1ecf237103"),
}


def test_the_frame_covers_every_spec_in_the_golden_table():
    """Also the floor that keeps an underived GOLDEN from reading green.

    test_every_spec_produces_the_golden_trades is parametrized over
    sorted(GOLDEN), so an empty table collects ZERO cases and pytest reports
    a pass having run nothing. This test is not parametrized, so it is the
    one that fails when the table has not been derived yet.
    """
    assert len(GOLDEN) == 9, f"GOLDEN has {len(GOLDEN)} entries, expected 9"
    assert set(_specs()) == set(GOLDEN)


def _diagnostic(name, market, result):
    """What a digest mismatch cannot say on its own: which numbers moved.

    A digest reports one bit. When it flips the next question is always whether
    the TRADES moved or the MACHINE did, and answering it needs the values, a
    fingerprint of the frames they were computed from, and the platform that
    computed them. Without those a red proof sends its reader to re-derive the
    table, which is the one response that cannot distinguish the two.
    """
    import platform

    _, frames = market
    lines = [
        f"{name}: the digest moved. n={len(result.trades)}, which MATCHES the "
        f"golden count if the assertion above passed.",
        f"  platform: {platform.machine()} {platform.system()} "
        f"python {platform.python_version()} "
        f"numpy {np.__version__} pandas {pd.__version__}",
    ]
    for key, frame in sorted(frames.items()):
        blob = json.dumps([[repr(v) for v in row]
                           for row in frame.to_numpy().tolist()])
        lines.append(f"  input {key[1]:>3}: {len(frame)} rows, sha256 "
                     f"{hashlib.sha256(blob.encode()).hexdigest()[:16]}")
    for trade in result.trades[:2]:
        cells = {name: _cell(getattr(trade, name)) for name in _exact_fields()}
        cells.update({name: repr(getattr(trade, name))
                      for name in _float_fields()})
        lines.append(f"  trade {trade.trade_ordinal}: "
                     f"{json.dumps(cells, sort_keys=True)}")
    return "\n".join(lines)


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_every_spec_produces_the_golden_trades(name, market):
    _, result = replay(name, market)
    n, digest = _trade_digest(result.trades)
    want_n, want_digest = GOLDEN[name]
    assert n >= 1, f"{name}: zero trades proves nothing"
    assert n == want_n, f"{name}: trade count moved from {want_n} to {n}"
    assert digest == want_digest, _diagnostic(name, market, result)

    reference = json.loads(FLOAT_REFERENCE.read_text())
    assert name in reference, (
        f"{name} has no committed float reference; derive it before this "
        f"test means anything")
    moved = _float_mismatches(_trade_floats(result.trades), reference[name])
    assert not moved, (
        f"{name}: {len(moved)} float field(s) moved by more than "
        f"{_FLOAT_TOLERANCE:g} relative, which is far above the 3.4e-13 two "
        f"architectures differ by.\n  " + "\n  ".join(moved[:12])
        + f"\n{_diagnostic(name, market, result)}")


def test_the_float_reference_covers_the_same_specs_as_the_table():
    """The fixture is half the comparison, so it gets the same floor.

    An empty or partial reference file would make the tolerant half compare
    nothing while the digest half stayed green, which is the same shape of
    hole the unparametrized GOLDEN floor exists to close.
    """
    reference = json.loads(FLOAT_REFERENCE.read_text())
    assert set(reference) == set(GOLDEN), (
        f"reference covers {sorted(set(reference) ^ set(GOLDEN))} differently "
        f"from the golden table")
    for name, rows in reference.items():
        assert len(rows) == GOLDEN[name][0], (
            f"{name}: reference holds {len(rows)} trades, table says "
            f"{GOLDEN[name][0]}")
        assert all(len(r) == len(_float_fields()) for r in rows), name
