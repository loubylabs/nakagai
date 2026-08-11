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

The digest is over EXACT values. Prices go in through repr(), which round-trips
a float bit-for-bit, and the sha256 is not truncated, because the requirement
is byte-identical trades: a rounded price and a shortened digest both admit a
change this test exists to refuse.

Digesting at full precision buys that strictness at a price, and the price is
paid in `_ohlcv`: every operation that builds the frame has to be bit-identical
on every machine, or the table records the architecture that derived it rather
than the behaviour it claims to pin. `np.sin` is not, which is what made this
proof pass locally and fail in CI on all nine specs at once.

Nothing is excluded from the digest, including the four identity fields
`PortfolioTrade` gained in Phase 1. That is a measured fact rather than a
convenience, and `test_the_identities_are_derived_and_never_generated` is what
holds it: every identity is a digest over the request and the signal, so a
moved identity always means a moved input and never a flaky run.
"""

import dataclasses
import hashlib
import json
from datetime import date, timedelta
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

    The trend was `np.sin` and is now a triangular wave, because `np.sin` is
    the one operation in this function that is NOT bit-portable: it goes to the
    platform's libm. Measured on numpy 2.5.1, the same call over 8840 points
    hashes 6f814b7b on arm64 Darwin and 88014336 on x86_64 Linux, while
    `linspace`, `cumsum` and every `default_rng` draw hash identically on both.
    That one difference reached every price, and CI went red on all nine specs
    with every trade COUNT matching, which is the signature of an input that
    moved under the arithmetic rather than a spec that changed.

    A triangle is +, -, * and floor on values a binary float represents
    exactly, so it rounds the same way everywhere. It reverses as often as the
    sine did and turns harder, which is if anything a better fixture for the
    turnaround-style users this frame exists to fire."""
    rng = np.random.default_rng(seed)
    n = len(idx)
    # Two reversals a cycle, matching what `cycles * pi` of sine gave.
    phase = np.arange(n, dtype="float64") * (cycles / (2.0 * n))
    trend = (4.0 * np.abs(phase - np.floor(phase) - 0.5) - 1.0) * trend_amp
    steps = rng.normal(0, 0.15, n)
    disp_rows = np.arange(displacement_every // 2, n, displacement_every)
    for row in disp_rows:
        direction = 1.0 if (row // displacement_every) % 2 == 0 else -1.0
        for k in range(min(3, n - row)):
            steps[row + k] += direction * rng.uniform(0.8, 1.4)
    close = 100 + trend + np.cumsum(steps)
    open_ = np.concatenate([[close[0] - 0.05], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.2, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.2, n))
    base_vol = 800 + rng.integers(0, 400, n).astype(float)
    spike_rows = rng.choice(n, size=max(1, n // vol_spike_every), replace=False)
    base_vol[spike_rows] *= rng.uniform(4, 8, size=len(spike_rows))
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


def _request(spec: dict, schedule: ReplaySchedule):
    intervals = schedule.base_intervals
    split = TRAIN_SESSIONS * SESSION_INTERVALS
    base = spec_base_digest(spec, core_vocabulary)
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
    request, base = _request(spec, schedule)
    registry = FrozenStrategyRegistry.from_definitions(
        (rules_definition(spec["name"], base, spec=spec,
                          vocabulary_factory=core_vocabulary),))
    declared = dependencies_for(request, registry).timeframes
    bars = PortfolioBars({key: frame for key, frame in frames.items()
                          if key[1] in declared})
    return request, run_portfolio(request, bars, registry, schedule)


# --------------------------------------------------------------- the digest


def _cell(v):
    """Deterministic, lossless, and total over PortfolioTrade's field types.

    repr() on floats rather than str(), because str() rounds and a digest
    that rounds cannot see a change below the rounding. Enums by name, so
    the digest does not depend on member order. Sequences elementwise, for
    setup_tags.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, Enum):
        return v.name
    if isinstance(v, (tuple, list)):
        return [_cell(x) for x in v]
    return str(v)


def _trade_digest(trades):
    """Every field on PortfolioTrade, discovered rather than listed.

    An earlier draft hand-listed seven fields, which left stop, target, qty,
    symbol, r_multiple, setup_tags, fees, mae and mfe uncompared: a trade
    whose target moved would have produced an identical digest, so hard rule
    3's "byte-identical trades" would have been proven over a subset while
    claiming the whole. Reading dataclasses.fields() also means a field added
    to PortfolioTrade later joins the digest with nobody remembering to come
    back here, which is how Phase 1's four new identity fields joined it.
    """
    names = [f.name for f in dataclasses.fields(PortfolioTrade)]
    rows = [{n: _cell(getattr(t, n)) for n in names} for t in trades]
    blob = json.dumps(rows, sort_keys=True)
    return len(rows), hashlib.sha256(blob.encode()).hexdigest()


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


def test_the_digest_notices_a_change_in_every_trade_field(market):
    """The digest's discriminating power, field by field.

    This is the guard that would have caught the seven-field draft. It fails
    with every offending field named, so a future narrowing of _trade_digest
    cannot pass quietly.

    Every field is checked before anything is asserted, rather than asserting
    inside the loop: an in-loop assert stops at the FIRST field that misses
    and reports only that one, which reads as a single oversight where the
    real fault is a whole class of fields left uncompared. Watched failing by
    hand-listing a subset again, and it reported every field left out.
    """
    _, result = replay(PROBE, market)
    assert result.trades, "need at least one real trade to perturb"
    base = result.trades[0]
    d0 = _trade_digest([base])[1]
    missed = []
    for field in dataclasses.fields(PortfolioTrade):
        before = getattr(base, field.name)
        bumped = _with_field(base, field.name, _perturb(before))
        if _trade_digest([bumped])[1] == d0:
            missed.append(field.name)
    assert not missed, (
        f"these fields do not reach the digest, so a change to them is "
        f"invisible: {missed}")


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
# frame is bit-identical everywhere, so these digests hold on x86_64 Linux and
# on arm64 Darwin alike; `_ohlcv` says why that took work.
#
# The counts are not carried over from the table this file held before engine
# Phase 1. That engine replayed each spec on its own timeframe with no account
# between the signal and the trade; this one drives a 15-minute base clock,
# funds every fill out of one account, and closes the window at its end. The
# numbers moved because the replay model moved, and the baseline run above is
# what attributes that rather than assuming it.
GOLDEN = {
    "catalog:macd_trend": (65, "1145523650f4f26ae7626224775c7f0173756f4c6d68c4d72457672210f69043"),
    "catalog:rsi_reversion": (7, "d25b3bcb6d1b3d8db0351fce1d12ccd229350088a772dee558df78ccbda026b3"),
    "catalog:sma_cross": (40, "c749f03f0a1165501fe89bde5578e9d7dfd5f7dfcc7b605fdb1c7d99d34496c9"),
    "fixture:bollinger_breakout": (4, "9de47e1009343fc8bd6bce9fb36d37e83248bdf3e0748dbf86947de5d2c3a1fb"),
    "fixture:discount_pullback": (23, "af178fedb4997a0bf36b62a427fcf5e0abc206cc6266d1bf2b196dc34b016489"),
    "fixture:ifvg_reversal": (65, "e55d9aa48c2cda2c7ccb382923b59b1d2e8f6dd59f762381d247db2c7cb4938d"),
    "fixture:ob_bounce": (55, "ba01cd1a8f70d51fdcfba4c70d7f5fb8d048c778f5f308942e7e91767cb24c7d"),
    "fixture:orb": (135, "d0c95a30dc43690f5f176cc65e1a68263305860983c2536bdffea481a9dbccad"),
    "fixture:sma_cross": (40, "c749f03f0a1165501fe89bde5578e9d7dfd5f7dfcc7b605fdb1c7d99d34496c9"),
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
        blob = json.dumps([[_cell(v) for v in row]
                           for row in frame.to_numpy().tolist()])
        lines.append(f"  input {key[1]:>3}: {len(frame)} rows, sha256 "
                     f"{hashlib.sha256(blob.encode()).hexdigest()[:16]}")
    for trade in result.trades[:2]:
        cells = {f.name: _cell(getattr(trade, f.name))
                 for f in dataclasses.fields(PortfolioTrade)}
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
