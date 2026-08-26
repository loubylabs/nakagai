# nakagai

The deterministic, LLM-free core for rule-driven trading agents: a point-in-time
bar cache, a statistically honest portfolio replay (one shared cash account,
look-ahead prevention, T+1 cash settlement), the RuleSpec strategy DSL, and a
screener compiler.

## What is here

- `data/`: `BarCache`/`MemoryBars` over local parquet, the `DataProvider` contract
  and its Alpaca implementation (single-symbol and batched multi-symbol), and a
  sync routine that keeps the cache current.
- `engine/`: the portfolio replay itself. `run_portfolio(request, bars, registry,
  schedule)` is the only entry point: it replays one account across every
  selected play and symbol in one causal chronology, and returns one canonical
  result with the trades, the structured rejections, the account equity curve,
  an independently calculated benchmark, per play-symbol attribution, and the
  metrics. It is deterministic and side-effect free, so it reads no cache,
  writes no file, mints no identifier, and consults no installed calendar.
- `strategies/`: rule-based (`rules/`), boolean-composed (`composite/`), and
  ICT-flavored (`ict/`) strategies, plus `catalog_definitions`, which turns a
  directory of JSON specs into frozen `StrategyDefinition` values a registry
  bundle takes.
- `screen/`: a conditions-only screener over the same RuleSpec grammar. Evaluation
  is deterministic and LLM-free; an optional English-to-spec compiler shares the
  `nlbuilder` extra with `nlbuilder/`, which installs `anthropic`.
- `nlbuilder/`: English-to-RuleSpec compilation via the Claude API, behind the
  optional `nlbuilder` extra (installs `anthropic`).
- `stats.py`: poolable return moments, the deflated-Sharpe family (PSR, DSR,
  minimum track record length), and the Benjamini-Hochberg false-discovery
  control, which together are how a candidate is priced for how many candidates
  were tried.
- `filelock.py`: cross-process advisory file locking for concurrent read-modify-write
  on shared result files.

## Quickstart

This builds one schedule, one frame of bars, one frozen registry, and one
request, then replays them through the single public entry point and prints the
result's metrics. No network, no credentials, no optional extras, and it prints
the same line every time: the whole contract is that a replay reads its four
arguments and nothing else.

It is longer than a one-liner on purpose. Core does not discover strategies,
resolve a cache, or decide what a trading session is; a caller states all of it,
which is what makes two runs of the same request byte-identical wherever they
run.

```python
import dataclasses
from datetime import date, timedelta

import numpy as np
import pandas as pd

from nakagai.engine import (
    ARITHMETIC_VERSION,
    AccountPolicy,
    BenchmarkSpec,
    ExchangeScheduleIdentity,
    ExecutionPolicy,
    FeeSpec,
    FrozenStrategyRegistry,
    PlayRequest,
    PortfolioBars,
    PortfolioReplayRequest,
    ReplaySchedule,
    ReplayWindow,
    ScheduledBaseInterval,
    SlippageSpec,
    definition_digest,
    expected_candidate_id,
    expected_replay_id,
    rules_definition,
    run_portfolio,
    schedule_digest,
    spec_base_digest,
)

SESSIONS, PER_SESSION = 8, 26          # eight regular sessions of 15-minute bars
WARMUP = 2 * PER_SESSION               # the first two are warmup, the rest is tested

# 1. The schedule IS the clock. Core never consults an installed calendar: it
#    replays exactly the intervals it is handed, so early closes and holidays
#    enter as data. These eight weekdays open at 14:30Z.
intervals, day = [], date(2026, 1, 5)
while len({row.session_date for row in intervals}) < SESSIONS or not intervals:
    if day.weekday() < 5:
        opens = pd.Timestamp(f"{day}T14:30:00Z")
        intervals += [
            ScheduledBaseInterval(
                session_date=day, interval_ordinal=n,
                open_ts=opens + pd.Timedelta(minutes=15 * n),
                close_ts=opens + pd.Timedelta(minutes=15 * (n + 1)))
            for n in range(PER_SESSION)]
    day += timedelta(days=1)
draft = ReplaySchedule(
    identity=ExchangeScheduleIdentity(
        calendar_id="XNYS", calendar_version="exchange_calendars:4.5.6:nakagai-rth-v1",
        schedule_digest="0" * 64, timezone="America/New_York", base_timeframe="15m"),
    base_intervals=tuple(intervals), context_bars=())
schedule = dataclasses.replace(draft, identity=dataclasses.replace(
    draft.identity, schedule_digest=schedule_digest(draft)))

# 2. One frame per (symbol, timeframe), labeled at the scheduled opens. Swap
#    this block for real bars once you have them; nothing below changes.
rng = np.random.default_rng(0)
index = pd.DatetimeIndex([row.open_ts for row in schedule.base_intervals], name="ts")
close = pd.Series(400 * np.exp(np.cumsum(rng.normal(0, 0.004, len(index)))), index=index)
prev = close.shift(1).bfill()
bars = PortfolioBars({("SPY", "15m"): pd.DataFrame({
    "open": prev, "high": np.maximum(close, prev) * 1.002,
    "low": np.minimum(close, prev) * 0.998, "close": close, "volume": 1_000_000.0,
}, index=index)})

# 3. A registry is a frozen bundle of definitions. `rules_definition` builds one
#    over a RuleSpec; the base digest covers the spec and the grammar it is read
#    under, because one spec under two grammars is two strategies.
spec = {
    "version": 2, "name": "sma_cross", "timeframe": "15m",
    "long": {"all": [{"lhs": {"ind": "sma", "n": 10}, "op": "crosses_above",
                      "rhs": {"ind": "sma", "n": 30}}]},
    "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
             "target": {"kind": "rr", "rr": 2.0}},
}
base_digest = spec_base_digest(spec)
registry = FrozenStrategyRegistry.from_definitions(
    (rules_definition("sma_cross", base_digest, spec=spec),))

# 4. The request names the plays, the symbols, and the whole visible policy.
#    Every play carries the digest binding its definition to its own params, and
#    core refuses the replay if it does not recompute.
params: dict = {}
window = ReplayWindow(
    train_start=intervals[0].open_ts, train_end=intervals[WARMUP].open_ts,
    test_start=intervals[WARMUP].open_ts, test_end=intervals[-1].close_ts)
draft = PortfolioReplayRequest(
    request_version=1,
    replay_id="replay:" + "0" * 64, candidate_id="candidate:" + "0" * 64,
    batch_id="0198b1c2-3d4e-7f80-8123-456789abcdef",
    registry_digest=registry.registry_digest,
    plays=(PlayRequest(play_id="play-1", strategy="sma_cross",
                       definition_digest=definition_digest(base_digest, params),
                       params=params, priority=100),),
    symbols=("SPY",), window=window, schedule_identity=schedule.identity,
    ic_horizons=(1, 5, 20), ic_tail_end=window.test_end,
    account=AccountPolicy(starting_equity=10_000.0, risk_pct=0.01,
                          max_open_positions=5, max_positions_per_play_symbol=1,
                          settlement_model="cash_t1"),
    execution=ExecutionPolicy(
        arithmetic_version=ARITHMETIC_VERSION, fill_mode="pessimistic",
        slippage=SlippageSpec(bps=1.0, min_per_share=0.01),
        fees=FeeSpec(per_fill=0.0, per_share=0.0),
        funding_order="play_priority_symbol_signal", missing_bar_policy="strict"),
    benchmark=BenchmarkSpec(kind="equal_weight_request_symbols", symbol=None,
                            weighting="equal", rebalance="never"))
named = dataclasses.replace(draft, candidate_id=expected_candidate_id(draft))
request = dataclasses.replace(named, replay_id=expected_replay_id(named))

# 5. One call, one canonical result.
result = run_portfolio(request, bars, registry, schedule)
metrics = result.metrics
print(f"trades: {metrics.all_trades.n_trades}, "
      f"rejections: {metrics.n_rejections}, "
      f"total_return: {metrics.total_return:.2%}, "
      f"benchmark: {metrics.benchmark_return:.2%}")
print(f"digest: {result.result_digest}")
```

Because the series is seeded and the arithmetic is canonical, this prints the
same line on every machine, which makes it a usable smoke test as well as an
example:

```
trades: 3, rejections: 0, total_return: -2.89%, benchmark: -2.18%
```

A trend follower run on a random walk is not supposed to make money, and it
doesn't. That is the example working, not failing: the replay's job is to tell
you that honestly. Point step 2 at real bars to see something worth judging.

Two details matter if you change it. Position size is `risk_pct` of frozen
equity divided by the protective distance, floored to whole shares, so a series
with a low price-to-volatility ratio asks for more shares than the account can
buy and every entry is refused for cash, which shows up as structured
rejections rather than as a silent zero-trade run. And every declared frame must
carry exactly the labels the schedule declares, with no gaps and nothing past
the boundary: a missing scheduled bar refuses the whole replay rather than
shrinking it.

## The RuleSpec DSL

A RuleSpec is plain JSON: an entry condition tree for `long` and `short`, and a
`risk` block for the stop and target. Conditions compare an indicator or price
source against another indicator or a constant, with operators like
`crosses_above` and `crosses_below`; `all`/`any` groups combine them into
arbitrarily nested boolean trees. `nakagai.strategies.rules.validate_spec` is the
single source of truth for the grammar, so a spec that loads has already been
checked. Here is the shipped `sma_cross.json` example, abridged to the DSL
fields (catalog card metadata like `category` and `tags` omitted):

```json
{
  "title": "Moving average crossover",
  "description": "The classic trend follower: long when the fast SMA crosses above the slow SMA on the 1h chart, short on the cross down. ATR-sized stop, fixed reward:risk target.",
  "spec": {
    "version": 2,
    "name": "sma_cross",
    "timeframe": "1h",
    "long": {"all": [
      {"lhs": {"ind": "sma", "n": 20}, "op": "crosses_above", "rhs": {"ind": "sma", "n": 50}}
    ]},
    "short": {"all": [
      {"lhs": {"ind": "sma", "n": 20}, "op": "crosses_below", "rhs": {"ind": "sma", "n": 50}}
    ]},
    "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0}, "target": {"kind": "rr", "rr": 2.0}}
  }
}
```

Two more examples ship in `nakagai/strategies/catalog/specs/`: `rsi_reversion.json`
(mean reversion) and `macd_trend.json` (momentum). `catalog_definitions(specs_dir,
core_vocabulary)` turns every JSON file in a directory like this one into a
frozen `StrategyDefinition` ready to enter a registry.

### Named time windows

An aggregate indicator can read one immutable, recurring window row. The row
belongs to the injected `Vocabulary`, so the same name means the same timezone,
boundaries, recurrence, and confidence everywhere the spec travels. Core does
not register venue-specific house rows on its own. A caller composes the rows
it supports and passes that vocabulary through validation, evaluation, identity,
description, and Pine export.

```python
from datetime import time

from nakagai.strategies.rules import WindowSpec, core_vocabulary

vocabulary = core_vocabulary().with_windows(WindowSpec(
    name="london",
    tz="Europe/London",
    start=time(8),
    end=time(16, 30),
    recurrence="weekday",
    confidence="low_iex",
))
```

The grammar then composes ordinary aggregate terms with that row:

```json
{
  "version": 2,
  "name": "london-range",
  "timeframe": "15m",
  "long": {"all": [{
    "lhs": {"ind": "highest", "of": {"src": "high"}, "window": "london"},
    "op": ">",
    "rhs": {"ind": "lowest", "of": {"src": "low"}, "window": "london"}
  }]}
}
```

`highest` and `lowest` retain their rolling `n` form when `window` is absent.
In window mode, `n` and `window` are mutually exclusive. `first` and `last`
require a window because they have no rolling interpretation. An aggregate may
also select its own `tf`; its `of` expression is evaluated on that frame.

Current windows are half-open. Their aggregate is NaN while an occurrence is
active, becomes visible at the first bar on or after its close, and carries
until the next occurrence opens. A row with `confidence="low_iex"` keeps the
same arithmetic and adds the sparse US-equity extended-hours IEX disclosure to
spec readback, prompts, and generated Pine source.

## What is NOT here

This repo does not include the curated Playbook content (the hand-authored
strategy specs), the evidence store and proving pipeline, the intraday scanner, or
the hosted platform: API, web UI, and the mandate and approvals judgment layer.
The hosted product at nakag.ai is built on top of this core.

## Release notes

### 0.8.2 (2026-08-26)

Generated Pine now identifies `xnys_session` membership on the New York
regular-session clock while keeping window boundaries in the row's declared
timezone. This restores engine-to-Pine parity for UTC window rows.

### 0.8.1 (2026-08-26)

The window-axis parity gate now executes generated Pine through data gaps that
cross both lifecycle boundaries. It also covers the prior-day high over a
holiday gap and the prior-day low after an observed early-close session.
Runtime output is unchanged; the executable parity tests characterize existing
window lifecycle behavior.
Current documentation describes the hard retirement in terms of the generic
window model, without preserving obsolete grammar spellings.

### 0.8.0 (2026-08-26)

RuleSpec now has one composable `window` axis. Immutable `WindowSpec` rows live
in the strategy vocabulary; `highest`, `lowest`, `first`, and `last` evaluate
the same row contract in the engine and Pine compiler. Window scope participates
in canonical spec and vocabulary identity. A grammar that adds or changes rows
therefore produces new definition, replay, and trade identifiers even when a
spec body is untouched.

**Breaking:** the five bespoke opening-range and previous-session primitives
are removed. Register immutable rows and use `highest(high)`, `lowest(low)`,
or `last(close)` over the matching window. No alias, fallback, or compatibility
path remains.

Low-confidence rows identify sparse US-equity extended-hours IEX coverage in
natural-language prompts, spec readback, and generated Pine source. Confidence
is disclosure metadata and does not change arithmetic.

### 0.7.0 (2026-08-23)

`benjamini_hochberg` is now the false-discovery control for an unordered
search. `effective_n_trials` is removed because it derived a result from the
sequential lags of a list and changed when the same candidate set was
reordered.

Migrate search callers by passing the raw candidate count to
`deflated_sharpe_ratio` and using `benjamini_hochberg` to decide the search
verdict. There is no compatibility path for `effective_n_trials`.

The Benjamini-Hochberg false-discovery guarantee is conditional. It controls
the expected proportion of false discoveries under independent p-values and
the supported positive regression dependence on a subset (PRDS) conditions.
It does not guarantee control under arbitrary dependence. The search
calibration covers independent null candidates and a deterministic one-factor
positively correlated null representative of shared market data. Those
calibrations do not certify arbitrary production dependence, so callers must
establish a valid dependence condition or calibrate their own procedure.

### 0.6.2

Restores one check 0.6.1 dropped, and records why dropping it was wrong.

A numeric operand outside the float range is refused again. 0.6.1 removed that
check on the argument that it changed the verdict on a spec the grammar accepts,
and fixed the readback instead. The argument was wrong, and the evidence it
lacked is in this repo: `canon.canonical_expr` returns `float(node)` for every
numeric scalar, deliberately, because that is what makes `20` and `20.0` one
spec. A number outside the float range therefore has no canonical form, so no
`spec_hash`, so it can be neither stored nor identified. It was never a usable
spec, and accepting it only moved the failure.

Downstream is where that showed: a spec compiled on attempt one, then took the
hosted platform's save path to `OverflowError: int too large to convert to
float`, which is a 500 on a request the compiler had just called good. The
refusal belongs in the validator, which can say why.

The test is exactly `float()` succeeding, not `math.isfinite`. JSON has no
infinity literal but `1e309` parses to one, and an infinity HAS a canonical form
and a hash, so it is accepted. Refusing it would be a different rule with a
different reason, and smuggling that in under this one is how a guard comes to
refuse more than it can explain.

The readback's fallback from 0.6.1 stays. A describer is read by surfaces that
must not raise whatever reaches them, and it is reachable with a spec this
validator never saw, so the two are defense in depth rather than duplicates.

### 0.6.1

A validator that raises is not a validator. `validate_composite_spec`,
`validate_composite_blocks` and `resolve_config_refs` each looked a value up in
a caller mapping without checking it was a name first, and `strategy` and
`config` are whatever arrived in the JSON. A list or an object is unhashable, so
`name not in members` raised `TypeError: unhashable type: 'list'` out of
functions whose whole contract is to return a list of errors.

The NL builder's retry loop is the caller that could not survive it. Its entire
purpose is to hand the model its own validator's errors and ask again, and a
model can emit anything, which is why the loop exists. Downstream, a malformed
reply reached the HTTP boundary as a 503 rather than as the retry the model
would have acted on.

The fix is the whole class, not the three sites the first pass found. Every
mapping and set in the rule grammar is keyed by string, so membership over a
caller value goes through one `names()` helper that answers False for a
non-string rather than raising: timeframes, sources, math ops, comparison ops,
indicator names, primitive names, and the session-alignment walk. The composite
block layer checks the shape of `spec` and `blocks` before iterating.
`resolve_config_refs` checks the shape of the saved config it is about to
inline, not just the name it looked it up by. And `nlbuilder`'s `_parse` now
requires one JSON OBJECT: `json.loads` happily returns a list for `[]`, and the
loop read `.get` off it.

A value that names nothing is reported as unknown, which is what it is, so the
model gets an error it can act on rather than a dead request. `_parse` also
catches a bare `ValueError`, not only `JSONDecodeError`: `json.loads` raises the
former for an integer past the interpreter's digit limit, and a reply the model
could have corrected was escaping the loop.

Three more the retry loop could not survive, all reached end to end through
`compile_strategy` rather than by calling internals:

- A reply nested thousands of levels deep raises `RecursionError` inside the
  JSON decoder, which is not a `ValueError`.
- An absurdly nested vote tree recursed `_check_tree` to the interpreter's
  limit. There is a bound either way; the choice is between a stated one and
  the interpreter's, so `MAX_TREE_DEPTH` is 32 and the builder prompt states it
  beside `MAX_BLOCKS`. A validator stricter than its own prompt burns retries on
  a rule nobody stated.
- A number past the float range validated CLEAN and then raised `OverflowError`
  in the readback, one step after the loop stopped watching. The RENDERER is
  what could not cope, so that is what changed: `_expr_text` falls back to an
  exact `str`. Refusing the number at validation was the first repair and it
  changed the verdict on a spec the grammar accepts, which is not this
  release's business.

One more, and it is the one that mattered most for an injecting caller.
`_ONE_BAR_SESSION` explains core's own primitives, and the flag it explains,
`Term.driving_frame_intraday`, is settable by anyone injecting a vocabulary. The
validator runs with the caller's terms in it, so the subscript raised `KeyError`
on a term the prompt had just taught the model. It falls back to the flag's own
meaning, which is true of every term that sets it.

The two describers are NOT made total, and that is deliberate. Their contract is
a spec that validated, every shipped path validates before describing, and
asserting totality they do not claim would be an overclaim of the same kind this
release exists to remove. What they do guarantee is a top-level shape check and
a name the grammar does not define, since both can reach them without the
validator having seen them.

Two generative tests keep the class closed rather than the cases: every awkward
JSON value crossed with every site that reads one.

### 0.6.0

A breaking release, small in size and narrow in blast radius: it closes the last
two doors in the library that had not moved onto the 0.5.0 value model.

**Breaking: the NL builder takes catalog cards, not member classes.**
`compile_strategy(..., members=...)` is now `compile_strategy(..., plays=...)`,
and `render_system_prompt(members)` is `render_system_prompt(plays)`. `plays` is
the caller's declared world, keyed by the name a composite block may reference,
whose values are CARD metadata: `title`, `description`, and the bound `spec` a
timeframe is read off. That is exactly what `strategies.catalog.load_entries`
returns.

0.5.0 replaced member classes with `StrategyDefinition` values, which carry a
name, a digest, the functions a replay builds and grades with, and the member
tree a composite lowers onto. None of that is presentation. `nlbuilder.prompt`
was still reading
`DEFAULT_PARAMS`, `title` and `description` off each member, and
`validate_composite_blocks` was still reading `PARAMS`, so both raised
`AttributeError` on the very values `catalog_definitions` and
`composite_definition` produce. The NL builder was uncallable from the value
model it shipped alongside.

A `rules` key in `plays` declares the bespoke leg, the block kind that writes
its own RuleSpec inline. The prompt teaches that leg, puts it in the worked
example, and words its risk sentence from that one value, so a caller who
registers no such member is told every block names a catalog play rather than
being taught a syntax it cannot build. The example's block names are read from
`plays` too, for the same reason, and a caller who declared the leg and no
catalog gets an example built from two inline legs.

**`validate_composite_blocks` reads `members` for membership alone.** Anything
answering `in` will do, which is how its structural sibling has always read it.
The rule the `PARAMS` read stood for is unconditional now: a block that is not
the bespoke leg carries no params. A CATALOG definition binds its spec at
construction, so an override has nowhere useful to land, and the two ways it can
fail are both worse than a refusal here. `params.spec` is refused outright at the
factory, with `ReplayInputError: this definition already binds its rule spec`, so
that block would die mid-replay rather than at validation. Any other key is
carried into the strategy and never read, so the play runs untuned while its
author believes otherwise. That second half is tracked as chrvsd/nakagai#460.

Known limitation, recorded rather than guessed at. `rules_definition` also
builds UNBOUND definitions, whose spec legitimately travels in `params`, and the
bespoke leg is recognized by the literal name `rules`. So an unbound definition
registered under any other name is refused here even though its own factory
builds it. The class model refused it identically, because `Strategy.PARAMS` was
empty on every unbound adapter too. Closing it needs the caller to say which
members are unbound, which the signature does not carry.

### 0.5.0

An intentional pre-1.0 breaking release. Every replay entry point 0.4.x offered
is gone, the strategy contract is strict, the result is one canonical value, and
the arithmetic is stamped `2`. A consumer migrates before pinning this version:
there is no release in which both contracts work, and the hosted platform's
cutover is a migration rather than a version bump.

**Breaking: one public replay, and the singleton engine is gone.**
`run_portfolio(request, bars, registry, schedule)` replaces `Engine`, `run_one`,
and `run_grid`. It replays ONE cash account across every selected play and
symbol in one causal chronology, so two candidates that were each affordable
alone now contend for the same settled cash and the same position capacity, and
the result carries a real account equity curve instead of per-symbol rows
combined after the fact.

Removed with them: `nakagai.engine.engine`, `nakagai.engine.runner`,
`nakagai.engine.provenance`, `nakagai.engine.costs`, `nakagai.icir`,
`BacktestResult`, the singleton `Trade`, `summarize`, `buy_and_hold_return`,
process-grid expansion, core-owned result parquet, `FeeModel`,
`SlippageModel`, `PreloadedBars`, `FrozenStrategyRegistry.definitions`, and
`CompositeStrategy.bound`. There is no adapter and no alias: a caller migrates
to the new contract or stays on 0.4.x.

`FeeSpec` and `SlippageSpec` now price a fill themselves, so the request's own
policy is the model. Composite membership arrives one way, as member factories
passed to `CompositeStrategy(...)`. `nakagai.engine` exports the complete
contract and nothing else.

**Breaking: `FeeSpec.per_fill` replaces `FeeModel.per_trade`, and `charge(qty)`
prices exactly one fill.** The old model returned `2 * (per_trade + per_share *
qty)` from a single call, on the assumption that a fee is priced once per round
trip. The portfolio replay charges the entry and the exit separately, as each
one happens, so the field is named for what it prices and the method returns
`per_fill + per_share * qty`. This changes fee arithmetic for any caller that
carried a non-zero `per_trade`: the same number passed as `per_fill` leaves the
round-trip total unchanged, while reading the old round-trip total as one fill
halves it. Zero stays zero, which is what the broker this core was built
against charges.

**Breaking: a strategy proposes values and never touches engine state.**
`Signal` moves out of `nakagai.strategies.base` into
`nakagai.engine.portfolio_types`, which owns the whole canonical contract, and
`nakagai.engine` exports it. It loses `entry` and gains `entry_ref`, the
deciding raw close its protective levels were bracketed against; a signal whose
reference is not that close is refused, so a play can no longer name a price the
replay did not decide on. `Strategy.on_bar` returns a `Sequence[Signal]` and
every element of it is a proposal: the singleton engine took `signals[0]` and
dropped the rest silently, while the order now carries into replay-wide signal
ordinals. `Strategy.manage` returns an immutable `ManagementDecision` in place
of the removed `PositionAction` enum, and the position it is handed is a frozen
`PositionView`, so a ratcheted stop or a replaced target travels back as a value
and assigning to the position raises. Every return is checked at the boundary
against a closed error taxonomy, and none of those errors becomes an empty
signal list: a strategy that refused, a strategy that returned something
invalid, and a strategy that saw nothing are three different observations, and a
replay that cannot tell them apart reports contention it never had.

**Canonical transport is core's, and only core's.** `canonical_replay_bytes` is
the one hashing encoding. Object keys sort lexically, a finite binary64 value
travels as its exact `float.hex()` inside a tagged object, a date travels
tagged, and a timestamp has one UTC spelling, so identical inputs produce
identical bytes no matter what order a mapping was built in or how a runtime
renders a decimal. Local, remote, and hosted agree byte for byte or they
disagree loudly. `result_digest` is taken over those bytes, and every identifier
formula lives beside it: `expected_candidate_id`, `expected_replay_id`,
`schedule_digest`, `definition_digest`, `spec_base_digest`, `trade_id`,
`rejection_id`. A caller recomputes rather than reimplementing, and core refuses
a request whose declared identifiers do not recompute. `encode_replay_*` and
`decode_replay_*` are a separate ordinary-JSON wire form for an API, a worker
envelope, or a database column; a receiver recomputes the canonical bytes from
the decoded values rather than trusting the transport text.

**Arithmetic version 2, and one result carries the whole reading.**
`ARITHMETIC_VERSION` is `"2"`, and the chronology, the cost model, and the
metric formulas are one arithmetic under it. A request declaring another version
is refused rather than reported under a label that does not describe how its
numbers were reached, so nothing stamped `1` is comparable to anything stamped
`2`. `PortfolioReplayResult` carries the trades, the structured rejections, the
account equity curve, an independently calculated benchmark, the portfolio
metrics, and one `PortfolioSlice` per play symbol holding that pair's own
counts, gross and net sums, fees, win rate, expectancy, and an `IcEstimate` at
each of the three horizons. The IC estimate is an in-sample diagnostic over the
window that was replayed, not a forecast, and `observations` is its load-bearing
field: a lens that never ran and a lens that ran and found nothing both report a
null coefficient, and only the count separates them.

**Breaking: one catalog door, and a grammar is a value.** `load_catalog` is
removed, with `RuleStrategy.bound` and the `RuleStrategy.VOCABULARY_FACTORY`
class attribute it was the only caller of. `catalog_definitions(specs_dir,
vocabulary_factory)` replaces it: a definition carries the name, binds the
immutable spec, records the grammar it is read under, and builds a plain
`RuleStrategy` fresh per candidate, so a catalog entry can no longer exist as
two minted classes in one process. `load_entries` is unchanged.

**Two digests, and the names say which is which.**
`spec_base_digest(spec, vocabulary_factory)` covers a strategy body and the
grammar it is read under, because one spec under two grammars is two
strategies. `definition_digest(base_digest, params)` binds that base to one
play's own params. They sit one keystroke apart and mean different things: a
registry freezes bases, a request declares definitions, and core refuses a play
whose declared definition digest does not recompute. Both names are new in this
release; 0.4.x had no digest of either kind.

**A definition's grammar now reaches the replay.** `StrategyDefinition` carries
`vocabulary_factory`, and the context a runtime decides through is built from
it. Before this, entries were always evaluated under the core grammar while the
IC lens graded the definition's own, so a play using an added term aborted and a
play using a redefined one was graded on a factor that did not produce its
trades. Nothing announced it: `vocabulary_digest` covers what a term declares,
not what it computes. Replays under the core grammar are byte-identical.

**Breaking: `pandas>=3` is the declared floor**, raised from `>=2.2`. The floor
is load-bearing rather than tidy. Copy-on-write is opt-in in pandas 2.x and
unconditional from 3.0, and `build_scheduled_context` hands a strategy zero-copy
prefixes of engine-owned frames. Under 2.x without copy-on-write, a strategy
writing into `ctx.bars[tf]` would write through into the replay's own prices,
and nothing would report it. Lowering this floor reintroduces that.

**New: a packaged per-term causality gate.**
`nakagai.strategies.rules.verify` adds `verify_term(term, bars)` and
`verify_vocabulary(vocabulary, bars)`, which ask of one term whether it reads
only rows at or before the row it answers for, by comparing a whole-frame
computation against the prefix computation at probe rows across the term's own
mandated argument sets. The answer is a `TermVerdict` carrying `CHECKED`,
`FAILED`, `EXEMPT` or `VACUOUS`, plus a machine-readable `cause` on a failure,
rather than a bare boolean: an end-anchored term returns a scalar with no
whole-frame series to index, and a condition-taking term cannot be called
without an evaluator, so under a boolean both would be indistinguishable from a
genuine pass. `tests/test_whole_frame_equivalence.py` pins causality for the
grammar from a hand-maintained list; this pins it for one term from the term's
own schema, which is what lets a term nobody wrote by hand be admitted or
refused. `reference_bars()` ships the frame the gate is meant to run on inside
the wheel rather than in `tests/`, because the consumer reaching this gate
lives in another repository and cannot get a test fixture. It merged to `main`
after 0.4.2 without a version bump, so 0.5.0 is the first release to carry it.

### 0.4.2

- Stamp every new replay run with arithmetic version `1` and fill mode
  `pessimistic`. These durable identities distinguish result semantics from
  package releases and source revisions. Existing replay arithmetic and trade
  output are unchanged.

### 0.4.1

- Canonicalize daily cache rows to midnight New York by UTC session date, so
  mixed provider labels cannot retain duplicate rows for one market session.

### 0.4.0

**Breaking: `nakagai.stats.pf_from_trades` and `PF_CLAMP` are removed.** They
computed a pooled profit factor over a trade ledger. The lab was their only
caller, and with the lab gone in 0.3.0 nothing reached them: not core, not the
hosted platform, which derives profit factor from its own gross sums. The
module no longer imports pandas.

**Breaking: `run_one` loses its `icir` keyword.** It opted a caller out of the
ICIR lens, and it had exactly two callers, the permutation harness and the
frontier open-window snapshots. Both were retired, so the flag has been dead
in production for some time and only a test still set it. The lens itself is
untouched and still runs for rule specs, still abstains to empty fields for
everything else, and still degrades to empty rather than killing a run row.

**Breaking: `Engine.slippage_for` is removed.** A one-line accessor over
`SlippageModel.per_share`, added so callers could ask the engine what it would
charge without reaching into the model. No caller ever did. Its only reference
was a test asserting the method exists, which is a test that cannot fail for
any reason worth catching, so it went too.

This release is also the first to carry everything merged since 0.3.0, which
shipped without a version bump: the deflated-Sharpe family, the injected
vocabulary reaching composites, and the session-open fix.

### 0.3.0

**Breaking: `nakagai.lab` is removed.** The module searched strategy space and
scored the winner against a best-of-N permutation null. It shipped in 0.2.0
with one consumer, the hosted platform's study subsystem, and that subsystem
was retired; nothing has imported the lab since. Gone with it: `Site`,
`Trial`, `composite_trials`, `literal_trials`, `mutable_sites`, `spec_hash`,
`best_of_n_null`, `study_verdict`, `StudyResult`, `StudySpec`, `TrialResult`,
`run_study`, `trial_pf`, and the `Calibration` workflow that gated them.

Note that `spec_hash` also exists, unrelated and unaffected, at
`nakagai.strategies.rules.canon.spec_hash`. Only the lab's is gone.

**Breaking: the bar-permutation Monte Carlo null is removed with it.** Gone:
`nakagai.engine.permutation` entirely (`permute_bars`, `permutation_seed`) and
`nakagai.stats.permutation_pvalue`. These were the two halves of one feature,
generating null price series and scoring an observation against them, and the
lab was the only thing that ever called either. Permutation testing is no
longer part of what this core does.

`nakagai.stats` keeps `pf_from_trades` and `PF_CLAMP`, and its module docstring
no longer describes it as permutation-test math.

The question the lab answered, "is this survivor real or did I just search
hard enough to find noise", is not being abandoned; it is moving to the
deflated-Sharpe family, which prices the same overfitting risk from the trial
count directly rather than by replaying the search on permuted bars.

### Session corrections (2026-08-06)

**Behavior change: every session-scoped term is anchored on the 09:30 bell and
scoped to regular hours.** Backtest output moves for any play reading the
opening range, elapsed session time, previous-session levels, `gap_pct`, or
`vwap`; re-run anything that depends on them. The bar caches are not
regular-hours-only, and these rules grouped a New York calendar date and treated
its first row as the session's start, which is ordinarily an 08:00 pre-market
print. The opening range was therefore a thin band nobody trades, elapsed
session time ran an hour and a half fast, previous-session levels came from
off-hours extremes, a gap was measured from 19:45 to 08:00, and session VWAP
was set by pre-market volume. A session now runs `[09:30, 16:00)` on the
exchange wall clock, from
`nakagai/data/schema.py`, and a bar before the bell reads NaN rather than a
value a condition would act on. A daily frame is unaffected: one row is its own
whole session.

**Behavior change: `day_of_week` reads the weekday off the FRAME, not off a
label's clock.** Backtest output moves for any play using `day_of_week` on an
intraday frame; re-run anything that depends on it. The old predicate decided
which clock to read by looking for a midnight-UTC label, and the bar caches are
not regular-hours-only, so a 19:00 New York post-market bar carries exactly the
label a resampled daily bar carries and was read as the next day: Tuesday, for a
Monday evening. It answered wrong on one bar of a session and right on all the
others, which is the shape of divergence a spec author never catches. The
weekday is now the frame's to decide, per `strategies/rules/primitives.py`.

### 0.2.0 (2026-08-04)

**New: a Pine v6 compiler for RuleSpec v2.** `compile_pine(spec, vocabulary)`
returns an indicator and a strategy, rendered from one lowering so the pair
cannot disagree about which bar decided; `lower_pine` returns the
target-neutral program underneath. Both are exported from
`nakagai.strategies.rules`, alongside `PineBundle` and `PineCompileError`.
Every export charts the engine's 15-minute driving cadence and requests a
play's own timeframe rather than charting it, so the script refuses any other
chart at runtime, and it requires extended trading hours for the same reason
the engine's own frames carry pre-market bars.

**Breaking: the catalog loaders require a vocabulary factory.**
`load_catalog(specs_dir)` becomes `load_catalog(specs_dir, core_vocabulary)`,
and the same for `load_entries`. Both are cached on their whole argument tuple,
so a defaulted call and an explicit one built two different strategy classes
over the same spec files, with `isinstance` quietly disagreeing and nothing
raising.

## Development

```bash
uv sync --all-extras
uv run pytest
```

`uv sync --all-extras` pulls in `anthropic` so the `nlbuilder` tests run too; the
rest of the package works fine without it.

## License

MIT
