# nakagai

The deterministic, LLM-free core for rule-driven trading agents: a point-in-time
bar cache, a statistically honest walk-forward backtester (look-ahead prevention,
T+1 cash settlement, bar-permutation Monte Carlo), the RuleSpec strategy DSL, and
a screener compiler.

## What is here

- `data/`: `BarCache`/`MemoryBars` over local parquet, the `DataProvider` contract
  and its Alpaca implementation (single-symbol and batched multi-symbol), and a
  sync routine that keeps the cache current.
- `engine/`: the walk-forward backtester itself, point-in-time `MarketContext`
  assembly, T+1 cash settlement, run metrics, and the bar-permutation Monte Carlo null.
- `strategies/`: rule-based (`rules/`), boolean-composed (`composite/`), and
  ICT-flavored (`ict/`) strategies, plus a catalog loader that turns JSON specs into
  strategy classes.
- `screen/`: a conditions-only screener over the same RuleSpec grammar. Evaluation
  is deterministic and LLM-free; an optional English-to-spec compiler shares the
  `nlbuilder` extra with `nlbuilder/`, which installs `anthropic`.
- `nlbuilder/`: English-to-RuleSpec compilation via the Claude API, behind the
  optional `nlbuilder` extra (installs `anthropic`).
- `stats.py`: permutation p-values, bootstrap confidence intervals, and the
  decision-exact null harness for backtest results.
- `icir.py`: rank-IC / IR of rule-spec margins vs forward returns (the informational ICIR lens).
- `filelock.py`: cross-process advisory file locking for concurrent read-modify-write
  on shared result files.

## Quickstart

This builds a `BarCache`, loads one of the shipped example strategies, runs the
walk-forward engine over the cached window, and prints run metrics next to
buy-and-hold. No network, no credentials, no optional extras, and it prints the
same numbers every time: the engine's whole contract is that a backtest reads
the cache and nothing else. Run it from the repo root with
`uv run python quickstart.py` (or paste it into a REPL):

```python
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from nakagai.data.cache import BarCache
from nakagai.data.schema import TimeframeSet, validate_bars
from nakagai.engine.engine import Engine
from nakagai.engine.metrics import buy_and_hold_return, summarize
from nakagai.strategies.catalog import load_catalog

# 1. Generate a deterministic hourly series. Swap this block for
#    AlpacaProvider().fetch_bars("SPY", "1h", start, end) once you have
#    ALPACA_KEY_ID / ALPACA_SECRET_KEY; everything below is unchanged, which is
#    the point of the DataProvider seam.
rng = np.random.default_rng(0)
idx = pd.date_range("2024-01-01", periods=2000, freq="1h", tz="UTC", name="ts")
close = pd.Series(400 * np.exp(np.cumsum(rng.normal(0, 0.006, len(idx)))), index=idx)
prev = close.shift(1).fillna(close.iloc[0])
bars = validate_bars(pd.DataFrame({
    "open": prev,
    "high": np.maximum(close, prev) * 1.004,
    "low": np.minimum(close, prev) * 0.996,
    "close": close,
    "volume": 1_000_000.0,
}, index=idx))

# 2. Store it in a local BarCache: parquet on disk, offline after this.
cache = BarCache(Path(tempfile.mkdtemp()))
cache.upsert("SPY", "1h", bars)

# 3. Load a shipped example strategy from the catalog.
specs_dir = Path("nakagai/strategies/catalog/specs")
catalog = load_catalog(specs_dir)
strategy = catalog["sma_cross"]({})

# 4. Run the engine over the cached window.
tfs = TimeframeSet(driving="1h", deltas={"1h": pd.Timedelta(hours=1)})
engine = Engine(strategy, cache, "SPY", bars.index[0], bars.index[-1], tfs=tfs)
result = engine.run()

# 5. Print metrics next to buy-and-hold.
bh = buy_and_hold_return(bars, bars.index[0], bars.index[-1])
metrics = summarize(result, bh_return=bh)
print(f"trades: {metrics['n_trades']}, win_rate: {metrics['win_rate']:.2f}, "
      f"profit_factor: {metrics['profit_factor']:.2f}, total_return: {metrics['total_return']:.2%}, "
      f"bh_return: {metrics['bh_return']:.2%}")
```

Because the series is seeded, this prints the same line on every machine, which
makes it a usable smoke test as well as an example:

```
trades: 25, win_rate: 0.32, profit_factor: 0.92, total_return: -1.27%, bh_return: -28.77%
```

A trend follower run on a random walk is not supposed to make money, and it
doesn't. That is the example working, not failing: the engine's job is to tell
you that honestly. Point step 1 at real bars to see something worth judging.

Two details of the generated series matter if you change it. Position size comes
from `risk_pct` divided by the ATR stop distance, so a series with a low
price-to-volatility ratio asks for more shares than `equity0` can buy and every
entry is skipped, which reads as a silent zero-trade run. And the bars are
continuous hourly, with no session gaps, which is fine for the `1h` driving
timeframe here but is not what session-aligned daily logic expects.

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
(mean reversion) and `macd_trend.json` (momentum). `load_catalog(specs_dir)` turns
every JSON file in a directory like this one into a `RuleStrategy` subclass.

## The lab

`nakagai/lab/` searches strategy space and scores the winner honestly.

A **trial** is a mutated spec, not a parameter set: v2 specs declare no tunable
params, so the tunable surface is the spec JSON itself. `literal_trials` moves
the numeric literals inside one spec; `composite_trials` assembles catalog
plays into composites. Every mutant is validated before it is returned.

A **study** runs a frozen trial set. N is fixed when the study is built and
cannot grow, because the null below is computed for exactly that N.

The **null** is what makes a survivor mean anything. Running four hundred
trials and keeping the best one finds noise with a good story; the fix is to
replay the entire search on permuted bars and take the best across all trials,
which gives the exact distribution of "best of N when there is nothing there".

`cache` must be built over the same bars as `frames`, i.e. `cache =
MemoryBars(frames)`; otherwise the observed statistic and the null are scored
on different histories and the resulting p-value means nothing.

```python
from nakagai.data.cache import MemoryBars
from nakagai.lab import (StudySpec, best_of_n_null, literal_trials,
                         run_study, study_verdict)

trials = literal_trials(base_spec, n=60, seed=7)
study = StudySpec(trials=tuple(trials), symbols=("SPY",),
                  windows=tuple(windows), seed=7)

cache = MemoryBars(frames)
observed = run_study(cache, study, registry)
nulls = best_of_n_null(frames, study, registry, n_permutations=200)
verdict = study_verdict(observed.best.pf, nulls,
                        n_trades=observed.best.n_trades)
# {"p_value": 0.015, "survived": True, ...}
```

`n_trades` is the WINNING trial's ledger, not the sum across the trial set.
The verdict is a statement about one trial's PF, so the trade floor has to
apply to that same trial: eight trials making five trades each sum to forty
and sail past a floor of twenty, while the winner's own record is five trades
and is noise.

The permutation count sets p-value resolution: 200 permutations resolve to
0.005. It is also the entire compute cost, scaling as
`trials x symbols x windows x permutations`.

`tests/test_lab_calibration.py` is the module's real specification. It runs the
whole pipeline on bars with no exploitable structure and asserts the p-values
come out uniform, then runs it on bars with a real effect and asserts it is
found. Run it with `uv run pytest -m slow`. The gate was measured at 24
replicates, 4 trials by 16 permutations: it took about 24 minutes and the mean
p-value on pure noise came out 0.5074 against an expectation of 9/17
(approximately 0.5294) at this permutation count, while the positive control
detected the real effect at the permutation resolution floor.

In CI, the gate runs automatically only when a change touches the lab or the
core modules it depends on (see `.github/workflows/calibration.yml`);
otherwise it can be triggered by hand via `workflow_dispatch`.

## What is NOT here

This repo does not include the curated Playbook content (the hand-authored
strategy specs), the evidence store and proving pipeline, the intraday scanner, or
the hosted platform: API, web UI, and the mandate and approvals judgment layer.
The hosted product at nakag.ai is built on top of this core.

## Release notes

### 0.2.0

**Behavior change: `day_of_week` reads the weekday off the FRAME, not off a
label's clock.** Backtest output moves for any play using `day_of_week` on an
intraday frame; re-run anything that depends on it. The old predicate decided
which clock to read by looking for a midnight-UTC label, and the bar caches are
not regular-hours-only, so a 19:00 New York post-market bar carries exactly the
label a resampled daily bar carries and was read as the next day: Tuesday, for a
Monday evening. It answered wrong on one bar of a session and right on all the
others, which is the shape of divergence a spec author never catches. The
weekday is now the frame's to decide, per `strategies/rules/primitives.py`.

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
