# nakagai

The deterministic, LLM-free core for rule-driven trading agents: a point-in-time
bar cache, a statistically honest walk-forward backtester (look-ahead prevention,
T+1 cash settlement, bar-permutation Monte Carlo), the RuleSpec strategy DSL, and
a screener compiler.

## What is here

- `data/`: `BarCache`/`MemoryBars` over local parquet, `DataProvider` implementations
  for Alpaca and yfinance, agent-mediated Robinhood bar normalization (no network
  code), and a sync routine that keeps the cache current.
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

This builds a `BarCache` from real SPY bars, loads one of the shipped example
strategies, runs the walk-forward engine over the cached window, and prints
run metrics next to buy-and-hold. It needs network access to Yahoo Finance and
no optional extras. Run it from the repo root with `uv run python quickstart.py`
(or paste it into a REPL):

```python
import tempfile
from pathlib import Path

import pandas as pd
import yfinance as yf

from nakagai.data.cache import BarCache
from nakagai.data.schema import TimeframeSet, validate_bars
from nakagai.engine.engine import Engine
from nakagai.engine.metrics import buy_and_hold_return, summarize
from nakagai.strategies.catalog import load_catalog

# 1. Fetch and normalize hourly bars for SPY. data/yf.py's YFinanceProvider
#    ships daily bars only; the shipped example specs run on 1h, so this
#    pulls intraday bars straight from yfinance.
raw = yf.download("SPY", period="730d", interval="1h", auto_adjust=True, progress=False)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
bars.index = pd.DatetimeIndex(bars.index).tz_convert("UTC")
bars = validate_bars(bars)

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

A real run against live SPY data printed:

```
trades: 19, win_rate: 0.53, profit_factor: 2.05, total_return: 9.51%, bh_return: 71.48%
```

Exact numbers depend on the trailing window Yahoo Finance returns on the day you run
it.

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

## What is NOT here

This repo does not include the curated Playbook content (the hand-authored
strategy specs), the evidence store and proving pipeline, the intraday scanner, or
the hosted platform: API, web UI, and the mandate and approvals judgment layer.
The hosted product at nakag.ai is built on top of this core.

## Development

```bash
uv sync --all-extras
uv run pytest
```

`uv sync --all-extras` pulls in `anthropic` so the `nlbuilder` tests run too; the
rest of the package works fine without it.

## License

MIT
