"""Deterministic one-shot screen evaluation. No LLM anywhere in this path.

Providers, when given, sync full-tier symbols incrementally before evaluating
(only for the timeframes the spec references). Daily-tier symbols are never
synced here: the nightly cron owns their freshness. A sync failure never
aborts evaluation: it is noted on the row and cached bars still evaluate.
Every row carries its own bar_time so staleness is visible, never hidden."""

import pandas as pd

from nakagai.data.sync import fetch_incremental
from nakagai.engine.context import build_context
from nakagai.screen.spec import is_intraday, max_lookback, referenced_timeframes
from nakagai.screen.universe import DAILY, FULL
from nakagai.strategies.rules.exprs import eval_group


def _row(symbol: str, tier: str, matched=None, last_close=None,
         bar_time: str = "", note: str = "") -> dict:
    return {"symbol": symbol, "tier": tier, "matched": matched,
            "last_close": last_close, "bar_time": bar_time, "note": note}


def run_screen(spec: dict, tiers: dict[str, str], cache, now=None,
               providers: dict | None = None, sync_days: int = 60) -> dict:
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    tf = spec.get("tf", "1d")
    needed = referenced_timeframes(spec)
    intraday = is_intraday(spec)
    lookback = max_lookback(spec)
    rows: list[dict] = []
    errors: list[str] = []
    skipped = 0
    for sym in sorted(tiers):
        tier = tiers[sym]
        sync_note = ""
        if intraday and tier == DAILY:
            rows.append(_row(sym, tier,
                             note="intraday screen: no intraday data for this tier"))
            skipped += 1
            continue
        if providers and tier == FULL:
            for sync_tf, provider in providers.items():
                if sync_tf not in needed:
                    continue
                # A run-time sync only has sync_days of headroom by default,
                # which a long lookback (e.g. sma200 on 1d) can't fit in.
                # Intraday timeframes have no such lookback pressure here.
                fetch_days = max(sync_days, lookback * 2) if sync_tf == "1d" else sync_days
                try:
                    fetch_incremental(cache, provider, sym, sync_tf,
                                      now - pd.Timedelta(days=fetch_days), now)
                except Exception as e:
                    # A transient sync failure shouldn't cost us the cached
                    # bars we already have: keep going and let the row's
                    # note carry the failure so cached bars still evaluate.
                    note = f"sync failed: {e}"
                    errors.append(f"{sym}: {note}")
                    sync_note = note
        try:
            ctx = build_context(cache, sym, now)
            bars = ctx.bars[tf]
            if bars.empty:
                note = f"no {tf} bars cached"
                rows.append(_row(sym, tier,
                                 note=f"{sync_note}; {note}" if sync_note else note))
                skipped += 1
                continue
            if len(bars) < lookback:
                note = (f"only {len(bars)} bars cached; the screen's "
                        f"longest indicator needs {lookback}")
                rows.append(_row(sym, tier,
                                 note=f"{sync_note}; {note}" if sync_note else note))
                skipped += 1
                continue
            matched = bool(eval_group(spec["conditions"], ctx, bars, {}))
            rows.append(_row(sym, tier, matched=matched,
                             last_close=float(bars["close"].iloc[-1]),
                             bar_time=bars.index[-1].isoformat(),
                             note=sync_note))
        except Exception as e:  # partial failure: other symbols still screen
            errors.append(f"{sym}: {e}")
            note = f"error: {e}"
            rows.append(_row(sym, tier,
                             note=f"{sync_note}; {note}" if sync_note else note))
            skipped += 1
    rows.sort(key=lambda r: (r["matched"] is not True, r["symbol"]))
    return {"bar_close": now.isoformat(),
            "rows": rows,
            "universe": {"full": sum(1 for t in tiers.values() if t == FULL),
                         "daily": sum(1 for t in tiers.values() if t == DAILY),
                         "skipped": skipped},
            "errors": errors}
