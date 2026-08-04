"""Deterministic one-shot screen evaluation. No LLM anywhere in this path.

Providers, when given, sync each symbol incrementally before evaluating (only
for the timeframes the spec references). A sync failure never aborts
evaluation: it is noted on the row and cached bars still evaluate. Every row
carries its own bar_time so staleness is visible, never hidden."""

import pandas as pd

from nakagai.data.sync import fetch_incremental
from nakagai.engine.context import build_context
from nakagai.screen.spec import max_lookback, referenced_timeframes
from nakagai.strategies.rules.vocabulary import Vocabulary, resolve_vocabulary


def _row(symbol: str, matched=None, last_close=None,
         bar_time: str = "", note: str = "") -> dict:
    return {"symbol": symbol, "matched": matched,
            "last_close": last_close, "bar_time": bar_time, "note": note}


def run_screen(spec: dict, symbols: list[str], cache, now=None,
               providers: dict | None = None, sync_days: int = 60, *,
               vocabulary: Vocabulary | None = None) -> dict:
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    # The same vocabulary the spec was validated against. A screen that
    # validated clean under an injected term must be evaluable under it too;
    # resolving per symbol inside build_context would put the core vocabulary
    # here and turn every row into an "unknown indicator" note.
    vocabulary = resolve_vocabulary(vocabulary)
    tf = spec.get("tf", "1d")
    needed = referenced_timeframes(spec)
    lookback = max_lookback(spec)
    rows: list[dict] = []
    errors: list[str] = []
    skipped = 0
    for sym in sorted(symbols):
        sync_note = ""
        if providers:
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
            ctx = build_context(cache, sym, now, vocabulary=vocabulary)
            bars = ctx.bars[tf]
            if bars.empty:
                note = f"no {tf} bars cached"
                rows.append(_row(sym,
                                 note=f"{sync_note}; {note}" if sync_note else note))
                skipped += 1
                continue
            if len(bars) < lookback:
                note = (f"only {len(bars)} bars cached; the screen's "
                        f"longest indicator needs {lookback}")
                rows.append(_row(sym,
                                 note=f"{sync_note}; {note}" if sync_note else note))
                skipped += 1
                continue
            # ctx.fe walks the frames build_context already cut at `now`, so the
            # last row of the screen's own timeframe IS the bar being screened.
            # Reading that row is the whole-frame spelling of "evaluate the tree
            # at now"; the driving index never enters it, and must not, because a
            # symbol screened on an intraday timeframe may have no intraday bars
            # to carry a cursor.
            matched = bool(ctx.fe.group_series(spec["conditions"], tf).iloc[-1])
            rows.append(_row(sym, matched=matched,
                             last_close=float(bars["close"].iloc[-1]),
                             bar_time=bars.index[-1].isoformat(),
                             note=sync_note))
        except Exception as e:  # partial failure: other symbols still screen
            errors.append(f"{sym}: {e}")
            note = f"error: {e}"
            rows.append(_row(sym,
                             note=f"{sync_note}; {note}" if sync_note else note))
            skipped += 1
    rows.sort(key=lambda r: (r["matched"] is not True, r["symbol"]))
    return {"bar_close": now.isoformat(),
            "rows": rows,
            "universe": {"screened": len(symbols), "skipped": skipped},
            "errors": errors}
