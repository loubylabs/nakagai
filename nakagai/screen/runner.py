"""Deterministic one-shot screen evaluation. No LLM anywhere in this path.

Providers, when given, sync each symbol's directly referenced timeframes and
the source timeframes of referenced derived frames before evaluating. A sync
failure never aborts evaluation: it is noted on the row and cached bars still
evaluate. Every row carries its own bar_time so staleness is visible, never
hidden."""

from collections.abc import Mapping

import pandas as pd

from nakagai.data.resample import DERIVED
from nakagai.data.schema import DEFAULT_TIMEFRAMES, TimeframeSet
from nakagai.data.sync import derive_incremental, fetch_incremental
from nakagai.engine.context import build_context
from nakagai.screen.planner import plan_symbol
from nakagai.screen.spec import (
    max_lookback,
    referenced_timeframes,
    screen_reference_pairs,
)
from nakagai.strategies.rules.vocabulary import Vocabulary, resolve_vocabulary


def _row(symbol: str, matched=None, last_close=None,
         bar_time: str = "", note: str = "") -> dict:
    return {"symbol": symbol, "matched": matched,
            "last_close": last_close, "bar_time": bar_time, "note": note}


def run_screen(spec: dict, symbols: list[str], cache, now=None,
               providers: dict | None = None, sync_days: int = 60, *,
               vocabulary: Vocabulary | None = None,
               facts: Mapping[
                   str, Mapping[str, float | int | None]
               ] | None = None,
               ) -> dict:
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    # The same vocabulary the spec was validated against. A screen that
    # validated clean under an injected term must be evaluable under it too;
    # resolving per symbol inside build_context would put the core vocabulary
    # here and turn every row into an "unknown indicator" note.
    vocabulary = resolve_vocabulary(vocabulary)
    tf = spec.get("tf", "1d")
    needed = referenced_timeframes(spec)
    reference_pairs = screen_reference_pairs(spec)
    context_tfs = TimeframeSet(
        driving=DEFAULT_TIMEFRAMES.driving,
        higher=tuple(
            timeframe for timeframe in DEFAULT_TIMEFRAMES.higher
            if timeframe in needed
        ),
        deltas=DEFAULT_TIMEFRAMES.deltas,
        session_aligned=DEFAULT_TIMEFRAMES.session_aligned,
    )
    lookback = max_lookback(spec)
    rows: list[dict] = []
    errors: list[str] = []
    skipped = 0
    for sym in sorted(symbols):
        symbol_facts = (facts or {}).get(sym, {})
        planned = plan_symbol(spec["conditions"], symbol_facts)
        if planned.verdict is not None:
            rows.append(_row(sym, matched=planned.verdict))
            continue
        if not planned.needs_technical:
            names = ", ".join(planned.missing_facts)
            rows.append(_row(sym, note=f"facts unavailable: {names}"))
            skipped += 1
            continue
        sync_note = ""
        pairs = list(dict.fromkeys(
            [(sym, timeframe) for timeframe in sorted(needed)]
            + list(reference_pairs)
        ))
        if providers:
            for pair_symbol, timeframe in pairs:
                sync_tf = DERIVED.get(timeframe, timeframe)
                provider = providers.get(sync_tf)
                if provider is None:
                    continue
                # A run-time sync only has sync_days of headroom by default,
                # which a long lookback (e.g. sma200 on 1d) can't fit in.
                # Intraday timeframes have no such lookback pressure here.
                fetch_days = max(sync_days, lookback * 2) if sync_tf == "1d" else sync_days
                try:
                    fetch_incremental(cache, provider, pair_symbol, sync_tf,
                                      now - pd.Timedelta(days=fetch_days), now)
                except Exception as e:
                    # A transient sync failure shouldn't cost us the cached
                    # bars we already have: keep going and let the row's
                    # note carry the failure so cached bars still evaluate.
                    note = f"sync failed: {e}"
                    errors.append(f"{pair_symbol}: {note}")
                    sync_note = note
        for pair_symbol, derived_tf in (
            pair for pair in pairs if pair[1] in DERIVED
        ):
            try:
                derive_incremental(cache, pair_symbol, derived_tf)
            except Exception as e:
                note = f"sync failed: {e}"
                errors.append(f"{pair_symbol}: {note}")
                sync_note = note
        try:
            ctx = build_context(
                cache, sym, now, context_tfs, reference_pairs=reference_pairs,
                vocabulary=vocabulary,
                facts=symbol_facts,
            )
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
            evaluated = ctx.fe.group_verdict(spec["conditions"], tf)
            if evaluated is None and planned.missing_facts:
                unavailable = ("facts unavailable: "
                               + ", ".join(planned.missing_facts))
                rows.append(_row(
                    sym,
                    last_close=float(bars["close"].iloc[-1]),
                    bar_time=bars.index[-1].isoformat(),
                    note=(f"{sync_note}; {unavailable}"
                          if sync_note else unavailable),
                ))
                skipped += 1
                continue
            matched = False if evaluated is None else evaluated
            rows.append(_row(
                sym,
                matched=matched,
                last_close=float(bars["close"].iloc[-1]),
                bar_time=bars.index[-1].isoformat(),
                note=sync_note,
            ))
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
