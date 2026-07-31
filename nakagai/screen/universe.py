"""Symbol resolution for one-shot screens.

There are no tiers. There used to be two, `full` (synced at run time) and
`daily` (a house set kept fresh by a nightly cron, 1d bars only), and they
existed only because some symbols had bars and others did not. On-demand bar
fetch dissolved that distinction, so every symbol is fetched the same way and
the concept is gone.

The cap is the caller's, not this module's. It used to be MAX_NAMED = 25 here,
which put a product bound in a library and left it unreachable by the plan
policy that governs every other allowance."""

import re

_TICKER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,4}$")


def validate_symbols(named: list[str], *, cap: int) -> list[str]:
    """Uppercase, dedupe (order kept), format-check, bound by `cap`.

    No existence check: a nonexistent ticker simply screens as a no-data row.
    Raises ValueError on a malformed ticker or on more than `cap` symbols."""
    symbols: list[str] = []
    for raw in named:
        sym = str(raw).strip().upper()
        if not sym:
            continue
        if not _TICKER.match(sym):
            raise ValueError(f"malformed ticker {raw!r}")
        if sym not in symbols:
            symbols.append(sym)
    if len(symbols) > cap:
        raise ValueError(
            f"at most {cap} named symbols per screen (got {len(symbols)})")
    return symbols


def resolve_screen_universe(watchlist: list[str], named: list[str], *,
                            cap: int) -> list[str]:
    """Every symbol this screen covers, order-stable and deduplicated.

    Named symbols are validated here (ValueError bubbles to the surface's 422);
    watchlist symbols are trusted, they came from the caller's own
    configuration, and they are deliberately outside the cap for the same
    reason."""
    out: list[str] = []
    for sym in [str(s).upper() for s in watchlist] + validate_symbols(named, cap=cap):
        if sym not in out:
            out.append(sym)
    return out
