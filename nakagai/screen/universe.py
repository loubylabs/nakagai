"""Tiered universe resolution for one-shot screens.

full  = the caller's watchlist + explicitly named symbols: all timeframes,
        synced at run time by surfaces that hold provider credentials.
daily = the house screening universe (supplied by the caller): 1d bars only,
        kept fresh by the nightly cron, never synced on demand. A symbol in
        both sets is full."""

import re

FULL = "full"
DAILY = "daily"
MAX_NAMED = 25
_TICKER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,4}$")


def validate_named(named: list[str]) -> list[str]:
    """Uppercase, dedupe (order kept), format-check. No existence check: a
    nonexistent ticker simply screens as a no-data row. Raises ValueError on
    a malformed ticker or more than MAX_NAMED symbols."""
    symbols: list[str] = []
    for raw in named:
        sym = str(raw).strip().upper()
        if not sym:
            continue
        if not _TICKER.match(sym):
            raise ValueError(f"malformed ticker {raw!r}")
        if sym not in symbols:
            symbols.append(sym)
    if len(symbols) > MAX_NAMED:
        raise ValueError(
            f"at most {MAX_NAMED} named symbols per screen (got {len(symbols)})")
    return symbols


def resolve_screen_universe(house: list[str], watchlist: list[str],
                            named: list[str]) -> dict[str, str]:
    """symbol -> tier. Named symbols are validated here (ValueError bubbles
    to the surface's 422); watchlist and house symbols are trusted, they came
    from the caller's own configuration."""
    tiers = {str(sym).upper(): DAILY for sym in house}
    for sym in [str(s).upper() for s in watchlist] + validate_named(named):
        tiers[sym] = FULL
    return tiers
