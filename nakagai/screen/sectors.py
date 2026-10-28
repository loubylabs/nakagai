"""SEC SIC code to one of the 11 board sectors.

An approximation: the names are GICS sectors, the codes are SEC's
SIC industries, and the two do not line up one to one. SIC cannot split
Communication Services from Technology (7370 is Alphabet and Meta, both filed
under computer services), so such filers land in Technology.
Ranges are inclusive and checked in order; the first match wins, so the narrow
overrides come before the wide ranges they cut out of."""

from __future__ import annotations

SECTOR_NAMES: tuple[str, ...] = (
    "Technology", "Financials", "Health Care", "Consumer Discretionary", "Consumer Staples",
    "Energy", "Industrials", "Materials", "Utilities", "Real Estate", "Communication Services",
)
UNKNOWN = "Unknown"

_RANGES: tuple[tuple[int, int, str], ...] = (
    (6798, 6798, "Real Estate"), (6500, 6599, "Real Estate"),
    (2830, 2836, "Health Care"), (3840, 3851, "Health Care"), (5122, 5122, "Health Care"),
    (8000, 8099, "Health Care"),
    (2840, 2844, "Consumer Staples"), (5331, 5331, "Consumer Staples"),
    (5399, 5399, "Consumer Staples"), (5400, 5499, "Consumer Staples"),
    (5912, 5912, "Consumer Staples"), (5140, 5149, "Consumer Staples"),
    (1531, 1531, "Consumer Discretionary"), (3630, 3639, "Consumer Discretionary"),
    (3710, 3716, "Consumer Discretionary"), (3751, 3751, "Consumer Discretionary"),
    (2911, 2911, "Energy"), (1200, 1399, "Energy"), (4610, 4619, "Energy"),
    (5171, 5172, "Energy"),
    (4950, 4959, "Industrials"),
    (4900, 4999, "Utilities"),
    (4800, 4899, "Communication Services"), (2710, 2741, "Communication Services"),
    (7810, 7849, "Communication Services"),
    (3570, 3579, "Technology"), (3600, 3699, "Technology"), (3820, 3829, "Technology"),
    (7370, 7379, "Technology"),
    (6000, 6799, "Financials"),
    (100, 999, "Consumer Staples"), (2000, 2199, "Consumer Staples"),
    (1000, 1099, "Materials"), (1400, 1499, "Materials"), (2400, 2499, "Materials"),
    (2600, 2699, "Materials"), (2800, 2899, "Materials"), (3080, 3089, "Materials"),
    (3210, 3399, "Materials"),
    (1500, 1799, "Industrials"), (2750, 2799, "Industrials"), (3400, 3569, "Industrials"),
    (3580, 3599, "Industrials"), (3720, 3809, "Industrials"), (3812, 3812, "Industrials"),
    (4000, 4799, "Industrials"), (5000, 5099, "Industrials"), (7300, 7369, "Industrials"),
    (7380, 7389, "Industrials"), (8700, 8799, "Industrials"),
    (2300, 2399, "Consumer Discretionary"), (2500, 2599, "Consumer Discretionary"),
    (3010, 3069, "Consumer Discretionary"), (3140, 3199, "Consumer Discretionary"),
    (3940, 3949, "Consumer Discretionary"), (5200, 5999, "Consumer Discretionary"),
    (7000, 7099, "Consumer Discretionary"), (7200, 7299, "Consumer Discretionary"),
    (7900, 7999, "Consumer Discretionary"), (8200, 8299, "Consumer Discretionary"),
)


def sector_for_sic(sic: str | None) -> str:
    """The board sector for a SIC code, or UNKNOWN for a missing, non-numeric or unmapped one."""
    if not sic or not sic.isascii() or not sic.isdigit():
        return UNKNOWN
    code = int(sic)
    for low, high, sector in _RANGES:
        if low <= code <= high:
            return sector
    return UNKNOWN
