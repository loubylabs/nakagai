import pytest

from nakagai.screen import sectors
from nakagai.screen.sectors import SECTOR_NAMES, UNKNOWN, sector_for_sic


@pytest.mark.parametrize("sic,sector", [
    ("7372", "Technology"), ("3674", "Technology"), ("7370", "Technology"),
    ("2834", "Health Care"), ("3841", "Health Care"), ("8062", "Health Care"),
    ("6021", "Financials"), ("6311", "Financials"), ("6770", "Financials"),
    ("6798", "Real Estate"), ("6512", "Real Estate"),
    ("4813", "Communication Services"), ("7812", "Communication Services"), ("2711", "Communication Services"),
    ("1311", "Energy"), ("2911", "Energy"), ("4922", "Utilities"), ("4911", "Utilities"),
    ("4953", "Industrials"), ("3721", "Industrials"), ("4512", "Industrials"), ("3812", "Industrials"),
    ("2844", "Consumer Staples"), ("2080", "Consumer Staples"), ("5411", "Consumer Staples"), ("5331", "Consumer Staples"),
    ("3711", "Consumer Discretionary"), ("5812", "Consumer Discretionary"), ("5961", "Consumer Discretionary"), ("1531", "Consumer Discretionary"),
    ("2821", "Materials"), ("1040", "Materials"), ("3312", "Materials"),
    (None, "Unknown"), ("", "Unknown"), ("abcd", "Unknown"), ("9995", "Unknown"), ("٣٦٧٤", "Unknown"),
])
def test_sector_for_sic(sic, sector):
    assert sector_for_sic(sic) == sector


def test_sector_names_are_the_eleven_board_sectors():
    assert SECTOR_NAMES == (
        "Technology", "Financials", "Health Care", "Consumer Discretionary", "Consumer Staples",
        "Energy", "Industrials", "Materials", "Utilities", "Real Estate", "Communication Services",
    )
    assert UNKNOWN not in SECTOR_NAMES


def test_every_range_maps_into_the_names():
    reached = {sector for _, _, sector in sectors._RANGES}
    assert reached == set(SECTOR_NAMES)
