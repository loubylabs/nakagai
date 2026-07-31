import pytest

from nakagai.screen.universe import resolve_screen_universe, validate_symbols


def test_validate_symbols_uppercases_dedupes_and_keeps_order():
    assert validate_symbols(["aapl", "MSFT", "aapl"], cap=25) == ["AAPL", "MSFT"]


def test_validate_symbols_rejects_a_malformed_ticker():
    with pytest.raises(ValueError, match="malformed ticker"):
        validate_symbols(["not a ticker"], cap=25)


def test_validate_symbols_honors_an_injected_cap():
    with pytest.raises(ValueError, match="at most 2"):
        validate_symbols(["AAPL", "MSFT", "NVDA"], cap=2)


def test_validate_symbols_cap_is_not_a_constant():
    # The same list that fails at 2 passes at 3: the bound is the caller's.
    assert validate_symbols(["AAPL", "MSFT", "NVDA"], cap=3) == ["AAPL", "MSFT", "NVDA"]


def test_resolve_screen_universe_unions_watchlist_and_named():
    assert resolve_screen_universe(["aapl"], ["msft"], cap=25) == ["AAPL", "MSFT"]


def test_resolve_screen_universe_dedupes_across_the_two_sources():
    assert resolve_screen_universe(["AAPL"], ["aapl"], cap=25) == ["AAPL"]


def test_resolve_screen_universe_caps_only_what_the_caller_named():
    # A watchlist is the account's own configuration and is trusted; the cap
    # bounds the ad-hoc half, which is the half that costs a fetch.
    assert len(resolve_screen_universe(["A", "B", "C"], ["D"], cap=1)) == 4
