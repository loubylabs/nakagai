"""Trial generation: spec mutation, validation, and stable hashing."""

from nakagai.lab.mutate import spec_hash


def test_spec_hash_ignores_key_order():
    a = {"version": 2, "name": "x", "timeframe": "1h"}
    b = {"timeframe": "1h", "name": "x", "version": 2}
    assert spec_hash(a) == spec_hash(b)


def test_spec_hash_changes_with_content():
    a = {"version": 2, "name": "x", "timeframe": "1h"}
    b = {"version": 2, "name": "x", "timeframe": "15m"}
    assert spec_hash(a) != spec_hash(b)


def test_spec_hash_is_short_hex():
    h = spec_hash({"name": "x"})
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


from nakagai.lab.mutate import Site, mutable_sites

# The shipped rsi_reversion play, which is the shape every literal mutation
# has to handle: periods nested under indicators, bare numeric comparands,
# and the risk block's own numbers.
BASE = {
    "version": 2,
    "name": "rsi_reversion",
    "timeframe": "1h",
    "long": {"all": [
        {"lhs": {"ind": "rsi", "n": 14}, "op": "crosses_above", "rhs": 30},
        {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 200}},
    ]},
    "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
             "target": {"kind": "rr", "rr": 1.5}},
}


def test_mutable_sites_finds_every_number_worth_moving():
    kinds = sorted(s.kind for s in mutable_sites(BASE))
    assert kinds == ["mult", "period", "period", "period", "rr", "threshold"]


def test_mutable_sites_reaches_into_a_dict_rhs():
    # The sma(200) trend filter lives inside an rhs object, not beside it.
    paths = {s.path for s in mutable_sites(BASE)}
    assert ("long", "all", 1, "rhs", "n") in paths


def test_mutable_sites_finds_the_bare_threshold():
    paths = {s.path for s in mutable_sites(BASE)}
    assert ("long", "all", 0, "rhs") in paths


def test_mutable_sites_never_touches_version_or_name():
    paths = {s.path for s in mutable_sites(BASE)}
    assert ("version",) not in paths
    assert ("name",) not in paths


def test_mutable_sites_on_a_spec_with_no_numbers_is_empty():
    assert mutable_sites({"version": 2, "name": "x", "timeframe": "1h"}) == []
