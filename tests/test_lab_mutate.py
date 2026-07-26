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


def test_mutable_sites_excludes_booleans_under_site_keys():
    # bool is an int subclass in Python, so a flag placed under a site key
    # must still be excluded; if it were not, this would report both paths
    # as mutable numeric literals.
    spec = {"lhs": {"ind": "rsi", "n": True}, "op": ">", "rhs": False}
    paths = {s.path for s in mutable_sites(spec)}
    assert ("lhs", "n") not in paths
    assert ("rhs",) not in paths


import pytest

from nakagai.lab.mutate import Trial, literal_trials
from nakagai.strategies.rules.spec import validate_spec


def test_literal_trials_are_all_valid_specs():
    trials = literal_trials(BASE, n=12, seed=7)
    assert len(trials) == 12
    for t in trials:
        assert validate_spec(t.spec) == [], t.spec


def test_literal_trials_are_distinct():
    trials = literal_trials(BASE, n=12, seed=7)
    assert len({t.id for t in trials}) == 12


def test_literal_trials_are_deterministic_for_a_seed():
    a = literal_trials(BASE, n=8, seed=3)
    b = literal_trials(BASE, n=8, seed=3)
    assert [t.spec_hash for t in a] == [t.spec_hash for t in b]


def test_literal_trials_differ_across_seeds():
    a = literal_trials(BASE, n=8, seed=3)
    b = literal_trials(BASE, n=8, seed=4)
    assert [t.spec_hash for t in a] != [t.spec_hash for t in b]


def test_literal_trials_do_not_mutate_the_base():
    before = spec_hash(BASE)
    literal_trials(BASE, n=6, seed=1)
    assert spec_hash(BASE) == before


def test_literal_trials_carry_the_rules_strategy_name():
    assert all(t.strategy == "rules" for t in literal_trials(BASE, n=3, seed=1))


def test_literal_trials_keep_periods_at_two_or_more():
    # A period of 0 or 1 is meaningless to every indicator in the grammar.
    for t in literal_trials(BASE, n=25, seed=11):
        for site in mutable_sites(t.spec):
            if site.kind == "period":
                node = t.spec
                for key in site.path[:-1]:
                    node = node[key]
                assert node[site.path[-1]] >= 2


def test_literal_trials_refuses_a_spec_with_nothing_to_move():
    with pytest.raises(ValueError, match="no mutable"):
        literal_trials({"version": 2, "name": "x", "timeframe": "1h"},
                       n=3, seed=1)
