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
