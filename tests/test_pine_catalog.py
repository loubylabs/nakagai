"""Every shipped play compiles, and compiles into both artifacts.

The catalog is the content slot: it grows a play whenever the book does, and
nothing else in this suite would notice a play the compiler cannot render. The
platform runs the same sweep over its own 57 specs through the pinned core, so
this is the near end of that guarantee rather than a duplicate of it.
"""

import pytest

from nakagai.strategies.catalog import load_entries
from nakagai.strategies.rules import compile_pine, core_vocabulary, spec_hash
from tests.test_catalog import SPECS

ENTRIES = sorted(load_entries(SPECS, core_vocabulary).items())


@pytest.mark.parametrize("name, entry", ENTRIES, ids=[n for n, _ in ENTRIES])
def test_every_shipped_play_renders_both_artifacts(name, entry):
    spec = entry["spec"]
    bundle = compile_pine(spec)
    assert bundle.spec_hash == spec_hash(spec)
    for source in (bundle.indicator, bundle.strategy):
        assert source.startswith("//@version=6\n")
        assert f"// Nakagai Pine export: {name}" in source
        assert f"// Spec hash: {bundle.spec_hash}" in source
        assert "nk_long_decision" in source and "nk_short_decision" in source
    assert "indicator(" in bundle.indicator
    assert "strategy(" in bundle.strategy


def test_the_catalog_is_not_empty():
    # A loader that silently returned {} would make every parametrized case
    # above vanish and the file pass with nothing run.
    assert len(ENTRIES) == 3
