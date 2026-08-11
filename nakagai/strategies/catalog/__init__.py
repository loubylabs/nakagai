"""Catalog loading: JSON spec files into frozen strategy definitions.

Each JSON file under a specs directory is card metadata plus a RuleSpec v2
`spec`. `catalog_definitions` turns them into `StrategyDefinition` values a
registry bundle can take. The directory is the content slot: the platform
points these loaders at its curated specs (nakagai/registry.py); the loaders
themselves know no fixed location.

BOTH ARGUMENTS ARE REQUIRED, and the vocabulary factory has no default on
purpose. One spec read under two grammars is two strategies, so a loader that
guessed would hand back definitions whose digests claim a grammar the caller
never chose. `load_entries` is @cache'd on its whole argument tuple, so a
defaulted call and an explicit one would also be two entries over one
directory. A caller that has a vocabulary must say so, and a caller that wants
the core's must pass `core_vocabulary` by name."""

import json
from functools import cache
from pathlib import Path

from nakagai.engine.registry import (
    StrategyDefinition, rules_definition, spec_base_digest,
)
from nakagai.strategies.rules import VocabularyFactory, validate_spec


@cache
def load_entries(specs_dir: Path,
                 vocabulary_factory: VocabularyFactory) -> dict[str, dict]:
    vocabulary = vocabulary_factory()
    out = {}
    for path in sorted(Path(specs_dir).glob("*.json")):
        entry = json.loads(path.read_text())
        errs = validate_spec(entry["spec"], vocabulary)
        if errs:
            raise ValueError(f"catalog spec {path.name} invalid: {'; '.join(errs)}")
        out[entry["spec"]["name"]] = entry
    return out


def catalog_definitions(specs_dir: Path,
                        vocabulary_factory: VocabularyFactory
                        ) -> tuple[StrategyDefinition, ...]:
    """One frozen definition per shipped spec, ready to enter a bundle.

    No minted subclass and no class attribute: the definition carries the
    name, binds the immutable spec, records the grammar it is read under, and
    builds a plain RuleStrategy fresh for every candidate. The base digest
    covers the spec AND that vocabulary, because one spec under two grammars is
    two strategies, which is the same reason `load_entries` demands one rather
    than guessing.

    Deliberately not cached. These are values, so two calls give two equal
    bundles rather than two disagreeing ones, and there is no class identity
    left for a second cache entry to split.
    """
    return tuple(
        rules_definition(
            name, spec_base_digest(entry["spec"], vocabulary_factory),
            spec=entry["spec"], vocabulary_factory=vocabulary_factory,
        )
        for name, entry in load_entries(specs_dir, vocabulary_factory).items()
    )
