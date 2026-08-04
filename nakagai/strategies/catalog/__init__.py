"""Catalog loading: JSON spec files into RuleStrategy subclasses.

Each JSON file under a specs directory is card metadata + a RuleSpec v2
`spec`. load_catalog() turns them into RuleStrategy subclasses so a registry
can serve them alongside user rules and composites. The directory is the
content slot: the platform points these loaders at its curated specs
(nakagai/registry.py); the loaders themselves know no fixed location.

BOTH ARGUMENTS ARE REQUIRED, and the factory has no default on purpose. These
loaders are @cache'd on their argument tuple, so `load_catalog(dir)` and
`load_catalog(dir, house_vocabulary)` would be two entries over the same specs
and hand back two different `Catalog_*` classes per play. Nothing raises on
that: isinstance and registry identity simply start disagreeing, in a process
where both spellings are reachable. A caller that has a vocabulary must say so,
and a caller that wants the core's must pass core_vocabulary by name."""

import json
from functools import cache
from pathlib import Path

from nakagai.strategies.rules import (
    RuleStrategy, VocabularyFactory, validate_spec,
)


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


@cache
def load_catalog(specs_dir: Path,
                 vocabulary_factory: VocabularyFactory
                 ) -> dict[str, type[RuleStrategy]]:
    out = {}
    base = RuleStrategy.bound(vocabulary_factory)
    for name, entry in load_entries(specs_dir, vocabulary_factory).items():
        out[name] = type(
            f"Catalog_{name}", (base,),
            {"name": name, "title": entry["title"],
             "description": entry["description"], "category": entry["category"],
             "tags": tuple(entry["tags"]), "timeframe": entry["spec"]["timeframe"],
             "DEFAULT_PARAMS": {"spec": entry["spec"]}, "PARAMS": {}})
    return out
