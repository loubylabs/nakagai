"""Trial generation: mutate a v2 RuleSpec's literals, or assemble composites.

Every mutant returned has already passed its own validator, so a caller that
asks for n trials receives n runnable specs. Generation knows nothing about
running them; that is study.py.
"""

import copy
import hashlib
import json
from dataclasses import dataclass

import numpy as np

from nakagai.strategies.rules.spec import validate_spec

HASH_CHARS = 16


def spec_hash(spec) -> str:
    """Stable digest of a spec. Canonical JSON (sorted keys, no whitespace) so
    two specs that differ only in dict insertion order hash identically, which
    is what makes the digest usable as a cross-process identity."""
    blob = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:HASH_CHARS]


# Key-driven rather than context-driven, because this grammar is consistent:
# "n" is always a lookback, "rhs" is always a comparand, "mult" and "rr" are
# always risk numbers. A context walk ("is this under an ind object?") would
# miss risk.stop.n, which is a period with no indicator above it.
_SITE_KINDS = {"n": "period", "rhs": "threshold", "mult": "mult", "rr": "rr"}


@dataclass(frozen=True)
class Site:
    """One mutable literal, addressed by the path of keys and list indices
    that reaches it from the spec root."""
    path: tuple
    kind: str


def _numeric(value) -> bool:
    # bool is an int subclass in Python; a True in a spec is a flag, not a
    # number to jitter.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def mutable_sites(spec) -> list[Site]:
    """Every numeric literal in `spec` that a literal mutation may move."""
    found: list[Site] = []

    def walk(node, path: tuple) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                here = path + (key,)
                kind = _SITE_KINDS.get(key)
                if kind is not None and _numeric(value):
                    found.append(Site(here, kind))
                else:
                    walk(value, here)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, path + (i,))

    walk(spec, ())
    return found


PERIOD_MIN, PERIOD_MAX = 2, 400
RISK_MIN = 0.2


@dataclass(frozen=True)
class Trial:
    """One runnable spec in a study's frozen trial set."""
    id: str
    strategy: str      # "rules" or "composite"
    spec: dict
    spec_hash: str


def _get(node, path: tuple):
    for key in path:
        node = node[key]
    return node


def _set(node, path: tuple, value) -> None:
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


def _perturb(value, kind: str, rng) -> float | int:
    """Move one literal. Multiplicative rather than additive, so a period of
    200 and a period of 14 both move by a comparable proportion.

    The no-op guard must run AFTER clamping, not before: a draw can land
    outside the valid range, get pulled back to the boundary by the clamp,
    and land exactly back on the original value without ever having equalled
    it pre-clamp. Checking the clamped result is what actually catches that."""
    if kind == "period":
        moved = int(round(value * float(rng.uniform(0.5, 1.8))))
        moved = max(PERIOD_MIN, min(PERIOD_MAX, moved))
        if moved == value:                      # a jitter that changed nothing
            moved = moved - 1 if moved >= PERIOD_MAX else moved + 1
        return moved
    if kind == "threshold":
        return round(float(value) * float(rng.uniform(0.6, 1.4)), 4)
    if kind in ("mult", "rr"):
        moved = round(max(RISK_MIN, float(value) * float(rng.uniform(0.5, 2.0))), 3)
        if moved == value:                      # is a wasted trial
            moved = round(moved + 0.001, 3)      # no upper clamp, so nudging up is always safe
        return moved
    raise ValueError(f"unknown site kind {kind!r}")


def _named(spec: dict, base_name: str) -> tuple[dict, str]:
    """Give a mutant a name derived from its own content, and return the
    name-free digest alongside it.

    The name-free digest is the dedupe key: two mutants with identical numbers
    must not both survive just because we generated different names for them.
    """
    bare = spec_hash({k: v for k, v in spec.items() if k != "name"})
    spec["name"] = f"{base_name}_{bare[:8]}"
    return spec, bare


def literal_trials(base_spec: dict, n: int, seed: int, *,
                   sites_per_mutant: int = 2,
                   max_attempts: int = 60) -> list[Trial]:
    """`n` distinct valid mutants of `base_spec`, moving numeric literals only.

    Structure is frozen: same indicators, same tree shape, same operators. An
    invalid or duplicate draw is discarded and retried, so every returned
    trial is runnable. Raises rather than returning a short list, because a
    study's N is frozen at commission and a caller asking for 60 trials must
    not silently get 41.
    """
    sites = mutable_sites(base_spec)
    if not sites:
        raise ValueError("base spec has no mutable literals to move")
    base_name = str(base_spec.get("name") or "spec")
    rng = np.random.default_rng(seed)
    k = min(sites_per_mutant, len(sites))

    trials: list[Trial] = []
    seen: set[str] = set()
    for _ in range(max_attempts * n):
        if len(trials) == n:
            break
        mutant = copy.deepcopy(base_spec)
        chosen = rng.choice(len(sites), size=k, replace=False)
        for i in chosen:
            site = sites[int(i)]
            _set(mutant, site.path, _perturb(_get(mutant, site.path), site.kind, rng))
        mutant, bare = _named(mutant, base_name)
        if bare in seen or validate_spec(mutant):
            continue
        seen.add(bare)
        trials.append(Trial(id=bare, strategy="rules", spec=mutant,
                            spec_hash=spec_hash(mutant)))

    if len(trials) != n:
        raise ValueError(
            f"could only build {len(trials)} distinct valid mutants of "
            f"{base_name!r}, asked for {n}; widen the spec or lower n")
    return trials
