"""Trial generation: mutate a v2 RuleSpec's literals, or assemble composites.

Every mutant returned has already passed its own validator, so a caller that
asks for n trials receives n runnable specs. Generation knows nothing about
running them; that is study.py.
"""

import hashlib
import json
from dataclasses import dataclass

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
