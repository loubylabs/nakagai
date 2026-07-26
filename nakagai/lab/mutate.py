"""Trial generation: mutate a v2 RuleSpec's literals, or assemble composites.

Every mutant returned has already passed its own validator, so a caller that
asks for n trials receives n runnable specs. Generation knows nothing about
running them; that is study.py.
"""

import hashlib
import json

HASH_CHARS = 16


def spec_hash(spec) -> str:
    """Stable digest of a spec. Canonical JSON (sorted keys, no whitespace) so
    two specs that differ only in dict insertion order hash identically, which
    is what makes the digest usable as a cross-process identity."""
    blob = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:HASH_CHARS]
