"""ScreenSpec v1: the conditions-only screening IR.

A screen is one all/any/not condition group in RuleSpec v2's exact operand
grammar, plus a base timeframe and an optional universe of sectors and SIC
industries that narrows which symbols are read. No entries, exits, sides, or
risk: a screen answers "does this condition hold for this symbol right now,"
nothing else.
Validation and the readback delegate to the rules grammar so the two IRs can
never drift, and both take the caller's vocabulary so a screen is validated,
described, and evaluated against exactly the terms its prompt advertised."""

import re

from nakagai.screen.planner import walk_values
from nakagai.screen.sectors import SECTOR_NAMES, sector_for_sic
from nakagai.strategies.rules.spec import (
    TIMEFRAMES, group_text, validate_condition_group,
)
from nakagai.strategies.rules.strategy import (
    ReferencePair, expression_reference_pairs,
)
from nakagai.strategies.rules.vocabulary import Vocabulary, resolve_vocabulary
from nakagai.screen.facts import DISCOVERY_FACTS

VERSION = 1
_KEYS = {"version", "tf", "universe", "conditions"}
_LOOKBACK_ARGS = ("n", "slow", "senkou_n", "kijun_n")
_MIN_LOOKBACK = 20
MAX_SECTORS = len(SECTOR_NAMES)
MAX_INDUSTRIES = 20
_UNIVERSE_KEYS = {"sectors", "industries"}


def _universe_errors(universe) -> list[str]:
    if not isinstance(universe, dict):
        return ["universe must be an object with sectors and/or industries"]
    if not universe:
        return ["universe needs sectors or industries"]
    errs: list[str] = []
    unknown = set(universe) - _UNIVERSE_KEYS
    if unknown:
        errs.append(f"universe: unknown keys {sorted(unknown)}")
    if "sectors" in universe:
        sectors = universe["sectors"]
        if (not isinstance(sectors, list)
                or not 1 <= len(sectors) <= MAX_SECTORS):
            errs.append(f"universe.sectors must be a list of 1 to "
                        f"{MAX_SECTORS} sector names")
        else:
            seen: list = []
            for name in sectors:
                if name not in SECTOR_NAMES:
                    errs.append(f"universe.sectors: unknown sector {name!r}")
                elif name in seen:
                    errs.append(f"universe.sectors: duplicate {name!r}")
                seen.append(name)
    if "industries" in universe:
        codes = universe["industries"]
        if (not isinstance(codes, list)
                or not 1 <= len(codes) <= MAX_INDUSTRIES):
            errs.append(f"universe.industries must be a list of 1 to "
                        f"{MAX_INDUSTRIES} SIC codes")
        else:
            seen = []
            for code in codes:
                if not isinstance(code, str) or not re.fullmatch(r"[0-9]{4}", code):
                    errs.append(f"universe.industries: {code!r} is not a "
                                f"four-digit SIC code string")
                elif code in seen:
                    errs.append(f"universe.industries: duplicate {code!r}")
                seen.append(code)
    return errs


def validate_screen_spec(spec, *,
                         vocabulary: Vocabulary | None = None) -> list[str]:
    """Structural + grammar validation. Empty list = usable spec."""
    vocabulary = resolve_vocabulary(vocabulary)
    if not isinstance(spec, dict):
        return ["spec must be a JSON object"]
    errs: list[str] = []
    if spec.get("version") != VERSION:
        errs.append(f"spec version must be {VERSION} (got {spec.get('version')!r})")
    tf = spec.get("tf", "1d")
    if tf not in TIMEFRAMES:
        errs.append(f"tf must be one of {TIMEFRAMES}, got {spec.get('tf')!r}")
    if "conditions" not in spec:
        errs.append("spec needs a conditions group")
    else:
        # A screen defaults to 1d, which is session-aligned, so "daily screen
        # plus one intraday reference" is the shape a user reaches for first
        # and the one the evaluator cannot carry. Refusing it here rather than
        # per symbol keeps it out of the rows, where it reads as no matches.
        errs.extend(validate_condition_group(spec["conditions"], "conditions",
                                             tf if tf in TIMEFRAMES else "1h",
                                             vocabulary=vocabulary,
                                             allowed_facts=DISCOVERY_FACTS))
    if "universe" in spec:
        errs.extend(_universe_errors(spec["universe"]))
    unknown = set(spec) - _KEYS
    if unknown:
        errs.append(f"unknown keys {sorted(unknown)}")
    return errs


def describe_screen(spec: dict, *,
                    vocabulary: Vocabulary | None = None) -> str:
    """Plain-English restatement of a validated screen: the trust step."""
    universe = spec.get("universe")
    scope = ""
    if universe is not None:
        parts = list(universe.get("sectors", ()))
        parts += [f"SIC {code}" for code in universe.get("industries", ())]
        scope = f"in {' or '.join(parts)}, "
    return (f"Screen on {spec.get('tf', '1d')} bars, {scope}matching symbols where "
            + group_text(spec["conditions"], resolve_vocabulary(vocabulary),
                         allowed_facts=DISCOVERY_FACTS))


def universe_of(spec: dict) -> tuple[frozenset[str], frozenset[str]] | None:
    """The (sectors, industries) a validated spec narrows to, or None."""
    universe = spec.get("universe")
    if universe is None:
        return None
    return (frozenset(universe.get("sectors", ())),
            frozenset(universe.get("industries", ())))


def universe_admits(universe: tuple[frozenset[str], frozenset[str]],
                    sic: str | None) -> bool:
    """Whether a symbol with this SIC code is inside the universe."""
    if not sic:
        return False
    sectors, industries = universe
    # SEC stores some codes unpadded ("100"); industries are four digits.
    code = sic.zfill(4) if sic.isascii() and sic.isdigit() else sic
    return code in industries or sector_for_sic(sic) in sectors


def referenced_timeframes(spec: dict) -> set[str]:
    """The base tf plus every per-node tf the condition tree mentions."""
    tfs = {spec.get("tf", "1d")}

    def walk(node):
        if isinstance(node, dict):
            tf = node.get("tf")
            if isinstance(tf, str):
                tfs.add(tf)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(spec.get("conditions", {}))
    return tfs


def referenced_facts(spec: dict) -> frozenset[str]:
    """Every discovery fact name the condition tree reads, on either side of
    any comparison at any depth, so a caller can skip loading the rest."""
    return frozenset(
        node["fact"] for node in walk_values(spec.get("conditions", {}))
        if isinstance(node, dict) and isinstance(node.get("fact"), str))


def screen_reference_pairs(spec: dict) -> tuple[ReferencePair, ...]:
    """Exact reference pairs declared by one ScreenSpec."""
    tf = spec.get("tf", "1d")
    host_timeframe = tf if isinstance(tf, str) else "1d"
    return expression_reference_pairs(spec.get("conditions", {}), host_timeframe)


def is_intraday(spec: dict) -> bool:
    return referenced_timeframes(spec) != {"1d"}


def max_lookback(spec: dict) -> int:
    """Coarse longest indicator window in the tree (floor 20): the runner uses
    it to call out symbols whose cached history cannot fill the indicators."""
    worst = _MIN_LOOKBACK

    def walk(node):
        nonlocal worst
        if isinstance(node, dict):
            for key in _LOOKBACK_ARGS:
                v = node.get(key)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    worst = max(worst, int(v))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(spec.get("conditions", {}))
    return worst
