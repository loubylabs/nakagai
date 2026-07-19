"""ScreenSpec v1: the conditions-only screening IR.

A screen is one all/any condition group in RuleSpec v2's exact operand
grammar, plus a base timeframe. No entries, exits, sides, or risk: a screen
answers "does this condition hold for this symbol right now," nothing else.
Validation and the readback delegate to the rules grammar so the two IRs can
never drift."""

from nakagai.strategies.rules.spec import (
    TIMEFRAMES, group_text, validate_condition_group,
)

VERSION = 1
_KEYS = {"version", "tf", "conditions"}
_LOOKBACK_ARGS = ("n", "slow", "senkou_n", "kijun_n")
_MIN_LOOKBACK = 20


def validate_screen_spec(spec) -> list[str]:
    """Structural + grammar validation. Empty list = usable spec."""
    if not isinstance(spec, dict):
        return ["spec must be a JSON object"]
    errs: list[str] = []
    if spec.get("version") != VERSION:
        errs.append(f"spec version must be {VERSION} (got {spec.get('version')!r})")
    if spec.get("tf", "1d") not in TIMEFRAMES:
        errs.append(f"tf must be one of {TIMEFRAMES}, got {spec.get('tf')!r}")
    if "conditions" not in spec:
        errs.append("spec needs a conditions group")
    else:
        errs.extend(validate_condition_group(spec["conditions"], "conditions"))
    unknown = set(spec) - _KEYS
    if unknown:
        errs.append(f"unknown keys {sorted(unknown)}")
    return errs


def describe_screen(spec: dict) -> str:
    """Plain-English restatement of a validated screen: the trust step."""
    return (f"Screen on {spec.get('tf', '1d')} bars, matching symbols where "
            + group_text(spec["conditions"]))


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
