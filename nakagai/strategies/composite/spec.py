"""CompositeSpec: boolean composition of strategies into one bigger strategy.

Plain JSON, sibling of RuleSpec. Blocks are member strategies: inline
{"strategy", "params"} or a saved-config ref {"config": name} that the API
inlines before the engine ever sees the spec. Members vote by emitting
signals; nestable all/any trees over block ids decide entries; the composite
owns its own risk block (members' stops/targets are ignored).

    {
      "version": 1,
      "name": "confluence-dip-buyer",
      "blocks": {
        "a": {"strategy": "rsi_reversion", "params": {"rsi_n": 10}},
        "b": {"config": "my-tuned-meanrev"}
      },
      "long": {"any": [{"all": ["a", "b"]}]},
      "window_bars": 4,
      "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
               "target": {"kind": "rr", "rr": 2.0}}
    }
"""

from nakagai.strategies.rules.spec import (
    DEFAULT_RISK, risk_text, validate_risk, validate_spec)

MAX_BLOCKS = 8
WINDOW_BARS_BOUNDS = (1, 20)
DEFAULT_WINDOW_BARS = 4


def _check_tree(tree, blocks: dict, path: str, errs: list[str]) -> None:
    if not isinstance(tree, dict) or len(tree) != 1 or next(iter(tree)) not in ("all", "any"):
        errs.append(f'{path}: expected {{"all": [...]}} or {{"any": [...]}}')
        return
    key, items = next(iter(tree.items()))
    if not isinstance(items, list) or not items:
        errs.append(f"{path}.{key}: must be a non-empty list")
        return
    for i, item in enumerate(items):
        p = f"{path}.{key}[{i}]"
        if isinstance(item, dict):
            _check_tree(item, blocks, p, errs)
        elif isinstance(item, str):
            if item not in blocks:
                errs.append(f"{p}: unknown block {item!r}")
        else:
            errs.append(f"{p}: entries are block ids or nested groups")


def validate_composite_spec(spec, members, allow_refs: bool = True) -> list[str]:
    """Structural validation; empty list = usable. `members` is the mapping of
    strategy name to class a block may reference (the caller's registry view).
    allow_refs=False also rejects unresolved {"config": ...} blocks, since the
    engine only accepts self-contained specs. Per-block param bounds are the
    API layer's job (it owns guardrails)."""
    if not isinstance(spec, dict):
        return ["spec must be a JSON object"]
    errs: list[str] = []
    if not str(spec.get("name", "")).strip():
        errs.append("spec needs a name")
    blocks = spec.get("blocks")
    if not isinstance(blocks, dict) or not blocks:
        errs.append("spec needs a non-empty blocks object")
        blocks = {}
    if len(blocks) > MAX_BLOCKS:
        errs.append(f"at most {MAX_BLOCKS} blocks per composite")
    for bid, block in blocks.items():
        p = f"blocks.{bid}"
        if not isinstance(block, dict):
            errs.append(f"{p}: block must be an object")
            continue
        if "config" in block:
            if not allow_refs:
                errs.append(f"{p}: unresolved config ref {block['config']!r}")
            continue
        name = block.get("strategy")
        if name == "composite":
            errs.append(f"{p}: composites cannot nest composites")
        elif name not in members:
            errs.append(f"{p}: unknown strategy {name!r}")
        if not isinstance(block.get("params", {}), dict):
            errs.append(f"{p}: params must be an object")
    sides = [s for s in ("long", "short") if spec.get(s)]
    if not sides:
        errs.append("spec needs at least one of long/short vote trees")
    for side in sides:
        _check_tree(spec[side], blocks, side, errs)
    wb = spec.get("window_bars", DEFAULT_WINDOW_BARS)
    lo, hi = WINDOW_BARS_BOUNDS
    if isinstance(wb, bool) or not isinstance(wb, int) or not lo <= wb <= hi:
        errs.append(f"window_bars must be an integer in [{lo}, {hi}]")
    errs.extend(validate_risk(spec.get("risk", {})))
    return errs


def validate_composite_blocks(spec: dict, members: dict) -> list[str]:
    """Per-block param validation, the layer validate_composite_spec leaves to
    its caller. Two block kinds carry params worth checking: a "rules" block
    whose params.spec is a full RuleSpec, and a catalog play, which declares no
    PARAMS at all and therefore takes no overrides. Blocks the structural
    validator already rejected (unknown member, non-dict params) are skipped so
    one mistake is reported once."""
    errs: list[str] = []
    for bid, block in (spec.get("blocks") or {}).items():
        if not isinstance(block, dict) or "config" in block:
            continue
        name = block.get("strategy")
        cls = members.get(name)
        params = block.get("params", {})
        if cls is None or not isinstance(params, dict):
            continue
        if name == "rules":
            inner = params.get("spec")
            if not isinstance(inner, dict):
                errs.append(f"blocks.{bid}: rules blocks need params.spec "
                            "(the rule JSON object)")
            else:
                errs.extend(f"blocks.{bid}: {e}" for e in validate_spec(inner))
        elif not cls.PARAMS and params:
            errs.append(f"blocks.{bid}: {name} is a built-in spec and takes no "
                        "param overrides; use a rules block for a tuned leg")
    return errs


def resolve_config_refs(spec: dict, configs: dict) -> tuple[dict, list[str]]:
    """Inline every {"config": name} block from the caller's saved configs.

    Returns (new spec, errors) without mutating the input. Resolution happens
    at API submission time, so a run is pinned to the referenced config as it
    was; later edits don't silently change what an old run meant."""
    if not isinstance(spec, dict) or not isinstance(spec.get("blocks"), dict):
        return spec, []  # structural validation reports the real problem
    errs: list[str] = []
    blocks: dict = {}
    for bid, block in spec["blocks"].items():
        if not (isinstance(block, dict) and "config" in block):
            blocks[bid] = block
            continue
        saved = configs.get(block["config"])
        if saved is None:
            errs.append(f"blocks.{bid}: no saved strategy config {block['config']!r}")
            blocks[bid] = block
        elif saved["strategy"] == "composite":
            errs.append(f"blocks.{bid}: composites cannot nest composites")
            blocks[bid] = block
        else:
            blocks[bid] = {"strategy": saved["strategy"], "params": dict(saved["params"])}
    return {**spec, "blocks": blocks}, errs


def tree_block_ids(tree: dict) -> set[str]:
    key, items = next(iter(tree.items()))
    out: set[str] = set()
    for item in items:
        out |= tree_block_ids(item) if isinstance(item, dict) else {item}
    return out


def eval_tree(tree: dict, live: set[str]) -> bool:
    """A tree passes when its all/any structure is satisfied by live votes."""
    key, items = next(iter(tree.items()))
    results = (eval_tree(i, live) if isinstance(i, dict) else (i in live) for i in items)
    return all(results) if key == "all" else any(results)


def _tree_text(tree, labels: dict) -> str:
    key, items = next(iter(tree.items()))
    parts = [_tree_text(i, labels) if isinstance(i, dict) else labels.get(i, i)
             for i in items]
    if len(parts) == 1:
        return parts[0]
    return "(" + (" AND " if key == "all" else " OR ").join(parts) + ")"


def describe_composite_spec(spec: dict) -> str:
    """Plain-English restatement: the trust step the composer shows live."""
    blocks = spec.get("blocks", {})
    labels = {bid: (b.get("config") or b.get("strategy") or bid)
              for bid, b in blocks.items() if isinstance(b, dict)}
    lines = [f'Composite "{spec.get("name", "unnamed")}" of {len(blocks)} blocks.']
    wb = spec.get("window_bars", DEFAULT_WINDOW_BARS)
    for side in ("long", "short"):
        if spec.get(side):
            lines.append(f"Enter {side} when {_tree_text(spec[side], labels)}"
                         f"; votes live {wb} bars.")
    lines.append(risk_text(spec.get("risk", DEFAULT_RISK)))
    return "\n".join(lines)
