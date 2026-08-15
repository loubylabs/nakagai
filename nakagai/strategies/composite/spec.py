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

from collections.abc import Container, Mapping

from nakagai.strategies.rules.spec import (
    DEFAULT_RISK, risk_text, validate_risk, validate_spec)
from nakagai.strategies.rules.vocabulary import Vocabulary, resolve_vocabulary

MAX_BLOCKS = 8
WINDOW_BARS_BOUNDS = (1, 20)
DEFAULT_WINDOW_BARS = 4

# The member name that means "this block writes its own RuleSpec inline",
# rather than naming a member whose body is already bound.
#
# It lives HERE, with the validator that acts on it, because this module owns
# the protocol: `validate_composite_blocks` decides what a block naming it must
# carry. The NL builder's prompt imports it rather than spelling it again, so
# renaming it moves the grammar the model is taught and the rule the reply is
# judged by together. Two spellings of one protocol name drift silently: the
# prompt would teach one word while the validator refused everything that was
# not the other, and every reply would burn a retry on advice naming a word the
# prompt no longer used.
BESPOKE_LEG = "rules"


# A vote tree over at most MAX_BLOCKS ids needs nothing like this much nesting;
# the bound exists so a tree that is absurd rather than merely wrong is REFUSED
# instead of recursing until the interpreter stops it. A RecursionError out of
# a validator is the same defect as any other raise: it reached the NL retry
# loop as a dead request rather than as an error the model could act on.
MAX_TREE_DEPTH = 32


def _check_tree(tree, blocks: dict, path: str, errs: list[str],
                depth: int = 0) -> None:
    if depth > MAX_TREE_DEPTH:
        errs.append(f"{path}: vote trees nest at most {MAX_TREE_DEPTH} deep")
        return
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
            _check_tree(item, blocks, p, errs, depth + 1)
        elif isinstance(item, str):
            if item not in blocks:
                errs.append(f"{p}: unknown block {item!r}")
        else:
            errs.append(f"{p}: entries are block ids or nested groups")


def validate_composite_spec(spec, members: Container,
                            allow_refs: bool = True) -> list[str]:
    """Structural validation; empty list = usable. `members` names the
    strategies a block may reference and is read for membership alone, so a
    mapping of definitions and a bare set of names answer identically. There is
    no class to read here: 0.5.0 replaced member classes with frozen values.
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
        elif not isinstance(name, str) or name not in members:
            # The isinstance guard is what keeps this a VALIDATOR. `name` is
            # whatever arrived in the JSON, and a list or an object is
            # unhashable, so testing membership on it raises TypeError out of a
            # function whose whole contract is to return errors. The NL
            # builder's retry loop is the caller that cannot survive it: a
            # malformed reply became a 503 rather than the retry the model
            # could have acted on.
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


def validate_composite_blocks(spec: dict, members: Container,
                              vocabulary: Vocabulary | None = None) -> list[str]:
    """Per-block param validation, the layer validate_composite_spec leaves to
    its caller. Two block kinds carry params worth checking: a "rules" block
    whose params.spec is a full RuleSpec, and any other member, whose body is
    already bound and which therefore takes no overrides. Blocks the structural
    validator already rejected (unknown member, non-dict params) are skipped so
    one mistake is reported once.

    `members` is read for MEMBERSHIP alone, so anything answering `in` will do:
    the sibling structural validator reads it the same way, and the NL builder
    hands a set of names it built from the caller's catalog cards. It used to
    read `cls.PARAMS` off each member and refuse an override only where that
    was empty, which stopped being callable when 0.5.0 replaced member classes
    with `StrategyDefinition` values carrying no such attribute: every caller
    on the value model got an AttributeError instead of an answer
    (chrvsd/nakagai#417).

    The rule the read stood for is unconditional now, and simpler for it: a
    block that is not the bespoke leg carries no params. A catalog definition
    binds its spec at construction, so an override has nowhere useful to land,
    and both ways it can fail are worse than a refusal here. `params.spec` is
    refused at the factory (`ReplayInputError`), so the block would die
    mid-replay rather than at validation. Any other key is carried into the
    strategy unread, so the play runs untuned while its author believes
    otherwise; that half is tracked as chrvsd/nakagai#460.

    KNOWN LIMITATION, pinned by
    tests/test_composite.py::test_an_unbound_member_under_another_name_is_refused.
    The bespoke leg is recognized by `BESPOKE_LEG`, so an UNBOUND
    definition registered under any other name is refused here even though its
    spec legitimately travels in `params` and its factory would build it. The
    class model refused it identically, for the same reason: `Strategy.PARAMS`
    is empty on every unbound adapter too. Closing it needs the caller to say
    which members are unbound, which this signature deliberately does not carry,
    so it is recorded rather than guessed at.

    vocabulary=None defaults to core_vocabulary(): the honest default for a
    library whose own tests and catalog need a vocabulary to exist. A caller
    holding its own vocabulary, such as the NL builder validating a spec
    against the house's injected terms, passes it through so a rules block
    naming one of those terms validates the same way here as it does when the
    composite actually runs."""
    vocabulary = resolve_vocabulary(vocabulary)
    errs: list[str] = []
    # Shape first, and silently: `validate_composite_spec` reports a spec that
    # is not an object and a `blocks` that is not one, so reporting it twice
    # would double every such message. What matters here is that neither shape
    # can raise out of a function whose contract is to return errors.
    if not isinstance(spec, dict):
        return errs
    blocks = spec.get("blocks")
    if not isinstance(blocks, dict):
        return errs
    for bid, block in blocks.items():
        if not isinstance(block, dict) or "config" in block:
            continue
        name = block.get("strategy")
        params = block.get("params", {})
        if not isinstance(name, str) or name not in members:
            continue          # unhashable or unknown: reported by the sibling
        if not isinstance(params, dict):
            continue
        if name == BESPOKE_LEG:
            inner = params.get("spec")
            if not isinstance(inner, dict):
                errs.append(f"blocks.{bid}: rules blocks need params.spec "
                            "(the rule JSON object)")
            else:
                errs.extend(f"blocks.{bid}: {e}"
                            for e in validate_spec(inner, vocabulary))
        elif params:
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
    if not isinstance(configs, Mapping):
        return spec, ["configs must be a mapping of saved strategy configs"]
    errs: list[str] = []
    blocks: dict = {}
    for bid, block in spec["blocks"].items():
        if not (isinstance(block, dict) and "config" in block):
            blocks[bid] = block
            continue
        ref = block["config"]
        saved = configs.get(ref) if isinstance(ref, str) else None
        # A saved config is a caller value too, and this inlines it into a spec
        # the engine will run, so the shape it must have is checked before it is
        # indexed rather than after it fails.
        if not isinstance(saved, dict) or not isinstance(saved.get("params"), dict) \
                or not isinstance(saved.get("strategy"), str):
            saved = None
        if saved is None:
            errs.append(f"blocks.{bid}: no saved strategy config {ref!r}")
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
    if not isinstance(spec, dict):
        return "Not a composite spec."
    blocks = spec.get("blocks")
    if not isinstance(blocks, dict):
        blocks = {}
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
