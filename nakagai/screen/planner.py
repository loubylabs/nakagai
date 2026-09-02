"""Cheap three-valued planning for ScreenSpec condition groups."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

from nakagai.strategies.rules.spec import is_group_node


@dataclass(frozen=True)
class PlannedSymbol:
    verdict: bool | None
    needs_technical: bool
    missing_facts: tuple[str, ...]


@dataclass(frozen=True)
class _ExprPlan:
    value: float | None
    needs_technical: bool
    missing_facts: frozenset[str]


@dataclass(frozen=True)
class _GroupPlan:
    verdict: bool | None
    needs_technical: bool
    missing_facts: frozenset[str]


def _walk_values(root):
    stack = [root]
    while stack:
        value = stack.pop()
        yield value
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)


def uses_facts(group) -> bool:
    return any(isinstance(node, dict) and "fact" in node
               for node in _walk_values(group))


def uses_technical(group) -> bool:
    return any(
        isinstance(node, dict) and any(
            kind in node for kind in ("src", "ind", "prim"))
        for node in _walk_values(group)
    )


def _fact_value(name: str, facts: Mapping[str, float | int | None]) -> _ExprPlan:
    value = facts.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _ExprPlan(None, False, frozenset((name,)))
    number = float(value)
    if not math.isfinite(number):
        return _ExprPlan(None, False, frozenset((name,)))
    return _ExprPlan(number, False, frozenset())


def _apply_math(op: str, values: list[float]) -> float | None:
    try:
        if op == "abs":
            result = abs(values[0])
        elif op == "+":
            result = sum(values)
        elif op == "-":
            result = values[0] - values[1]
        elif op == "*":
            result = math.prod(values)
        elif op == "/":
            result = float("nan") if values[1] == 0 else values[0] / values[1]
        elif op == "min":
            result = min(values)
        elif op == "max":
            result = max(values)
        else:
            return None
    except (ArithmeticError, OverflowError, ValueError):
        return None
    return float(result) if math.isfinite(float(result)) else None


def _plan_expr(node, facts: Mapping[str, float | int | None]) -> _ExprPlan:
    plans: dict[int, _ExprPlan] = {}
    stack = [(node, False)]
    while stack:
        current, visited = stack.pop()
        key = id(current)
        if isinstance(current, bool):
            plans[key] = _ExprPlan(None, False, frozenset())
        elif isinstance(current, (int, float)):
            value = float(current)
            plans[key] = _ExprPlan(
                value if math.isfinite(value) else None,
                False,
                frozenset(),
            )
        elif not isinstance(current, dict):
            plans[key] = _ExprPlan(None, False, frozenset())
        elif "fact" in current:
            plans[key] = _fact_value(current["fact"], facts)
        elif "op" in current and isinstance(current.get("args"), list):
            if not visited:
                stack.append((current, True))
                stack.extend((child, False) for child in reversed(current["args"]))
                continue
            children = [plans[id(child)] for child in current["args"]]
            missing = frozenset().union(
                *(child.missing_facts for child in children))
            technical = any(child.needs_technical for child in children)
            if any(child.value is None for child in children):
                plans[key] = _ExprPlan(None, technical, missing)
            else:
                value = _apply_math(
                    current["op"],
                    [child.value for child in children if child.value is not None],
                )
                plans[key] = _ExprPlan(value, technical, missing)
        else:
            plans[key] = _ExprPlan(None, True, frozenset())
    return plans[id(node)]


def _plan_condition(cond: dict,
                    facts: Mapping[str, float | int | None]) -> _GroupPlan:
    lhs = _plan_expr(cond["lhs"], facts)
    rhs = _plan_expr(cond["rhs"], facts)
    technical = lhs.needs_technical or rhs.needs_technical
    missing = lhs.missing_facts | rhs.missing_facts
    if lhs.value is None or rhs.value is None or cond["op"].startswith("crosses_"):
        return _GroupPlan(None, technical, missing)
    verdict = {
        ">": lhs.value > rhs.value,
        "<": lhs.value < rhs.value,
        ">=": lhs.value >= rhs.value,
        "<=": lhs.value <= rhs.value,
    }[cond["op"]]
    return _GroupPlan(bool(verdict), False, frozenset())


def _reduce_group(key: str, children: list[_GroupPlan]) -> _GroupPlan:
    if key == "not":
        child = children[0]
        if child.verdict is None:
            return child
        return _GroupPlan(not child.verdict, False, frozenset())
    known = [child.verdict for child in children]
    if key == "all" and False in known:
        return _GroupPlan(False, False, frozenset())
    if key == "any" and True in known:
        return _GroupPlan(True, False, frozenset())
    if all(verdict is not None for verdict in known):
        verdict = all(known) if key == "all" else any(known)
        return _GroupPlan(bool(verdict), False, frozenset())
    return _GroupPlan(
        None,
        any(child.needs_technical for child in children),
        frozenset().union(*(child.missing_facts for child in children)),
    )


def plan_symbol(group, facts: Mapping[str, float | int | None]) -> PlannedSymbol:
    """Classify one symbol before any bar cache access."""
    plans: dict[int, _GroupPlan] = {}
    stack = [(group, False)]
    while stack:
        current, visited = stack.pop()
        key = id(current)
        if not is_group_node(current):
            plans[key] = _plan_condition(current, facts)
            continue
        group_key, value = next(iter(current.items()))
        children = [value] if group_key == "not" else value
        if not visited:
            stack.append((current, True))
            stack.extend((child, False) for child in reversed(children))
            continue
        plans[key] = _reduce_group(
            group_key, [plans[id(child)] for child in children])
    result = plans[id(group)]
    return PlannedSymbol(
        result.verdict,
        result.needs_technical if result.verdict is None else False,
        tuple(sorted(result.missing_facts)) if result.verdict is None else (),
    )
