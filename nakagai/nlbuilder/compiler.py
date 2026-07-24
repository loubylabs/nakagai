"""English -> a validated RuleSpec v2 or a validated composite spec, via the
Claude API with a validator-driven retry loop. The reply's "kind" selects the
validator, rules or composite, and it stays the single source of truth; the
model is asked to fix precisely the errors it reports. API failures come back
as CompileResult.error rather than an exception, so usage accumulated across
retries always survives."""

import json
import re
from dataclasses import dataclass, field

from nakagai.nlbuilder.prompt import render_system_prompt
from nakagai.strategies.composite import (
    describe_composite_spec, validate_composite_blocks, validate_composite_spec)
from nakagai.strategies.rules import describe_spec, validate_spec

MODEL = "claude-opus-4-8"
MAX_TOKENS = 8000


@dataclass
class CompileResult:
    spec: dict | None = None
    kind: str = "rules"
    readback: str = ""
    clarifications: list[str] = field(default_factory=list)
    not_expressible: str = ""
    attempts: int = 0
    model: str = ""
    error: str = ""
    usage: dict = field(default_factory=lambda: {
        "input_tokens": 0, "output_tokens": 0,
        "cache_read_tokens": 0, "cache_write_tokens": 0})


def _client_or_default(client):
    if client is not None:
        return client
    import anthropic
    return anthropic.Anthropic()


def _parse(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(text)


def _text(resp) -> str:
    return next(b.text for b in resp.content if getattr(b, "type", "") == "text")


def _add_usage(result: CompileResult, resp) -> None:
    u = getattr(resp, "usage", None)
    if u is None:
        return
    result.usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
    result.usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0
    result.usage["cache_read_tokens"] += getattr(u, "cache_read_input_tokens", 0) or 0
    result.usage["cache_write_tokens"] += getattr(u, "cache_creation_input_tokens", 0) or 0


def _check(kind: str, spec, members: dict | None):
    """(errors, describer) for the spec kind the model claims it produced.
    A composite needs the caller's member registry both to validate block
    references and to render the prompt, so without one the only honest answer
    is to send the model back to a single rules spec."""
    if kind == "composite":
        if not members:
            return (["composite specs are not available here; "
                     "return a single rules spec instead"], describe_composite_spec)
        errors = (validate_composite_spec(spec, members, allow_refs=False)
                  or validate_composite_blocks(spec, members))
        return errors, describe_composite_spec
    return validate_spec(spec), describe_spec


def compile_strategy(description: str, current_spec: dict | None = None,
                     client=None, model: str = MODEL,
                     max_retries: int = 2, members: dict | None = None) -> CompileResult:
    client = _client_or_default(client)
    system = [{"type": "text", "text": render_system_prompt(members),
               "cache_control": {"type": "ephemeral"}}]
    user = description.strip()
    if current_spec is not None:
        user += "\n\nCurrent spec (revise it rather than starting over):\n" \
                + json.dumps(current_spec)
    messages = [{"role": "user", "content": user}]
    result = CompileResult()
    result.model = model
    last_errors: list[str] = []
    for _ in range(max_retries + 1):
        result.attempts += 1
        try:
            resp = client.messages.create(
                model=model, max_tokens=MAX_TOKENS, system=system,
                thinking={"type": "adaptive"}, messages=messages)
        except Exception as e:
            result.error = f"model call failed: {e}"
            return result
        _add_usage(result, resp)
        raw = ""
        try:
            raw = _text(resp)
            doc = _parse(raw)
        except (json.JSONDecodeError, StopIteration):
            last_errors = ["reply was not a single JSON object"]
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "That was not parseable JSON. Reply with "
                                            "exactly one JSON object per the contract."}]
            continue
        if doc.get("not_expressible"):
            result.not_expressible = str(doc["not_expressible"])
            return result
        spec = doc.get("spec")
        kind = doc.get("kind") or "rules"
        if kind not in ("rules", "composite"):
            kind = "rules"
        errors, describe = _check(kind, spec, members)
        if not errors:
            result.spec = spec
            result.kind = kind
            result.readback = describe(spec)
            result.clarifications = [str(c) for c in doc.get("clarifications", [])]
            return result
        last_errors = errors
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "The spec failed validation. Fix exactly these "
                                        "errors and resend the full JSON object:\n- "
                                        + "\n- ".join(errors)}]
    result.not_expressible = ("could not produce a valid spec; last errors: "
                              + "; ".join(last_errors))
    return result
