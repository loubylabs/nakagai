"""English -> validated ScreenSpec v1, via the Claude API with the same
validator-driven retry loop the strategy builder uses. The small helpers and
the CompileResult shape are shared with nlbuilder; the loop itself is a thin
copy because the two prompts' contracts differ (extract a common core only if
a third compiler ever appears)."""

import json

from nakagai.nlbuilder.compiler import (
    CompileResult, _add_usage, _client_or_default, _parse, _text,
)
from nakagai.screen.prompt import render_screen_prompt
from nakagai.screen.spec import describe_screen, validate_screen_spec
from nakagai.strategies.rules.vocabulary import Vocabulary, resolve_vocabulary

MODEL = "claude-opus-4-8"
MAX_TOKENS = 4000


def compile_screen(description: str, client=None, model: str = MODEL,
                   max_retries: int = 2, *,
                   vocabulary: Vocabulary | None = None) -> CompileResult:
    # One vocabulary for the whole loop. The prompt advertises exactly the
    # terms the validator will accept and the readback will render, so an
    # injected term cannot be offered to the model and then refused on the way
    # back (or accepted and then rendered against a name the renderer lacks).
    vocabulary = resolve_vocabulary(vocabulary)
    client = _client_or_default(client)
    system = [{"type": "text",
               "text": render_screen_prompt(vocabulary),
               "cache_control": {"type": "ephemeral"}}]
    messages = [{"role": "user", "content": description.strip()}]
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
        errors = validate_screen_spec(spec, vocabulary=vocabulary)
        if not errors:
            result.spec = spec
            result.readback = describe_screen(spec, vocabulary=vocabulary)
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
