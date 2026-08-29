"""English -> a validated RuleSpec v2 or a validated composite spec, via the
model callable in `nakagai.model` with a validator-driven retry loop. The
reply's "kind" selects the validator, rules or composite, and it stays the
single source of truth; the model is asked to fix precisely the errors it
reports. Model failures come back as CompileResult.error rather than an
exception, so usage accumulated across retries always survives."""

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

from nakagai.model import Complete, ModelReply, openrouter_complete
from nakagai.nlbuilder.prompt import render_system_prompt
from nakagai.strategies.composite import (
    describe_composite_spec, validate_composite_blocks, validate_composite_spec)
from nakagai.strategies.rules import describe_spec, validate_spec
from nakagai.strategies.rules.vocabulary import Vocabulary, resolve_vocabulary

MODEL = "deepseek/deepseek-v4-flash-0731"
MAX_TOKENS = 8000

# A model id is not a machine. The same id is served by many providers at
# different quantizations, and default routing has made a good model look
# broken here before, so the endpoint is pinned: this provider, no silent
# fallback to another, and it must honour the parameters we send rather than
# dropping the ones it does not implement.
PROVIDER = {"require_parameters": True, "order": ["alibaba"],
            "allow_fallbacks": False}

CandidateNormalizer: TypeAlias = Callable[[str, dict], dict]
CandidateValidator: TypeAlias = Callable[[str, dict], Sequence[str]]


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


def _client_or_default(client, model: str = MODEL) -> Complete:
    """The `Complete` this compile runs against.

    A caller may still supply its own, and the platform does: the one that
    meters and bills belongs to whoever owns the ledger, not to this package.
    Without one, the batteries-included OpenRouter callable stands in. It is
    built per compile rather than held at module scope because it closes over
    the model id, and `model` selects it exactly as it did when this function
    constructed a vendor SDK client.
    """
    if client is not None:
        return client
    return openrouter_complete(model=model, provider=PROVIDER)


def _parse(text: str) -> dict:
    """The reply as a JSON OBJECT, or a raise the retry loop already handles.

    `json.loads` happily returns a list, a string or a number for a reply that
    is valid JSON and not the contract, and the caller reads `.get` off the
    result immediately. That was an AttributeError out of the loop rather than
    the retry the model could have acted on, which is the same defect as a
    validator that raises: the loop exists because a model can emit anything.
    """
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    doc = json.loads(text)
    if not isinstance(doc, dict):
        raise json.JSONDecodeError(
            f"expected one JSON object, got {type(doc).__name__}", text, 0)
    return doc


def _add_usage(result: CompileResult, reply: ModelReply) -> None:
    """Add one reply's counts to the running total.

    Every reply that ARRIVED is added, a failing one included: the provider
    charged for it either way. Reporting zero there would record a billed call
    as free, and a caller settling a reserve against `usage` would never see
    the real amount. Only a transport failure that never reached a provider
    carries zeros, and it carries them honestly.
    """
    result.usage["input_tokens"] += reply.input_tokens
    result.usage["output_tokens"] += reply.output_tokens
    result.usage["cache_read_tokens"] += reply.cache_read_tokens
    result.usage["cache_write_tokens"] += reply.cache_write_tokens


def _check(kind: str, spec, plays: Mapping[str, Mapping] | None,
           vocabulary: Vocabulary):
    """(errors, describer) for the spec kind the model claims it produced.
    A composite needs the caller's declared world both to validate block
    references and to render the prompt, so without it the only honest answer
    is to send the model back to a single rules spec.

    `plays` goes to both validators unchanged, and core adds no name of its
    own. It used to union in a `rules` member here, which made the compiler
    accept a composite the caller could not build: core's own
    `catalog_definitions` registers no `rules`, so a leg core had invented came
    back clean from validation and then raised `unknown strategy 'rules'` at
    `CompositeStrategy` construction. Whether the bespoke leg exists is the
    caller's fact, and `render_system_prompt` teaches it on the same condition,
    so the prompt, the validator and the caller's registry stay one world."""
    if kind == "composite":
        if not plays:
            return (["composite specs are not available here; "
                     "return a single rules spec instead"], describe_composite_spec)
        errors = (validate_composite_spec(spec, plays, allow_refs=False)
                  or validate_composite_blocks(spec, plays, vocabulary))
        return errors, describe_composite_spec
    return validate_spec(spec, vocabulary), lambda value: describe_spec(value, vocabulary)


def compile_strategy(description: str, current_spec: dict | None = None,
                     client=None, model: str = MODEL,
                     max_retries: int = 2,
                     plays: Mapping[str, Mapping] | None = None, *,
                     vocabulary: Vocabulary | None = None,
                     prompt_policy: str = "",
                     candidate_normalizer: CandidateNormalizer | None = None,
                     candidate_validator: CandidateValidator | None = None,
                     ) -> CompileResult:
    vocabulary = resolve_vocabulary(vocabulary)
    complete = _client_or_default(client, model)
    system = render_system_prompt(plays, vocabulary=vocabulary)
    if prompt_policy.strip():
        system += "\n\n" + prompt_policy.strip()
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
            reply = complete(system=system, messages=messages,
                             max_tokens=MAX_TOKENS)
        except Exception as e:
            # `Complete` says a failure comes back as `ModelReply.error`, but
            # the callable is the caller's, so this loop cannot assume it obeys.
            # A raise here would throw away every count already accumulated.
            result.error = f"model call failed: {e}"
            return result
        # Bill first, read second. The counts are a fact about a call that
        # already happened, so nothing below may reach a return ahead of them.
        _add_usage(result, reply)
        if reply.error:
            result.error = reply.error
            return result
        raw = reply.text
        try:
            doc = _parse(raw)
        except (ValueError, RecursionError):
            # Two, and each for its own reason. JSONDecodeError is a ValueError
            # subclass, so naming ValueError catches both it and the BARE
            # ValueError `json.loads` raises for an integer past the
            # interpreter's digit limit. RecursionError is what a reply nested
            # thousands of levels deep produces, inside the decoder itself.
            # Both are the model sending something it can be asked to fix.
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
        errors, describe = _check(kind, spec, plays, vocabulary)
        if not errors and candidate_normalizer is not None:
            spec = candidate_normalizer(kind, spec)
            errors, describe = _check(kind, spec, plays, vocabulary)
        if not errors and candidate_validator is not None:
            callback_errors = candidate_validator(kind, spec)
            errors = ([callback_errors] if isinstance(callback_errors, str)
                      else list(callback_errors))
        if not errors:
            result.spec = spec
            result.kind = kind
            result.readback = describe(spec)
            # The model owns this field, so it is whatever the model sent. A
            # null or a string here used to raise AFTER validation passed,
            # which is the worst place: the spec was good and the request died.
            clarifications = doc.get("clarifications")
            result.clarifications = ([str(c) for c in clarifications]
                                     if isinstance(clarifications, list) else [])
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
