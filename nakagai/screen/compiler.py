"""English -> validated ScreenSpec v1, via the model callable in
`nakagai.model` with the same validator-driven retry loop the strategy builder
uses. The small helpers and the CompileResult shape are shared with nlbuilder;
the loop itself is a thin copy because the two prompts' contracts differ
(extract a common core only if a third compiler ever appears)."""

from nakagai.nlbuilder.compiler import (
    CandidateNormalizer, CandidateValidator, CompileResult,
    _AggregateBillingEvidence, _add_usage, _client_or_default, _parse,
    _reply_attempted,
)
from nakagai.screen.prompt import render_screen_prompt
from nakagai.screen.spec import describe_screen, validate_screen_spec
from nakagai.strategies.rules.vocabulary import Vocabulary, resolve_vocabulary

MODEL = "deepseek/deepseek-v4-flash-0731"
MAX_TOKENS = 4000


def compile_screen(description: str, client=None, model: str = MODEL,
                   max_retries: int = 2, *,
                   vocabulary: Vocabulary | None = None,
                   prompt_policy: str = "",
                   candidate_normalizer: CandidateNormalizer | None = None,
                   candidate_validator: CandidateValidator | None = None,
                   ) -> CompileResult:
    # One vocabulary for the whole loop. The prompt advertises exactly the
    # terms the validator will accept and the readback will render, so an
    # injected term cannot be offered to the model and then refused on the way
    # back (or accepted and then rendered against a name the renderer lacks).
    vocabulary = resolve_vocabulary(vocabulary)
    complete = _client_or_default(client, model)
    system = render_screen_prompt(vocabulary)
    if prompt_policy.strip():
        system += "\n\n" + prompt_policy.strip()
    messages = [{"role": "user", "content": description.strip()}]
    result = CompileResult()
    result.model = model
    billing_evidence: _AggregateBillingEvidence | None = None
    last_errors: list[str] = []
    for _ in range(max_retries + 1):
        result.attempts += 1
        result.retries_taken = result.attempts - 1
        try:
            reply = complete(system=system, messages=messages,
                             max_tokens=MAX_TOKENS)
        except Exception as e:
            # `Complete` says a failure comes back as `ModelReply.error`, but
            # the callable is the caller's, so this loop cannot assume it
            # obeys. A raise escaping here would take every count already
            # accumulated by the earlier rounds with it.
            result.error = f"model call failed: {e}"
            result.cost_from_provider = False
            result.spend_unknown = True
            return result
        if not _reply_attempted(reply):
            result.attempts -= 1
            result.retries_taken = max(result.attempts - 1, 0)
            result.error = reply.error
            return result
        # Bill first, read second: the counts are a fact about a call that
        # already happened, and a failure must not lose them on the way out.
        billing_evidence = _add_usage(result, reply, billing_evidence)
        if reply.error:
            result.error = reply.error
            return result
        raw = reply.text
        try:
            doc = _parse(raw)
        except (ValueError, RecursionError):
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
        if not errors and candidate_normalizer is not None:
            spec = candidate_normalizer("screen", spec)
            errors = validate_screen_spec(spec, vocabulary=vocabulary)
        if not errors and candidate_validator is not None:
            callback_errors = candidate_validator("screen", spec)
            errors = ([callback_errors] if isinstance(callback_errors, str)
                      else list(callback_errors))
        if not errors:
            result.spec = spec
            result.readback = describe_screen(spec, vocabulary=vocabulary)
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
