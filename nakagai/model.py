"""The one thing a compiler needs from a model, and nothing else.

A compiler used to be handed an `anthropic.Anthropic` and call
`client.messages.create`, which put a vendor SDK in this package's dependency
list and made "run the builder" mean "have an Anthropic key". Swapping that for
another SDK would have moved the coupling rather than removed it.

So a compiler is handed a CALLABLE instead. It knows nothing about providers,
HTTP, or credentials: it passes a system prompt, some messages and a token
ceiling, and gets back text, token counts, a cost numerator, and whether any
provider delivery remains uncertain. The numerator is trillionths of a dollar,
and its provenance says whether every completed attempt reported its exact
provider bill. The platform that owns the ledger builds the callable; anyone
else can pass their own, and `openrouter_complete` below is the
batteries-included one.
"""

from __future__ import annotations

import json
import os
from decimal import Decimal
from typing import NamedTuple, Protocol


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_COST_SCALE = 1_000_000_000_000


class ModelReply(NamedTuple):
    """One model answer, and what it cost, including when it failed.

    `error` is empty on success. It is NOT the only thing a caller reads on
    failure: the token counts are populated whenever a response ARRIVED, even
    a failing one, because a call that reached the provider was billed by it.
    Reporting zero there would record a billed call as free, and a caller
    holding a reserve against it would never settle the real amount.

    A transport failure has no arrived response, so it reports zero observed
    usage. Delivery is uncertain, however, because the provider may have
    accepted the request before the transport failed.

    `cost_numerator` is in trillionths of a dollar. `cost_from_provider` is
    true only when this arrived response reported its exact bill.
    `spend_unknown` is true when an attempted call may have an unobserved bill.
    """

    text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_numerator: int
    cost_from_provider: bool
    spend_unknown: bool
    error: str


class Complete(Protocol):
    """What both compilers are handed. Keyword-only, so a positional mistake
    is a TypeError at the call rather than a system prompt sent as a message."""

    def __call__(self, *, system: str, messages: list[dict],
                 max_tokens: int) -> ModelReply: ...


def openrouter_complete(*, model: str, api_key: str | None = None,
                        provider: dict | None = None,
                        timeout_s: float = 120.0) -> Complete:
    """A `Complete` over OpenRouter's chat completions.

    ONE HTTP call per call, and no retries, ever. That is not a default this
    function picked: a caller pricing a reserve needs one call to equal one
    billable attempt, and an SDK whose retry default moved from 2 to 4 would
    let five attempts run against a reserve priced for three. A compiler's own
    retry rounds are its business and it counts them itself.

    `provider` is passed through untouched so a caller can pin an exact
    endpoint. A model id is not a machine: the same id is served by many
    providers at different quantizations, and default routing has made a good
    model look broken before.

    The key is read at CALL time, not here, so constructing this costs nothing
    without one and keyless development and CI reach the compilers exactly as
    they did.
    """
    import httpx

    def complete(*, system: str, messages: list[dict],
                 max_tokens: int) -> ModelReply:
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            return ModelReply("", 0, 0, 0, 0, 0, False, False,
                              "no OPENROUTER_API_KEY is configured")
        body: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, *messages],
            # Ask the router to report what it actually charged, so a caller
            # that prices from a table can tell an estimate from a bill.
            "usage": {"include": True},
        }
        if provider is not None:
            body["provider"] = provider
        try:
            with httpx.Client(timeout=timeout_s) as http:
                resp = http.post(
                    OPENROUTER_URL, json=body,
                    headers={"Authorization": f"Bearer {key}"})
        except Exception as e:  # noqa: BLE001 - transport, nothing arrived
            return ModelReply("", 0, 0, 0, 0, 0, False, True,
                              f"model call failed: {e}")
        return _reply(resp)

    return complete


def _cost_numerator(usage: dict) -> tuple[int, bool]:
    cost = usage.get("cost")
    if not isinstance(cost, (int, float)) or isinstance(cost, bool):
        return 0, False
    amount = Decimal(repr(cost))
    if not amount.is_finite() or amount < 0:
        return 0, False
    return int(amount * _COST_SCALE), True


def _token_count(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _reply(resp) -> ModelReply:
    """One arrived response, read for its text AND its cost.

    Read in that order deliberately: usage first, so that a malformed body
    cannot lose the counts on the way to raising about the text. Everything
    after a response arrives is a fact about a call that already happened, and
    nothing here may raise.
    """
    try:
        doc = resp.json()
    except Exception:  # noqa: BLE001
        doc = {}
    if not isinstance(doc, dict):
        doc = {}
    usage = doc.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    if not isinstance(prompt_details, dict):
        prompt_details = {}
    cost_numerator, cost_from_provider = _cost_numerator(usage)
    counts = (
        _token_count(usage.get("prompt_tokens")),
        _token_count(usage.get("completion_tokens")),
        _token_count(prompt_details.get("cached_tokens")),
        0,
    )
    status = getattr(resp, "status_code", 0)
    if status < 200 or status >= 300:
        detail = doc.get("error") or {}
        message = detail.get("message") if isinstance(detail, dict) else detail
        return ModelReply(
            "", *counts, cost_numerator, cost_from_provider, False,
            f"model call failed: HTTP {status}: {message or 'no detail'}")
    try:
        choice = (doc.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
    except Exception:  # noqa: BLE001
        return ModelReply("", *counts, cost_numerator, cost_from_provider, False,
                          "model reply had no readable content")
    if not isinstance(text, str):
        text = json.dumps(text)
    return ModelReply(text, *counts, cost_numerator, cost_from_provider, False, "")
