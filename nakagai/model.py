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
_LEDGER_INTEGER_MAX = 2_147_483_647
_SIGNED_BIGINT_MAX = 2 ** 63 - 1


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
    `rate_table_complete` is true only when its validated counters can price
    the complete response from a conservative rate table.
    `spend_unknown` is true when an attempted call may have an unobserved bill.
    """

    text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_numerator: int
    cost_from_provider: bool
    rate_table_complete: bool
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
            return ModelReply("", 0, 0, 0, 0, 0, False, False, False,
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
            return ModelReply("", 0, 0, 0, 0, 0, False, False, True,
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
    numerator = amount * _COST_SCALE
    if numerator > _SIGNED_BIGINT_MAX:
        return 0, False
    return int(numerator), True


def _token_count(payload: dict, key: str, *, required: bool) -> tuple[int, bool]:
    """Return one safe ledger count and whether its evidence is valid."""
    if key not in payload:
        return 0, not required
    value = payload[key]
    if type(value) is not int or not 0 <= value <= _LEDGER_INTEGER_MAX:
        return 0, False
    return value, True


def _billing_evidence(raw_usage) -> tuple[tuple[int, int, int, int], int,
                                          bool, bool, bool]:
    """Classify the complete bill while retaining each valid lower bound.

    An exact provider cost settles the response on its own. Without one, the
    two required token counters and the optional prompt detail counters must
    form the same valid shape the platform can price from its rate table.
    Invalid fields contribute zero independently, so one malformed counter
    cannot erase valid counts beside it.
    """
    usage_valid = isinstance(raw_usage, dict)
    usage = raw_usage if usage_valid else {}
    cost_numerator, cost_from_provider = _cost_numerator(usage)

    input_tokens, input_valid = _token_count(
        usage, "prompt_tokens", required=True)
    output_tokens, output_valid = _token_count(
        usage, "completion_tokens", required=True)

    raw_details = usage.get("prompt_tokens_details", {})
    details_valid = isinstance(raw_details, dict)
    details = raw_details if details_valid else {}
    cache_read_tokens, cache_read_valid = _token_count(
        details, "cached_tokens", required=False)
    cache_write_tokens, cache_write_valid = _token_count(
        details, "cache_write_tokens", required=False)

    subsets_valid = cache_read_tokens + cache_write_tokens <= input_tokens
    counters_complete = (
        usage_valid
        and input_valid
        and output_valid
        and details_valid
        and cache_read_valid
        and cache_write_valid
        and subsets_valid
    )
    counts = (input_tokens, output_tokens,
              cache_read_tokens, cache_write_tokens)
    spend_unknown = not (cost_from_provider or counters_complete)
    return (counts, cost_numerator, cost_from_provider,
            counters_complete, spend_unknown)


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
    (counts, cost_numerator, cost_from_provider, rate_table_complete,
     spend_unknown) = (
        _billing_evidence(doc.get("usage"))
    )
    status = getattr(resp, "status_code", 0)
    if status < 200 or status >= 300:
        detail = doc.get("error") or {}
        message = detail.get("message") if isinstance(detail, dict) else detail
        return ModelReply(
            "", *counts, cost_numerator, cost_from_provider,
            rate_table_complete, spend_unknown,
            f"model call failed: HTTP {status}: {message or 'no detail'}")
    try:
        choice = (doc.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
    except Exception:  # noqa: BLE001
        return ModelReply("", *counts, cost_numerator, cost_from_provider,
                          rate_table_complete, spend_unknown,
                          "model reply had no readable content")
    if not isinstance(text, str):
        text = json.dumps(text)
    return ModelReply(text, *counts, cost_numerator, cost_from_provider,
                      rate_table_complete, spend_unknown, "")
