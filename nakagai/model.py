"""The one thing a compiler needs from a model, and nothing else.

A compiler used to be handed an `anthropic.Anthropic` and call
`client.messages.create`, which put a vendor SDK in this package's dependency
list and made "run the builder" mean "have an Anthropic key". Swapping that for
another SDK would have moved the coupling rather than removed it.

So a compiler is handed a CALLABLE instead. It knows nothing about providers,
HTTP, credentials or money: it passes a system prompt, some messages and a
token ceiling, and gets back text and the counts its caller needs to bill. The
platform that owns the ledger builds the callable; anyone else can pass their
own, and `openrouter_complete` below is the batteries-included one.
"""

from __future__ import annotations

import json
import os
from typing import NamedTuple, Protocol


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class ModelReply(NamedTuple):
    """One model answer, and what it cost, including when it failed.

    `error` is empty on success. It is NOT the only thing a caller reads on
    failure: the token counts are populated whenever a response ARRIVED, even
    a failing one, because a call that reached the provider was billed by it.
    Reporting zero there would record a billed call as free, and a caller
    holding a reserve against it would never settle the real amount.

    A transport failure that never reached the provider is the other case, and
    it is the only one that reports zeros: nothing was billed because nothing
    arrived.
    """

    text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
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
            return ModelReply("", 0, 0, 0, 0, "no OPENROUTER_API_KEY is configured")
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
            return ModelReply("", 0, 0, 0, 0, f"model call failed: {e}")
        return _reply(resp)

    return complete


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
    usage = doc.get("usage") or {}
    counts = (
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
        int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
        0,
    )
    status = getattr(resp, "status_code", 0)
    if status < 200 or status >= 300:
        detail = doc.get("error") or {}
        message = detail.get("message") if isinstance(detail, dict) else detail
        return ModelReply(
            "", *counts,
            f"model call failed: HTTP {status}: {message or 'no detail'}")
    try:
        choice = (doc.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
    except Exception:  # noqa: BLE001
        return ModelReply("", *counts, "model reply had no readable content")
    if not isinstance(text, str):
        text = json.dumps(text)
    return ModelReply(text, *counts, "")
