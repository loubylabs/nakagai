"""`nakagai.model`: the callable a compiler is handed, and what it must report.

Two properties carry everything else here.

**A call that arrived was billed.** The counts come back on a FAILING response
too, because the provider charged for it either way, and a caller holding a
reserve settles against what it reports. Reporting zero on failure records a
billed call as free.

**One call is one call.** A compiler counts its own retry rounds and prices a
reserve from that count, so a transport that retried underneath would let more
attempts run than the reserve was priced for.
"""

import json

import pytest

from nakagai import model


class _Resp:
    def __init__(self, status, doc):
        self.status_code = status
        self._doc = doc

    def json(self):
        return self._doc


def _ok(text="{}", **usage):
    return _Resp(200, {"choices": [{"message": {"content": text}}],
                       "usage": {"prompt_tokens": 11, "completion_tokens": 7,
                                 **usage}})


def _complete(monkeypatch, responses, **kwargs):
    """Build the callable over a fake transport that records its calls."""
    calls = []

    class _Client:
        def __init__(self, timeout=None):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def post(self, url, json=None, headers=None):
            calls.append({"url": url, "body": json, "headers": headers})
            return responses[len(calls) - 1]

    import httpx
    monkeypatch.setattr(httpx, "Client", _Client)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")
    return model.openrouter_complete(model="deepseek/x", **kwargs), calls


def test_one_compile_call_makes_exactly_one_http_call(monkeypatch):
    """No retries, ever. A reserve is priced from the compiler's own attempt
    count, so a transport retrying underneath spends more than was reserved."""
    complete, calls = _complete(monkeypatch, [_ok()])

    complete(system="s", messages=[{"role": "user", "content": "u"}],
             max_tokens=100)

    assert len(calls) == 1


def test_a_failing_response_still_reports_what_it_billed(monkeypatch):
    """The property this module exists for.

    A 400 that arrived carrying a usage block was charged. Reporting zeros
    would settle a billed call as free and leave its reserve held."""
    failed = _Resp(400, {"error": {"message": "context length exceeded"},
                         "usage": {"prompt_tokens": 900,
                                   "completion_tokens": 3}})
    complete, _ = _complete(monkeypatch, [failed])

    reply = complete(system="s", messages=[], max_tokens=10)

    assert reply.error
    assert "400" in reply.error
    assert "context length exceeded" in reply.error
    assert reply.input_tokens == 900, "a billed call must not report zero cost"
    assert reply.output_tokens == 3
    assert reply.text == ""
    assert reply.spend_unknown is False


# Production break caught: a transport exception could hide an uncertain bill.
def test_a_transport_failure_reports_zero_observed_usage_and_unknown_spend(
        monkeypatch):
    """No response supplied usage, but delivery may have reached the provider."""
    import httpx

    class _Boom:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def post(self, *_a, **_k):
            raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(httpx, "Client", _Boom)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")
    complete = model.openrouter_complete(model="deepseek/x")

    reply = complete(system="s", messages=[], max_tokens=10)

    assert "model call failed" in reply.error
    assert reply.input_tokens == 0 and reply.output_tokens == 0
    assert (reply.cost_numerator, reply.cost_from_provider) == (0, False)
    assert reply.spend_unknown is True


def test_the_system_prompt_rides_as_a_message_not_a_block_list(monkeypatch):
    """A plain string, because that is what the callable's signature says.

    The Anthropic shape was a list of blocks carrying `cache_control`, and
    passing one here would serialize an object where a string belongs."""
    complete, calls = _complete(monkeypatch, [_ok()])

    complete(system="the system prompt",
             messages=[{"role": "user", "content": "hello"}], max_tokens=10)

    sent = calls[0]["body"]["messages"]
    assert sent[0] == {"role": "system", "content": "the system prompt"}
    assert sent[1] == {"role": "user", "content": "hello"}


def test_an_exact_provider_pin_is_passed_through_untouched(monkeypatch):
    """A model id is not a machine. The caller's pin reaches the router."""
    pin = {"order": ["alibaba"], "allow_fallbacks": False}
    complete, calls = _complete(monkeypatch, [_ok()], provider=pin)

    complete(system="s", messages=[], max_tokens=10)

    assert calls[0]["body"]["provider"] == pin


def test_no_pin_sends_no_provider_key(monkeypatch):
    complete, calls = _complete(monkeypatch, [_ok()])

    complete(system="s", messages=[], max_tokens=10)

    assert "provider" not in calls[0]["body"]


# Production break caught: a locally refused call could claim a provider bill.
def test_a_missing_key_refuses_without_calling_out(monkeypatch):
    """Constructing costs nothing without a key; the refusal is at call time."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    complete = model.openrouter_complete(model="deepseek/x")

    reply = complete(system="s", messages=[], max_tokens=10)

    assert "OPENROUTER_API_KEY" in reply.error
    assert (reply.cost_numerator, reply.cost_from_provider) == (0, False)
    assert reply.spend_unknown is False


def test_the_router_is_asked_to_report_what_it_charged(monkeypatch):
    complete, calls = _complete(monkeypatch, [_ok()])

    complete(system="s", messages=[], max_tokens=10)

    assert calls[0]["body"]["usage"] == {"include": True}


def test_cached_prompt_tokens_are_reported_when_the_router_sends_them(monkeypatch):
    complete, _ = _complete(
        monkeypatch, [_ok(prompt_tokens_details={"cached_tokens": 40})])

    reply = complete(system="s", messages=[], max_tokens=10)

    assert reply.cache_read_tokens == 40


# Production break caught: malformed content could discard arrived usage.
def test_an_unreadable_body_keeps_the_counts_it_could_read(monkeypatch):
    """Nothing after a response arrives may raise, and nothing may lose the
    cost on its way to complaining about the text."""
    weird = _Resp(200, {"usage": {"prompt_tokens": 5, "completion_tokens": 2}})
    complete, _ = _complete(monkeypatch, [weird])

    reply = complete(system="s", messages=[], max_tokens=10)

    assert reply.input_tokens == 5 and reply.output_tokens == 2
    assert reply.text == ""
    assert reply.spend_unknown is False


# Production break caught: arrival alone could clear an unreadable bill.
def test_unreadable_json_has_unknown_spend():
    class _Unreadable:
        status_code = 200

        def json(self):
            raise ValueError("not JSON")

    reply = model._reply(_Unreadable())

    assert (reply.input_tokens, reply.output_tokens) == (0, 0)
    assert (reply.cost_numerator, reply.cost_from_provider) == (0, False)
    assert reply.spend_unknown is True


# Production break caught: successful responses could discard usage.cost.
def test_an_arrived_reply_carries_the_routers_exact_bill():
    reply = model._reply(_ok(cost=0.00003256))

    assert reply.cost_numerator == 32_560_000
    assert reply.cost_from_provider is True
    assert reply.spend_unknown is False


# Production break caught: non-2xx responses could discard usage.cost.
def test_a_failing_response_keeps_its_exact_bill():
    reply = model._reply(_Resp(400, {
        "error": {"message": "bad request"},
        "usage": {"prompt_tokens": 9, "completion_tokens": 2,
                  "cost": 0.000001234567},
    }))

    assert reply.error
    assert reply.cost_numerator == 1_234_567
    assert reply.cost_from_provider is True
    assert reply.spend_unknown is False


@pytest.mark.parametrize("cost", [None, True, "0.000001", float("nan"),
                                  float("inf"), -0.000001, 10_000_000])
# Production break caught: malformed provider prices could raise or erase usage.
def test_an_invalid_provider_cost_keeps_arrived_token_counts(cost):
    reply = model._reply(_ok(cost=cost))

    assert (reply.cost_numerator, reply.cost_from_provider) == (0, False)
    assert (reply.input_tokens, reply.output_tokens) == (11, 7)


# Production break caught: a non-object JSON body could escape the model boundary.
def test_a_non_dictionary_response_returns_a_zero_usage_reply():
    reply = model._reply(_Resp(200, ["not", "an", "object"]))

    assert reply.text == "" and reply.error == ""
    assert (reply.input_tokens, reply.output_tokens) == (0, 0)
    assert (reply.cost_numerator, reply.cost_from_provider) == (0, False)
    assert reply.spend_unknown is True


@pytest.mark.parametrize("usage", [None, [], "missing"])
# Production break caught: missing usage could settle an arrived call at zero.
def test_missing_or_non_object_usage_has_unknown_spend(usage):
    doc = {"choices": [{"message": {"content": "{}"}}]}
    if usage is not None:
        doc["usage"] = usage

    reply = model._reply(_Resp(200, doc))

    assert (reply.input_tokens, reply.output_tokens) == (0, 0)
    assert reply.spend_unknown is True


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"prompt_tokens": 11}, (11, 0, 0, 0)),
        ({"completion_tokens": 7}, (0, 7, 0, 0)),
        ({"prompt_tokens": True, "completion_tokens": 7},
         (0, 7, 0, 0)),
        ({"prompt_tokens": 11, "completion_tokens": -1},
         (11, 0, 0, 0)),
        ({"prompt_tokens": 11, "completion_tokens": 7,
          "prompt_tokens_details": {"cached_tokens": "2",
                                    "cache_write_tokens": 3}},
         (11, 7, 0, 3)),
        ({"prompt_tokens": 11, "completion_tokens": 7,
          "prompt_tokens_details": {"cached_tokens": 2,
                                    "cache_write_tokens": 2_147_483_648}},
         (11, 7, 2, 0)),
    ],
)
# Production break caught: coerced or partial counters could pose as a bill.
def test_incomplete_or_invalid_counters_preserve_valid_lower_bounds(
        usage, expected):
    reply = model._reply(_Resp(200, {
        "choices": [{"message": {"content": "{}"}}],
        "usage": usage,
    }))

    assert (reply.input_tokens, reply.output_tokens,
            reply.cache_read_tokens, reply.cache_write_tokens) == expected
    assert reply.spend_unknown is True


@pytest.mark.parametrize("details", [None, [], {"cached_tokens": -1}])
def test_invalid_prompt_details_make_rate_table_evidence_incomplete(details):
    reply = model._reply(_Resp(200, {
        "choices": [{"message": {"content": "{}"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7,
                  "prompt_tokens_details": details},
    }))

    assert (reply.input_tokens, reply.output_tokens) == (11, 7)
    assert reply.spend_unknown is True


def test_prompt_subsets_cannot_exceed_the_reported_prompt_total():
    reply = model._reply(_Resp(200, {
        "choices": [{"message": {"content": "{}"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7,
                  "prompt_tokens_details": {"cached_tokens": 8,
                                            "cache_write_tokens": 4}},
    }))

    assert (reply.input_tokens, reply.output_tokens,
            reply.cache_read_tokens, reply.cache_write_tokens) == (11, 7, 8, 4)
    assert reply.spend_unknown is True


def test_complete_rate_table_counters_make_arrived_spend_known():
    reply = model._reply(_Resp(200, {
        "choices": [{"message": {"content": "{}"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7,
                  "prompt_tokens_details": {"cached_tokens": 2,
                                            "cache_write_tokens": 3}},
    }))

    assert (reply.input_tokens, reply.output_tokens,
            reply.cache_read_tokens, reply.cache_write_tokens) == (11, 7, 2, 3)
    assert (reply.cost_numerator, reply.cost_from_provider) == (0, False)
    assert reply.spend_unknown is False


# Production break caught: compilers could not distinguish the two complete
# settlement bases after combining retries.
def test_each_reply_exposes_whether_its_rate_table_evidence_is_complete():
    exact_only = model._reply(_Resp(200, {
        "choices": [{"message": {"content": "{}"}}],
        "usage": {"prompt_tokens": "invalid", "cost": 0.000001234567},
    }))
    counters_only = model._reply(_Resp(200, {
        "choices": [{"message": {"content": "{}"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }))

    assert exact_only.cost_from_provider is True
    assert exact_only.rate_table_complete is False
    assert exact_only.spend_unknown is False
    assert counters_only.cost_from_provider is False
    assert counters_only.rate_table_complete is True
    assert counters_only.spend_unknown is False


def test_http_error_uses_the_same_incomplete_evidence_rule():
    reply = model._reply(_Resp(429, {
        "error": {"message": "limited"},
        "usage": {"prompt_tokens": 11},
    }))

    assert reply.error
    assert reply.input_tokens == 11
    assert reply.spend_unknown is True


# Production break caught: malformed tokens could erase an arrived exact provider bill.
def test_malformed_token_counts_do_not_discard_an_arrived_exact_bill():
    reply = model._reply(_Resp(200, {
        "choices": [{"message": {"content": "{}"}}],
        "usage": {"prompt_tokens": "not an integer", "completion_tokens": 7,
                  "prompt_tokens_details": {"cached_tokens": []},
                  "cost": 0.000001234567},
    }))

    assert (reply.input_tokens, reply.output_tokens) == (0, 7)
    assert reply.cache_read_tokens == 0
    assert (reply.cost_numerator, reply.cost_from_provider) == (1_234_567, True)
    assert reply.spend_unknown is False
