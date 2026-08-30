"""compile_screen: the NL->ScreenSpec loop, exercised with a fake model."""

import json

from nakagai.model import ModelReply
from nakagai.screen import compiler
from nakagai.screen.compiler import MAX_TOKENS, MODEL, compile_screen
from nakagai.screen.prompt import render_screen_prompt

GOOD_SPEC = {"version": 1, "tf": "1d",
             "conditions": {"all": [{"lhs": {"ind": "rsi", "n": 14}, "op": "<", "rhs": 30}]}}

RELATIVE_SCOPE_SPEC = {
    "version": 1,
    "tf": "15m",
    "conditions": {"all": [
        {
            "lhs": {"src": "close"},
            "op": ">",
            "rhs": {"src": "close", "sym": "SPY"},
        },
        {
            "lhs": {
                "ind": "sma", "n": 20, "of": {"src": "close"},
                "sym": "SPY", "tf": "15m",
            },
            "op": ">",
            "rhs": {
                "ind": "sma", "n": 20,
                "of": {"src": "close", "sym": "QQQ", "tf": "1d"},
                "sym": "SPY", "tf": "15m",
            },
        },
    ]},
}

RELATIVE_SCOPE_READBACK = (
    "Screen on 15m bars, matching symbols where ALL of:\n"
    "  - close is above SPY:close\n"
    "  - SPY:sma(20)[15m] is above "
    "SPY:sma(20, of=QQQ:close[1d])[15m]"
)


class _FakeModel:
    """A `Complete` over a fixed list of replies, recording every call.

    Keyword-only like the protocol it stands in for, so a positional mistake
    in the compiler is a TypeError here rather than a system prompt quietly
    delivered as a message.
    """

    def __init__(self, replies, tokens=(10, 5, 0, 0), error="",
                 cost_numerator=0, cost_from_provider=False,
                 spend_unknown=False):
        self._replies = list(replies)
        self._tokens = tokens
        self._error = error
        self._cost_numerator = cost_numerator
        self._cost_from_provider = cost_from_provider
        self._spend_unknown = spend_unknown
        self.calls = []

    def __call__(self, *, system, messages, max_tokens):
        self.calls.append({"system": system, "messages": messages,
                           "max_tokens": max_tokens})
        return ModelReply(self._replies.pop(0), *self._tokens,
                          self._cost_numerator, self._cost_from_provider,
                          self._spend_unknown, self._error)


# Production break caught: the shared usage adder could omit provider cost.
def test_compile_screen_happy_path():
    client = _FakeModel(
        [json.dumps({"spec": GOOD_SPEC, "clarifications": ["assumed daily"]})],
        cost_numerator=1_111_111, cost_from_provider=True)
    r = compile_screen("oversold on the daily", client=client)
    assert r.spec == GOOD_SPEC
    assert r.readback.startswith("Screen on 1d bars")
    assert r.clarifications == ["assumed daily"]
    assert r.attempts == 1 and r.error == ""
    assert (r.cost_numerator, r.cost_from_provider) == (1_111_111, True)
    assert r.spend_unknown is False
    assert r.retries_taken == 0


def test_screen_clarifications_accept_only_a_list():
    for sent in (None, "assumed daily"):
        client = _FakeModel([
            json.dumps({"spec": GOOD_SPEC, "clarifications": sent})])
        result = compile_screen("oversold", client=client)
        assert result.spec == GOOD_SPEC
        assert result.clarifications == []


def test_blank_prompt_policy_keeps_the_screen_prompt_byte_identical():
    expected = render_screen_prompt().encode("utf-8")
    for policy in ("", " \n\t "):
        client = _FakeModel([json.dumps({"spec": GOOD_SPEC})])
        compile_screen("oversold", client=client, prompt_policy=policy)
        assert client.calls[0]["system"].encode("utf-8") == expected


def test_prompt_policy_is_appended_once_before_the_first_screen_call():
    policy = "# House policy\n- one rule"
    bad = {"version": 7}
    client = _FakeModel([
        json.dumps({"spec": bad}),
        json.dumps({"spec": GOOD_SPEC}),
    ])
    result = compile_screen(
        "oversold", client=client, prompt_policy=f"  {policy}\n")
    expected = render_screen_prompt() + "\n\n" + policy
    assert result.attempts == 2
    assert [call["system"] for call in client.calls] == [
        expected, expected,
    ]
    assert expected.count(policy) == 1


def test_plain_string_candidate_validator_is_one_complete_error():
    seen = 0

    def retry_validator(kind, spec):
        nonlocal seen
        seen += 1
        assert kind == "screen"
        assert spec == GOOD_SPEC
        return "house policy failed" if seen == 1 else []

    client = _FakeModel([
        json.dumps({"spec": GOOD_SPEC}),
        json.dumps({"spec": GOOD_SPEC}),
    ])
    result = compile_screen(
        "oversold", client=client, max_retries=1,
        candidate_validator=retry_validator)
    assert result.spec == GOOD_SPEC
    assert client.calls[1]["messages"][-1]["content"] == (
        "The spec failed validation. Fix exactly these errors and resend the "
        "full JSON object:\n- house policy failed"
    )

    terminal = compile_screen(
        "oversold",
        client=_FakeModel([json.dumps({"spec": GOOD_SPEC})]),
        max_retries=0,
        candidate_validator=lambda kind, spec: "house policy failed",
    )
    assert terminal.not_expressible == (
        "could not produce a valid spec; last errors: house policy failed"
    )
    assert "h; o; u; s; e" not in terminal.not_expressible

    ordered = compile_screen(
        "oversold",
        client=_FakeModel([json.dumps({"spec": GOOD_SPEC})]),
        max_retries=0,
        candidate_validator=lambda kind, spec: ("first error", "second error"),
    )
    assert ordered.not_expressible == (
        "could not produce a valid spec; last errors: first error; second error"
    )


def test_screen_normalizer_is_revalidated_before_the_caller_validator():
    validator_calls = []
    client = _FakeModel([
        json.dumps({"spec": GOOD_SPEC}),
        json.dumps({"spec": GOOD_SPEC}),
    ])
    calls = 0

    def normalize(kind, spec):
        nonlocal calls
        calls += 1
        assert kind == "screen"
        return {"version": 7} if calls == 1 else GOOD_SPEC

    result = compile_screen(
        "oversold", client=client, max_retries=1,
        candidate_normalizer=normalize,
        candidate_validator=lambda kind, spec: validator_calls.append((kind, spec)) or [],
    )
    assert result.spec == GOOD_SPEC
    assert result.attempts == 2
    assert "version" in client.calls[1]["messages"][-1]["content"]
    assert validator_calls == [("screen", GOOD_SPEC)]


def test_screen_caller_channels_share_the_default_three_attempt_stream():
    replies = [
        json.dumps({"spec": {"version": 7}}),
        json.dumps({"spec": GOOD_SPEC}),
        json.dumps({"spec": GOOD_SPEC}),
    ]
    normalized = []

    def normalize(kind, spec):
        normalized.append((kind, spec))
        return spec

    result = compile_screen(
        "oversold", client=_FakeModel(replies),
        candidate_normalizer=normalize,
        candidate_validator=lambda kind, spec: (
            "integer policy failed", "symbol policy failed"),
    )
    assert result.attempts == 3
    assert result.usage == {
        "input_tokens": 30,
        "output_tokens": 15,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }
    assert normalized == [("screen", GOOD_SPEC), ("screen", GOOD_SPEC)]
    assert result.not_expressible == (
        "could not produce a valid spec; last errors: "
        "integer policy failed; symbol policy failed"
    )


def test_screen_readback_uses_the_native_valid_normalized_candidate():
    seen = []
    client = _FakeModel([json.dumps({"spec": GOOD_SPEC})])
    result = compile_screen(
        "compare the market", client=client,
        candidate_normalizer=lambda kind, spec: RELATIVE_SCOPE_SPEC,
        candidate_validator=lambda kind, spec: seen.append((kind, spec)) or [],
    )
    assert seen == [("screen", RELATIVE_SCOPE_SPEC)]
    assert result.spec == RELATIVE_SCOPE_SPEC
    assert result.readback == RELATIVE_SCOPE_READBACK


def test_compile_screen_retries_on_validator_errors_and_feeds_them_back():
    bad = {"version": 1, "conditions": {"all": [{"lhs": {"ind": "nope"}, "op": "<", "rhs": 1}]}}
    client = _FakeModel([json.dumps({"spec": bad}), json.dumps({"spec": GOOD_SPEC})])
    r = compile_screen("oversold", client=client)
    assert r.spec == GOOD_SPEC and r.attempts == 2
    retry_user = client.calls[1]["messages"][-1]["content"]
    assert "failed validation" in retry_user and "conditions.all[0]" in retry_user


def test_screen_reply_with_an_integer_past_the_decoder_limit_retries():
    huge = '{"spec": ' + "9" * 5000 + "}"
    client = _FakeModel([huge, json.dumps({"spec": GOOD_SPEC})])
    result = compile_screen("oversold", client=client)
    assert result.spec == GOOD_SPEC
    assert result.attempts == 2
    assert "not parseable JSON" in client.calls[1]["messages"][-1]["content"]


def test_screen_reply_nested_past_the_decoder_limit_retries():
    nested = "[" * 10_000 + "]" * 10_000
    client = _FakeModel([
        '{"spec": ' + nested + "}",
        json.dumps({"spec": GOOD_SPEC}),
    ])
    result = compile_screen("oversold", client=client)
    assert result.spec == GOOD_SPEC
    assert result.attempts == 2
    assert "not parseable JSON" in client.calls[1]["messages"][-1]["content"]


def test_compile_screen_not_expressible_passes_through():
    client = _FakeModel([json.dumps({"not_expressible": "no earnings data"})])
    r = compile_screen("earnings beats", client=client)
    assert r.not_expressible == "no earnings data" and r.spec is None


def test_compile_screen_gives_up_after_retries():
    bad = json.dumps({"spec": {"version": 7}})
    client = _FakeModel([bad, bad, bad])
    r = compile_screen("x", client=client, max_retries=2)
    assert r.spec is None and "could not produce a valid spec" in r.not_expressible


def test_compile_screen_bills_a_failing_reply_that_arrived():
    """A response that arrived was billed, even when it carried no spec.

    The counts must land in `usage` BEFORE the error short-circuits the loop.
    Dropping that leaves a caller settling a reserve against zeros, which
    records a real charge as free.
    """
    failing = _FakeModel([""], tokens=(11, 7, 3, 2),
                         error="model call failed: HTTP 500: overloaded")
    r = compile_screen("x", client=failing)
    assert r.error == "model call failed: HTTP 500: overloaded"
    assert r.spec is None and r.attempts == 1
    assert r.usage == {"input_tokens": 11, "output_tokens": 7,
                       "cache_read_tokens": 3, "cache_write_tokens": 2}


def test_compile_screen_reports_zeros_when_nothing_arrived():
    """No response supplied usage, but the attempted delivery may be billed."""
    never_sent = _FakeModel([""], tokens=(0, 0, 0, 0),
                            error="model call failed: connection refused",
                            spend_unknown=True)
    r = compile_screen("x", client=never_sent)
    assert r.error == "model call failed: connection refused"
    assert r.usage == {"input_tokens": 0, "output_tokens": 0,
                       "cache_read_tokens": 0, "cache_write_tokens": 0}
    assert r.spend_unknown is True
    assert r.retries_taken == 0


# Production break caught: a raised retry could leave a partial bill marked exact.
def test_compile_screen_keeps_earlier_usage_when_a_callable_raises():
    """A caller's own `Complete` may break its contract and raise.

    The rounds before it were still billed, so the failure comes back as
    `result.error` rather than as an exception that carries those counts out
    of the function with it.
    """
    class _RaisesOnTheSecondRound(_FakeModel):
        def __call__(self, *, system, messages, max_tokens):
            if self.calls:
                raise RuntimeError("socket closed")
            return super().__call__(system=system, messages=messages,
                                    max_tokens=max_tokens)

    client = _RaisesOnTheSecondRound(
        [json.dumps({"spec": {"version": 7}})], cost_numerator=1_111_111,
        cost_from_provider=True)
    r = compile_screen("x", client=client)
    assert r.error == "model call failed: socket closed"
    assert r.attempts == 2 and r.spec is None
    assert r.usage == {"input_tokens": 10, "output_tokens": 5,
                       "cache_read_tokens": 0, "cache_write_tokens": 0}
    assert r.cost_numerator == 1_111_111
    assert r.cost_from_provider is False
    assert r.spend_unknown is True
    assert r.retries_taken == 1


def test_compile_screen_sends_a_plain_string_system_and_the_token_ceiling():
    client = _FakeModel([json.dumps({"spec": GOOD_SPEC})])
    compile_screen("oversold", client=client)
    call = client.calls[0]
    assert isinstance(call["system"], str)
    assert call["max_tokens"] == MAX_TOKENS
    assert call["messages"] == [{"role": "user", "content": "oversold"}]


def test_compile_screen_model_argument_selects_the_callable(monkeypatch):
    """`model` names the model, it is not just the label on the result.

    It used to ride along on every `messages.create` call. A `Complete` closes
    over its model id instead, so the argument has to reach the place the
    callable is built or it silently becomes decoration on a compile that ran
    against the default.
    """
    seen = []
    client = _FakeModel([json.dumps({"spec": GOOD_SPEC})])

    def _spy(passed, model):
        seen.append((passed, model))
        return client

    monkeypatch.setattr(compiler, "_client_or_default", _spy)
    result = compile_screen("oversold", model="deepseek/some-other-id")
    assert seen == [(None, "deepseek/some-other-id")]
    assert result.model == "deepseek/some-other-id"


def test_compile_screen_no_longer_defaults_to_an_anthropic_model():
    """The point of the port: nothing here names Claude or needs an SDK."""
    assert "claude" not in MODEL and "/" in MODEL
    client = _FakeModel([json.dumps({"spec": GOOD_SPEC})])
    assert compile_screen("oversold", client=client).model == MODEL


def test_screen_prompt_teaches_the_conditions_only_contract():
    p = render_screen_prompt()
    assert '"version": 1' in p
    assert "conditions" in p and "rsi" in p and "donchian" in p
    assert '"sym"?: <SYMBOL>' in p
    assert "[A-Z][A-Z0-9.-]{0,9}" in p
    assert "risk" not in p.lower().replace("not_expressible", "")
