"""compile_screen: the NL->ScreenSpec loop, exercised with a fake client."""

import json
from types import SimpleNamespace

from nakagai.screen.compiler import compile_screen
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


class _FakeClient:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        text = self._replies.pop(0)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5,
                                  cache_read_input_tokens=0,
                                  cache_creation_input_tokens=0))


def test_compile_screen_happy_path():
    client = _FakeClient([json.dumps({"spec": GOOD_SPEC, "clarifications": ["assumed daily"]})])
    r = compile_screen("oversold on the daily", client=client)
    assert r.spec == GOOD_SPEC
    assert r.readback.startswith("Screen on 1d bars")
    assert r.clarifications == ["assumed daily"]
    assert r.attempts == 1 and r.error == ""


def test_blank_prompt_policy_keeps_the_screen_prompt_byte_identical():
    expected = render_screen_prompt().encode("utf-8")
    for policy in ("", " \n\t "):
        client = _FakeClient([json.dumps({"spec": GOOD_SPEC})])
        compile_screen("oversold", client=client, prompt_policy=policy)
        assert client.calls[0]["system"][0]["text"].encode("utf-8") == expected


def test_prompt_policy_is_appended_once_before_the_first_screen_call():
    policy = "# House policy\n- one rule"
    bad = {"version": 7}
    client = _FakeClient([
        json.dumps({"spec": bad}),
        json.dumps({"spec": GOOD_SPEC}),
    ])
    result = compile_screen(
        "oversold", client=client, prompt_policy=f"  {policy}\n")
    expected = render_screen_prompt() + "\n\n" + policy
    assert result.attempts == 2
    assert [call["system"][0]["text"] for call in client.calls] == [
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

    client = _FakeClient([
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
        client=_FakeClient([json.dumps({"spec": GOOD_SPEC})]),
        max_retries=0,
        candidate_validator=lambda kind, spec: "house policy failed",
    )
    assert terminal.not_expressible == (
        "could not produce a valid spec; last errors: house policy failed"
    )
    assert "h; o; u; s; e" not in terminal.not_expressible

    ordered = compile_screen(
        "oversold",
        client=_FakeClient([json.dumps({"spec": GOOD_SPEC})]),
        max_retries=0,
        candidate_validator=lambda kind, spec: ("first error", "second error"),
    )
    assert ordered.not_expressible == (
        "could not produce a valid spec; last errors: first error; second error"
    )


def test_screen_normalizer_is_revalidated_before_the_caller_validator():
    validator_calls = []
    client = _FakeClient([
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
        "oversold", client=_FakeClient(replies),
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
    client = _FakeClient([json.dumps({"spec": GOOD_SPEC})])
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
    client = _FakeClient([json.dumps({"spec": bad}), json.dumps({"spec": GOOD_SPEC})])
    r = compile_screen("oversold", client=client)
    assert r.spec == GOOD_SPEC and r.attempts == 2
    retry_user = client.calls[1]["messages"][-1]["content"]
    assert "failed validation" in retry_user and "conditions.all[0]" in retry_user


def test_compile_screen_not_expressible_passes_through():
    client = _FakeClient([json.dumps({"not_expressible": "no earnings data"})])
    r = compile_screen("earnings beats", client=client)
    assert r.not_expressible == "no earnings data" and r.spec is None


def test_compile_screen_gives_up_after_retries():
    bad = json.dumps({"spec": {"version": 7}})
    client = _FakeClient([bad, bad, bad])
    r = compile_screen("x", client=client, max_retries=2)
    assert r.spec is None and "could not produce a valid spec" in r.not_expressible


def test_compile_screen_api_failure_keeps_usage():
    class _Boom(_FakeClient):
        def _create(self, **kwargs):
            raise RuntimeError("api down")
    r = compile_screen("x", client=_Boom([]))
    assert r.error.startswith("model call failed") and r.spec is None


def test_screen_prompt_teaches_the_conditions_only_contract():
    p = render_screen_prompt()
    assert '"version": 1' in p
    assert "conditions" in p and "rsi" in p and "donchian" in p
    assert '"sym"?: <SYMBOL>' in p
    assert "[A-Z][A-Z0-9.-]{0,9}" in p
    assert "risk" not in p.lower().replace("not_expressible", "")
