"""compile_screen: the NL->ScreenSpec loop, exercised with a fake client."""

import json
from types import SimpleNamespace

from nakagai.screen.compiler import compile_screen
from nakagai.screen.prompt import render_screen_prompt

GOOD_SPEC = {"version": 1, "tf": "1d",
             "conditions": {"all": [{"lhs": {"ind": "rsi", "n": 14}, "op": "<", "rhs": 30}]}}


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
    assert "risk" not in p.lower().replace("not_expressible", "")
