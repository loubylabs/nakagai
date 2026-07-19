import json

from nakagai.nlbuilder.compiler import compile_strategy
from nakagai.nlbuilder.prompt import render_system_prompt

GOOD_SPEC = {"version": 2, "name": "dip", "timeframe": "1h",
             "long": {"all": [{"lhs": {"ind": "rsi", "n": 14}, "op": "crosses_above", "rhs": 30}]},
             "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
                      "target": {"kind": "rr", "rr": 2.0}}}


class _Block:
    type = "text"
    def __init__(self, text): self.text = text


class _Usage:
    def __init__(self, input_tokens=100, output_tokens=50,
                 cache_read_input_tokens=10, cache_creation_input_tokens=5):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens


class _Resp:
    def __init__(self, text, usage=None):
        self.content = [_Block(text)]
        self.usage = usage or _Usage()


class FakeClient:
    """Yields queued replies; records every request for prompt assertions."""
    def __init__(self, replies):
        self._replies = list(replies)
        self.requests = []
        self.messages = self
    def create(self, **kwargs):
        self.requests.append(kwargs)
        return _Resp(self._replies.pop(0))


def test_prompt_renders_registries_and_is_deterministic():
    p1, p2 = render_system_prompt(), render_system_prompt()
    assert p1 == p2
    for needle in ("crosses_above", "opening_range_high", "bars_since", "supertrend",
                   "time_stop", "not_expressible", '"version": 2'):
        assert needle in p1, needle


def test_happy_path_returns_spec_and_readback():
    client = FakeClient([json.dumps({"spec": GOOD_SPEC, "clarifications": ["defaulted timeframe to 1h"]})])
    res = compile_strategy("buy rsi dips", client=client)
    assert res.spec == GOOD_SPEC
    assert "dip" in res.readback
    assert res.clarifications == ["defaulted timeframe to 1h"]
    assert res.attempts == 1
    assert client.requests[0]["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_validation_errors_trigger_retry_with_error_feedback():
    bad = {**GOOD_SPEC, "timeframe": "4h"}
    client = FakeClient([json.dumps({"spec": bad}), json.dumps({"spec": GOOD_SPEC})])
    res = compile_strategy("buy rsi dips", client=client)
    assert res.spec == GOOD_SPEC and res.attempts == 2
    retry_text = json.dumps(client.requests[1]["messages"])
    assert "timeframe" in retry_text            # the validator errors were fed back


def test_gives_up_after_max_retries():
    bad = json.dumps({"spec": {**GOOD_SPEC, "timeframe": "4h"}})
    client = FakeClient([bad, bad, bad])
    res = compile_strategy("x", client=client, max_retries=2)
    assert res.spec is None and res.attempts == 3
    assert "timeframe" in res.not_expressible


def test_not_expressible_passthrough():
    client = FakeClient([json.dumps({"not_expressible": "renko bars are not supported"})])
    res = compile_strategy("renko magic", client=client)
    assert res.spec is None and "renko" in res.not_expressible


def test_json_fences_are_tolerated():
    client = FakeClient(["```json\n" + json.dumps({"spec": GOOD_SPEC}) + "\n```"])
    assert compile_strategy("x", client=client).spec == GOOD_SPEC


def test_revision_includes_current_spec_in_user_turn():
    client = FakeClient([json.dumps({"spec": GOOD_SPEC})])
    compile_strategy("tighten the stop", current_spec=GOOD_SPEC, client=client)
    assert "current spec" in json.dumps(client.requests[0]["messages"]).lower()


class _NoTextResp:
    """A response with no text content block at all: a thinking-only turn.
    `_text` raises StopIteration on this; the except clause names
    StopIteration precisely to turn it into a retry, not a crash."""
    def __init__(self):
        self.content = []


class _RawClient:
    """Like FakeClient, but hands back exactly what's queued instead of
    wrapping every reply in a text _Resp: the seam for injecting a response
    object that has no text block at all."""
    def __init__(self, replies):
        self._replies = list(replies)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self._replies.pop(0)


def test_textless_reply_retries_instead_of_crashing():
    client = _RawClient([_NoTextResp(), _Resp(json.dumps({"spec": GOOD_SPEC}))])
    res = compile_strategy("buy rsi dips", client=client)
    assert res.spec == GOOD_SPEC
    assert res.attempts == 2


def test_usage_is_summed_across_retries_with_cache_split():
    bad = {**GOOD_SPEC, "timeframe": "4h"}
    client = FakeClient([json.dumps({"spec": bad}), json.dumps({"spec": GOOD_SPEC})])
    res = compile_strategy("buy rsi dips", client=client)
    assert res.attempts == 2
    assert res.usage == {"input_tokens": 200, "output_tokens": 100,
                         "cache_read_tokens": 20, "cache_write_tokens": 10}
    assert res.model == "claude-opus-4-8"


def test_usage_present_on_not_expressible():
    client = FakeClient([json.dumps({"not_expressible": "no renko"})])
    res = compile_strategy("renko", client=client)
    assert res.not_expressible and res.usage["input_tokens"] == 100


class _RaisingClient:
    """First create() succeeds (invalid spec -> retry), second raises: the
    seam for proving partial-loop usage survives an API failure."""
    def __init__(self, first_reply):
        self._sent = False
        self._first = first_reply
        self.messages = self

    def create(self, **kwargs):
        if self._sent:
            raise RuntimeError("api down")
        self._sent = True
        return _Resp(self._first)


def test_client_exception_returns_error_result_with_partial_usage():
    bad = json.dumps({"spec": {**GOOD_SPEC, "timeframe": "4h"}})
    res = compile_strategy("x", client=_RaisingClient(bad))
    assert "api down" in res.error
    assert res.spec is None
    assert res.attempts == 2                       # the raising attempt counts
    assert res.usage["input_tokens"] == 100        # attempt 1's usage survives


def test_usageless_response_is_tolerated():
    # _NoTextResp has no usage attribute at all; must not crash the summing.
    client = _RawClient([_NoTextResp(), _Resp(json.dumps({"spec": GOOD_SPEC}))])
    res = compile_strategy("buy rsi dips", client=client)
    assert res.spec == GOOD_SPEC and res.usage["input_tokens"] == 100
