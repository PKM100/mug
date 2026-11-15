import asyncio
import json

import httpx
import pytest

from mug import config
from mug.config import provider_for, supports_logprobs
from mug.llm import LLMClient, LLMError


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setattr(config.settings, "provider", "auto")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def set_keys(openai=False, anthropic=False):
        if openai:
            monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        if anthropic:
            monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")
    return set_keys


def test_auto_routing(keys):
    assert provider_for("gpt-4o-mini") == "copilot"
    keys(openai=True, anthropic=True)
    assert provider_for("gpt-4o-mini") == "openai"
    assert provider_for("claude-haiku-4-5") == "anthropic"
    assert provider_for("claude-haiku-4-5", "copilot") == "copilot"


def test_anthropic_has_no_logprobs(keys):
    keys(anthropic=True)
    assert not supports_logprobs("claude-haiku-4-5")
    assert supports_logprobs("gpt-4o-mini")


def _client(handler) -> LLMClient:
    c = LLMClient(token="gh-test")
    c._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c


def test_anthropic_request_and_parse(keys):
    keys(anthropic=True)
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"], seen["key"], seen["body"] = str(req.url), req.headers["x-api-key"], json.loads(req.content)
        return httpx.Response(200, json={"type": "message", "content": [{"type": "text", "text": "B"}],
                                         "usage": {"input_tokens": 12, "output_tokens": 1}})

    r = asyncio.run(_client(handler).chat("q", model="claude-haiku-4.5", logprobs=True))
    assert seen["url"].endswith("/v1/messages") and seen["key"] == "ak-test"
    assert seen["body"]["model"] == "claude-haiku-4-5" and "logprobs" not in seen["body"]
    assert (r.content, r.top_logprobs, r.prompt_tokens, r.completion_tokens) == ("B", None, 12, 1)


def test_openai_request_and_logprobs(keys):
    keys(openai=True)
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"], seen["body"] = str(req.url), req.headers["authorization"], json.loads(req.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "A"},
                         "logprobs": {"content": [{"top_logprobs": [{"token": "A", "logprob": -0.1}]}]}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 1}})

    r = asyncio.run(_client(handler).chat("q", model="gpt-4o-mini", logprobs=True))
    assert seen["url"] == "https://api.openai.com/v1/chat/completions" and seen["auth"] == "Bearer sk-test"
    assert seen["body"]["max_completion_tokens"] == 16 and seen["body"]["top_logprobs"] == 10
    assert "max_tokens" not in seen["body"]
    assert r.top_logprobs == [{"token": "A", "logprob": -0.1}]


def test_missing_key_fails_fast(keys, monkeypatch):
    monkeypatch.setattr(config.settings, "provider", "openai")
    calls = []
    c = _client(lambda req: calls.append(req) or httpx.Response(500))
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        asyncio.run(c.chat("q", model="gpt-4o-mini", retries=3))
    assert calls == []
