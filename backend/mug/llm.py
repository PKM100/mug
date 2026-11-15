"""Async chat client for the GitHub Copilot endpoint, the OpenAI API and the Anthropic Messages API."""
from __future__ import annotations

import asyncio
import os
import random
import subprocess
from dataclasses import dataclass

import httpx

from .config import provider_for, settings, supports_logprobs
from .trace import Trace


@dataclass
class LLMResponse:
    content: str
    top_logprobs: list[dict] | None  # first generated token's top alternatives [{token, logprob}]
    prompt_tokens: int
    completion_tokens: int


def _resolve_token() -> str:
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        return tok
    try:
        return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True,
                              check=True, timeout=15).stdout.strip()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("No GitHub token: set GITHUB_TOKEN or run `gh auth login`.") from e


class LLMError(RuntimeError):
    pass


def anthropic_model_id(model: str) -> str:
    """Accept Copilot-style names (claude-haiku-4.5) for the Anthropic API (claude-haiku-4-5)."""
    return model.replace(".", "-")


class LLMClient:
    """One client for all providers; each call is routed by ``config.provider_for(model)``."""

    def __init__(self, token: str | None = None, base_url: str | None = None,
                 max_concurrency: int | None = None, provider: str | None = None) -> None:
        self._copilot_token = token
        self._base = (base_url or settings.copilot_base_url).rstrip("/")
        self._provider = provider
        self._sem = asyncio.Semaphore(max_concurrency or settings.max_concurrency)
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0))

    async def aclose(self) -> None:
        await self._http.aclose()

    def _request(self, model: str, prompt: str, max_tokens: int, temperature: float,
                 want_lp: bool) -> tuple[str, str, dict, dict]:
        """Return (provider, url, headers, body) for one call."""
        prov = provider_for(model, self._provider)
        msgs = [{"role": "user", "content": prompt}]
        if prov == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise LLMError("ANTHROPIC_API_KEY is not set")
            return prov, f"{settings.anthropic_base_url.rstrip('/')}/v1/messages", {
                "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json",
            }, {"model": anthropic_model_id(model), "messages": msgs, "max_tokens": max_tokens,
                "temperature": temperature}
        body: dict = {"model": model, "messages": msgs, "temperature": temperature}
        if want_lp:
            body.update(logprobs=True, top_logprobs=10)
        if prov == "openai":
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise LLMError("OPENAI_API_KEY is not set")
            body["max_completion_tokens"] = max_tokens
            return prov, f"{settings.openai_base_url.rstrip('/')}/chat/completions", {
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"}, body
        if self._copilot_token is None:
            self._copilot_token = _resolve_token()
        body["max_tokens"] = max_tokens
        return prov, f"{self._base}/chat/completions", {
            "Authorization": f"Bearer {self._copilot_token}", "Copilot-Integration-Id": "vscode-chat",
            "Editor-Version": "vscode/1.90.0", "Content-Type": "application/json"}, body

    @staticmethod
    def _parse(prov: str, d: dict) -> tuple[str, list[dict] | None, dict]:
        """Return (content, first-token top logprobs, {prompt_tokens, completion_tokens})."""
        if prov == "anthropic":
            if d.get("type") == "error" or "content" not in d:
                raise LLMError(f"empty response: {str(d)[:200]}")
            text = "".join(b.get("text", "") for b in d["content"] if b.get("type") == "text")
            u = d.get("usage") or {}
            return text, None, {"prompt_tokens": u.get("input_tokens"), "completion_tokens": u.get("output_tokens")}
        if not d.get("choices"):
            raise LLMError(f"empty response: {str(d)[:200]}")
        choice = d["choices"][0]
        content = (choice.get("message") or {}).get("content") or ""
        top = None
        lp = choice.get("logprobs") or {}
        if lp.get("content"):
            top = lp["content"][0].get("top_logprobs") or None
        return content, top, d.get("usage") or {}

    async def chat(self, prompt: str, *, model: str, trace: Trace | None = None, name: str = "llm",
                   max_tokens: int = 16, temperature: float = 0.0, logprobs: bool = False,
                   speculative: bool = False, retries: int = 5) -> LLMResponse:
        want_lp = logprobs and supports_logprobs(model)
        span = trace.start(name, "llm", model=model, speculative=speculative,
                           prompt_tokens=len(prompt) // 4) if trace else None
        last: Exception | None = None
        ok = False
        try:
            for attempt in range(retries + 1):
                try:
                    prov, url, headers, body = self._request(model, prompt, max_tokens, temperature, want_lp)
                    async with self._sem:
                        r = await self._http.post(url, json=body, headers=headers)
                    if r.status_code in (429, 500, 502, 503, 504, 529):
                        raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                    if r.status_code >= 400:
                        raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")  # non-retryable
                    content, top, usage = self._parse(prov, r.json())
                    resp = LLMResponse(content=content, top_logprobs=top,
                                       prompt_tokens=int(usage.get("prompt_tokens") or len(prompt) // 4),
                                       completion_tokens=int(usage.get("completion_tokens") or 0))
                    if not content.strip() and top is None:
                        raise LLMError("empty response")
                    if span:
                        span.prompt_tokens, span.completion_tokens = resp.prompt_tokens, resp.completion_tokens
                    ok = True
                    return resp
                except (LLMError, httpx.HTTPError, KeyError, ValueError) as e:
                    last = e
                    retryable = not (isinstance(e, LLMError) and (str(e).startswith("HTTP 4") and "429" not in str(e)
                                                                  or "is not set" in str(e)))
                    if attempt == retries or not retryable:
                        break
                    await asyncio.sleep(min(20.0, 1.5 ** attempt + random.random()))
            raise LLMError(f"{model} failed: {last}")
        finally:
            if span and trace:
                await trace.end(span, ok=ok, error=None if ok else str(last)[:200])


CopilotClient = LLMClient  # backwards-compatible name
