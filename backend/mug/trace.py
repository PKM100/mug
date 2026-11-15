"""Per-request trace: records every LLM/retrieval span for cost & latency accounting
and optionally streams events to a client (SSE)."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable

Emitter = Callable[[dict], Awaitable[None]]


@dataclass
class Span:
    name: str
    kind: str                 # llm | retrieval | decision
    start_ms: float
    end_ms: float = 0.0
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    speculative: bool = False  # issued before we knew it was needed
    used: bool = True          # False if a speculative result was discarded
    detail: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


class Trace:
    def __init__(self, emit: Emitter | None = None) -> None:
        self.t0 = time.perf_counter()
        self.spans: list[Span] = []
        self._emit = emit

    def now_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0

    def start(self, name: str, kind: str, **kw: Any) -> Span:
        s = Span(name=name, kind=kind, start_ms=self.now_ms(), **kw)
        self.spans.append(s)
        return s

    async def end(self, span: Span, **detail: Any) -> None:
        span.end_ms = self.now_ms()
        span.detail.update(detail)
        await self.emit({"type": "span", "span": self.span_dict(span)})

    async def emit(self, event: dict) -> None:
        if self._emit is not None:
            await self._emit(event)

    @staticmethod
    def span_dict(s: Span) -> dict:
        d = asdict(s)
        d["duration_ms"] = round(s.duration_ms, 1)
        return d

    def summary(self) -> dict:
        llm = [s for s in self.spans if s.kind == "llm"]
        return {
            "wall_ms": round(self.now_ms(), 1),
            "llm_calls": len(llm),
            "llm_calls_used": sum(1 for s in llm if s.used),
            "prompt_tokens": sum(s.prompt_tokens for s in llm),
            "completion_tokens": sum(s.completion_tokens for s in llm),
            "wasted_prompt_tokens": sum(s.prompt_tokens for s in llm if not s.used),
            # Sum of span durations = latency a fully sequential schedule would pay.
            "sequential_ms": round(sum(s.duration_ms for s in self.spans if s.kind != "decision"), 1),
        }
