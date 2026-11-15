"""End-to-end serving pipeline: hybrid retrieval -> MUG gate -> answer + trace."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import asdict
from typing import Any

from . import prompts as P
from .config import Settings, settings as default_settings, supports_logprobs
from .gates import GATES, Question, Runtime, ctx_tokens, run_gate
from .llm import LLMClient, LLMError
from .retrieval.pubmed import PubMedClient, clean_query, keyword_query
from .retrieval.textbooks import TextbookIndex
from .trace import Emitter, Trace


class Engine:
    def __init__(self, cfg: Settings | None = None, client: LLMClient | None = None) -> None:
        self.cfg = cfg or default_settings
        self.client = client or LLMClient()
        self.textbooks = TextbookIndex(self.cfg.index_dir)
        self.pubmed = PubMedClient(api_key=self.cfg.ncbi_api_key)
        self.use_cache = True
        self._pm_cache: OrderedDict[tuple, tuple[str, list[dict]]] = OrderedDict()

    async def warm(self) -> None:
        if self.cfg.enable_textbooks:
            await asyncio.to_thread(self.textbooks.load)

    # ------------------------------------------------------------ retrieval
    async def _textbooks(self, q: Question, k: int, trace: Trace) -> list[dict]:
        if k <= 0 or not self.textbooks.ready:
            return []
        span = trace.start("retrieve(textbooks)", "retrieval")
        query = q.text + (" " + " ".join(q.options.values()) if q.options else "")
        hits = await asyncio.to_thread(self.textbooks.search, query, k)
        await trace.end(span, n=len(hits))
        return hits

    async def _pubmed(self, q: Question, k: int, trace: Trace, rewrite: bool, model: str) -> list[dict]:
        if k <= 0:
            return []
        fallback = keyword_query(q.text)
        term = fallback
        if rewrite:
            # Optional stage with a deadline: a stalled rewrite must not stall the request.
            try:
                r = await asyncio.wait_for(
                    self.client.chat(P.pubmed_query_prompt(q.text, q.options), model=model, trace=trace,
                                     name="rewrite(pubmed query)", max_tokens=40, retries=1),
                    timeout=self.cfg.rewrite_deadline_s)
                term = clean_query(r.content) or fallback
            except (LLMError, asyncio.TimeoutError):
                term = fallback
        key = (term, k)
        span = trace.start("retrieve(pubmed)", "retrieval")
        try:
            if self.use_cache and key in self._pm_cache:
                used, hits = self._pm_cache[key]
            else:
                used, hits = await self.pubmed.search(term, k, fallback=fallback)
                self._pm_cache[key] = (used, hits)
                if len(self._pm_cache) > 512:
                    self._pm_cache.popitem(last=False)
            kept = [dict(h) for h in hits if not (h.get("meta") or {}).get("quality", {}).get("retracted")]
            dropped = [h["id"] for h in hits if h not in kept and (h.get("meta") or {}).get("quality", {}).get("retracted")]
            await trace.end(span, n=len(kept), query=used, **({"dropped_retracted": dropped} if dropped else {}))
            return kept
        except Exception as e:  # noqa: BLE001
            await trace.end(span, n=0, query=term, error=f"{type(e).__name__}: {e}"[:200])
            return []

    async def retrieve(self, q: Question, trace: Trace, *, k_textbook: int, k_pubmed: int,
                       rewrite: bool, user_passages: list[str], model: str) -> list[dict]:
        tb, pm = await asyncio.gather(self._textbooks(q, k_textbook, trace),
                                      self._pubmed(q, k_pubmed, trace, rewrite, model))
        user = [{"id": f"user:{i}", "source": "user", "title": "Provided passage", "content": t.strip(),
                 "url": None} for i, t in enumerate(user_passages) if t.strip()]
        pool = user + pm + tb
        for i, c in enumerate(pool):
            c["pool_index"] = i
        await trace.emit({"type": "pool", "pool": pool, "pool_tokens": ctx_tokens(pool)})
        return pool

    # ------------------------------------------------------------ ask
    async def ask(self, *, question: str, options: dict[str, str] | None = None, gate: str = "flip",
                  answer_model: str | None = None, probe_model: str | None = None,
                  k_textbook: int | None = None, k_pubmed: int | None = None, rewrite_query: bool = True,
                  user_passages: list[str] | None = None, pool: list[dict] | None = None,
                  speculative_filter: bool = False, overlap_retrieval: bool = True, parallel_probes: bool = True,
                  scout_model: str | None = None, verifier_model: str | None = None, tau_m: float | None = None, tau_g: float | None = None, tau_sem: float = 0.5,
                  emit: Emitter | None = None) -> dict[str, Any]:
        if gate not in GATES:
            raise ValueError(f"unknown gate {gate!r}")
        options = {k.strip().upper(): v for k, v in (options or {}).items() if v and str(v).strip()} or None
        q = Question(question.strip(), options)
        trace = Trace(emit)
        rt = Runtime(self.client, trace, answer_model or self.cfg.answer_model,
                     probe_model or self.cfg.probe_model,
                     tau_m=self.cfg.tau_m if tau_m is None else tau_m,
                     tau_g=self.cfg.tau_g if tau_g is None else tau_g, tau_sem=tau_sem,
                     speculative_filter=speculative_filter, overlap_retrieval=overlap_retrieval,
                     parallel_probes=parallel_probes, scout_model=scout_model or None,
                     verifier_model=verifier_model or None)
        await trace.emit({"type": "start", "question_type": "mcq" if q.is_mcq else "open", "gate": gate,
                          "answer_model": rt.answer_model, "probe_model": rt.probe_model, "scout_model": rt.scout_model, "verifier_model": rt.verifier_model,
                          "logprobs": supports_logprobs(rt.answer_model)})
        if pool is not None:
            fixed = [dict(c, pool_index=i) for i, c in enumerate(pool)]

            async def _given() -> list[dict]:
                return fixed
            pool_task = asyncio.ensure_future(_given())
        elif gate == "closed_book":
            pool_task = asyncio.ensure_future(asyncio.sleep(0, result=[]))
        else:
            pool_task = asyncio.ensure_future(self.retrieve(
                q, trace, k_textbook=self.cfg.k_textbook if k_textbook is None else k_textbook,
                k_pubmed=self.cfg.k_pubmed if k_pubmed is None else k_pubmed, rewrite=rewrite_query,
                user_passages=user_passages or [], model=self.cfg.probe_model))
        res = await run_gate(gate, rt, q, pool_task)
        final_pool = pool_task.result() if pool_task.done() else []
        out = {
            "type": "result",
            "question": q.text,
            "options": q.options,
            "question_type": "mcq" if q.is_mcq else "open",
            "result": asdict(res),
            "pool": final_pool,
            "summary": trace.summary(),
            "spans": [Trace.span_dict(s) for s in trace.spans],
        }
        await trace.emit(out)
        return out
