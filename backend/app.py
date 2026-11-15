"""FastAPI server for the MUG medical Q&A app."""
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from mug.audit import OUTCOMES, PROXY_MODES, AuditStore
from mug.config import CHAT_MODELS, LOGPROB_MODELS, settings
from mug.gates import GATES, Question, answer_text
from mug.pipeline import Engine

engine: Engine | None = None
audit = AuditStore(Path(os.environ.get("MUG_AUDIT_DB", Path.home() / ".mug" / "audit.db")))


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine
    engine = Engine()
    asyncio.create_task(engine.warm())
    yield
    await engine.client.aclose()


app = FastAPI(title="MUG — Marginal-Utility Gated Medical QA", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    options: dict[str, str] | None = None
    gate: str = "flip"
    answer_model: str | None = None
    probe_model: str | None = None
    k_textbook: int = Field(default=settings.k_textbook, ge=0, le=12)
    k_pubmed: int = Field(default=settings.k_pubmed, ge=0, le=10)
    rewrite_query: bool = True
    user_passages: list[str] = Field(default_factory=list, max_length=5)
    speculative_filter: bool = False
    overlap_retrieval: bool = True
    parallel_probes: bool = True
    scout_model: str | None = None
    verifier_model: str | None = None
    tau_m: float | None = Field(default=None, ge=0, le=1)
    tau_g: float | None = Field(default=None, ge=0, le=1)
    tau_sem: float = Field(default=0.5, ge=0, le=1)


def _validate(req: AskRequest) -> dict:
    if req.gate not in GATES:
        raise HTTPException(400, f"gate must be one of {GATES}")
    for m in (req.answer_model, req.probe_model, req.scout_model, req.verifier_model):
        if m and m not in CHAT_MODELS:
            raise HTTPException(400, f"unsupported model {m}")
    if req.probe_model and req.probe_model not in LOGPROB_MODELS:
        raise HTTPException(400, "probe_model must expose logprobs: " + ", ".join(sorted(LOGPROB_MODELS)))
    return req.model_dump()


@app.get("/api/health")
async def health():
    tb = engine.textbooks if engine else None
    return {"ok": True, "textbooks_ready": bool(tb and tb.ready), "textbooks_error": tb.error if tb else None,
            "textbook_snippets": len(tb.ids) if tb else 0}


@app.get("/api/config")
async def config():
    return {"gates": GATES, "models": CHAT_MODELS, "logprob_models": sorted(LOGPROB_MODELS),
            "defaults": {"gate": "flip", "answer_model": settings.answer_model, "probe_model": settings.probe_model,
                         "k_textbook": settings.k_textbook, "k_pubmed": settings.k_pubmed,
                         "tau_m": settings.tau_m, "tau_g": settings.tau_g, "tau_sem": 0.5}}


@app.post("/api/ask")
async def ask(req: AskRequest):
    return await engine.ask(**_validate(req))


@app.post("/api/ask/stream")
async def ask_stream(req: AskRequest):
    params = _validate(req)
    queue: asyncio.Queue = asyncio.Queue()

    async def emit(ev: dict) -> None:
        await queue.put(ev)

    async def run() -> None:
        try:
            await engine.ask(**params, emit=emit)
        except Exception as e:  # noqa: BLE001
            await queue.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            await queue.put(None)

    async def gen():
        task = asyncio.create_task(run())
        try:
            while (ev := await queue.get()) is not None:
                yield f"data: {json.dumps(ev, default=str)}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ------------------------------------------------------------------ audit-proxy mode
class ChatMessage(BaseModel):
    role: str
    content: str | list | None = None


class ProxyDocument(BaseModel):
    content: str = Field(min_length=1, max_length=20000)
    title: str | None = None
    url: str | None = None


class ProxyRequest(BaseModel):
    """OpenAI chat-completions request plus the documents the caller's RAG retrieved.

    ``documents`` (and ``options``, ``verifier_model``, ``mug_mode``) are NOT part of the OpenAI schema: a stock
    OpenAI SDK has to pass them via ``extra_body``. Only the last user message is read (no system prompt, history or
    the client's own prompt template), and streaming is not supported."""
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    documents: list[ProxyDocument | str] = Field(default_factory=list, max_length=40)
    # MCQ mode (optional): options give letter answers and exact movement.
    options: dict[str, str] | None = None
    verifier_model: str | None = os.environ.get("MUG_VERIFIER_MODEL") or None
    # "gate" (default): return MUG's gated answer. "shadow": return the plain with-documents answer unchanged and
    # only label it. Also settable per request with the X-MUG-Mode header, or globally with MUG_PROXY_MODE.
    mug_mode: str | None = None
    stream: bool | None = None
    user: str | None = None


def _text(c) -> str:
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


@app.post("/v1/chat/completions")
async def proxy(req: ProxyRequest, x_mug_mode: str | None = Header(default=None)):
    """Audit proxy for a RAG system's generation call (OpenAI-compatible request shape plus ``documents``).

    The last user message is the question. Gate mode answers through the flip gate (plus the verifier if
    configured). Shadow mode runs the same gate to label the request but returns the ungated with-documents answer,
    so the caller's behaviour is unchanged except for latency. Every decision is logged for audit."""
    mode = (req.mug_mode or x_mug_mode or os.environ.get("MUG_PROXY_MODE") or "gate").strip().lower()
    if mode not in PROXY_MODES:
        raise HTTPException(400, f"mug_mode must be one of {PROXY_MODES}")
    if req.stream:
        raise HTTPException(400, "streaming is not supported by the audit proxy")
    question = next((_text(m.content) for m in reversed(req.messages) if m.role == "user"), "").strip()
    if len(question) < 3:
        raise HTTPException(400, "no user question in messages")
    for m in (req.model, req.verifier_model):
        if m and m not in CHAT_MODELS:
            raise HTTPException(400, f"unsupported model {m}")
    docs = [d if isinstance(d, ProxyDocument) else ProxyDocument(content=d) for d in req.documents]
    pool = [{"id": f"doc:{i}", "source": "user", "title": d.title or f"Document {i + 1}", "content": d.content,
             "url": d.url} for i, d in enumerate(docs)]
    model = req.model or settings.answer_model
    out = await engine.ask(question=question, options=req.options, gate="flip", answer_model=model, pool=pool,
                           verifier_model=req.verifier_model)
    res, summ = out["result"], out["summary"]
    if mode == "shadow":
        # What plain RAG would have returned: the answer with all documents (closed-book if there were none).
        plain = res.get("context_answer") if docs else res.get("closed_book_answer")
        q = Question(question, {k.strip().upper(): v for k, v in (req.options or {}).items() if v} or None)
        content = answer_text(q, plain) if plain is not None else res["answer_text"]
    else:
        content = res["answer_text"]
    row = audit.record(client=req.user or "anonymous", model=model, verifier=req.verifier_model,
                       question=question, documents=[d.model_dump() for d in docs], out=out, proxy_mode=mode,
                       returned=content)
    return {
        "id": row["id"], "object": "chat.completion", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": summ["prompt_tokens"], "completion_tokens": summ["completion_tokens"],
                  "total_tokens": summ["prompt_tokens"] + summ["completion_tokens"]},
        "mug_audit": {k: row[k] for k in ("id", "outcome", "proxy_mode", "gate_answer", "rag_answer",
                                          "closed_book_answer", "kept", "quote", "movement")} | {"note": res["note"]},
    }


@app.get("/api/audit")
async def audit_list(limit: int = 100, outcome: str | None = None):
    if outcome and outcome not in OUTCOMES:
        raise HTTPException(400, f"outcome must be one of {OUTCOMES}")
    return {"stats": audit.stats(), "items": audit.recent(min(limit, 500), outcome)}


@app.get("/api/audit/{id_}")
async def audit_get(id_: str):
    r = audit.get(id_)
    if r is None:
        raise HTTPException(404, "not found")
    return r


_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    async def spa(path: str):
        f = _DIST / path
        return FileResponse(f if path and f.is_file() else _DIST / "index.html")
