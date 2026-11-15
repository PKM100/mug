"""Marginal-Utility Gating (MUG) gates, served.

Ports the gates from the MUG paper (answer-flip + marginal-utility filter, logit-shift,
grounded) and extends them to open-ended questions via *semantic* answer-movement:

    MCQ        m = 1[a_cb != a_ctx]  (flip)   or   TV(p_cb, p_ctx)  (logit-shift)
    open-ended m = 1 - P(equivalent(a_cb, a_ctx))   read from a yes/no probe's logprobs

Scheduling (the serving contribution): the closed-book probe needs no evidence, so it is
launched before retrieval finishes and overlaps it; the evaluator filter depends only on
(question, pool), so it can optionally be launched *speculatively* alongside the
context probe and cancelled when the answer does not move.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass, field

from . import prompts as P
from .llm import LLMClient, LLMError
from .trace import Trace

GATES = ["flip", "grounded", "logit_shift", "naive", "closed_book"]


@dataclass
class Question:
    text: str
    options: dict[str, str] | None = None

    @property
    def is_mcq(self) -> bool:
        return bool(self.options)


@dataclass
class Runtime:
    client: LLMClient
    trace: Trace
    answer_model: str
    probe_model: str
    tau_m: float = 0.15
    tau_g: float = 0.5
    tau_sem: float = 0.5
    speculative_filter: bool = False
    overlap_retrieval: bool = True
    parallel_probes: bool = True  # False = fully serial: closed-book probe completes before full-pool probe
    # Cascade: if set, this (cheaper) model runs the probes and the filter, so the answer model never reads
    # the full pool; it answers only closed-book or on the admitted context.
    scout_model: str | None = None
    # Reliability check: if set, an independent (ideally different-family) model must quote a passage that
    # directly establishes the moved answer before the flip gate admits context.
    verifier_model: str | None = None


@dataclass
class GateResult:
    gate: str
    mode: str
    answer: str | None
    answer_text: str
    admitted: bool
    kept_context: list[dict] = field(default_factory=list)
    closed_book_answer: str | None = None
    context_answer: str | None = None
    movement: float | None = None
    groundedness: float | None = None
    retained_tokens: int = 0
    pool_tokens: int = 0
    note: str = ""
    # Set when an evidence verifier ran: {"supported": bool, "doc": n|None, "quote": str, "model": str}
    verification: dict | None = None


def approx_tokens(text: str) -> int:
    return max(0, len(text) // 4)


def ctx_tokens(ctx: list[dict]) -> int:
    return sum(approx_tokens(c["content"]) for c in ctx)


# ------------------------------------------------------------------ primitives
def option_dist(top: list[dict] | None, valid: list[str]) -> dict[str, float] | None:
    if not top:
        return None
    raw: dict[str, float] = {}
    for t in top:
        tok = t["token"].strip().upper()
        if tok in valid and tok not in raw:
            raw[tok] = t["logprob"]
    if not raw:
        return None
    mx = max(raw.values())
    ex = {k: math.exp(v - mx) for k, v in raw.items()}
    z = sum(ex.values())
    return {k: ex.get(k, 0.0) / z for k in valid}


def total_variation(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def yes_prob(top: list[dict] | None, content: str) -> float:
    yes = no = None
    for t in top or []:
        tok = t["token"].strip().lower()
        if tok in ("yes", "y") and yes is None:
            yes = t["logprob"]
        elif tok in ("no", "n") and no is None:
            no = t["logprob"]
    if yes is not None or no is not None:
        if yes is None:
            return 0.0
        if no is None:
            return 1.0
        my = max(yes, no)
        ey, en = math.exp(yes - my), math.exp(no - my)
        return ey / (ey + en)
    low = content.strip().lower()
    if low.startswith("y"):
        return 1.0
    if low.startswith("n"):
        return 0.0
    return 0.5


def parse_letter(text: str, valid: list[str]) -> str | None:
    up = text.strip().upper()
    m = re.search(r"\b([A-J])\b", up)
    if m and m.group(1) in valid:
        return m.group(1)
    return next((c for c in up if c in valid), None)


def answer_text(q: Question, ans: str | None) -> str:
    if ans is None:
        return ""
    if q.is_mcq:
        return f"{ans}. {q.options.get(ans, '')}".strip()
    return ans


async def answer(rt: Runtime, q: Question, ctx: list[dict], name: str,
                 logprobs: bool = False, model: str | None = None) -> tuple[str | None, dict | None]:
    model = model or rt.answer_model
    try:
        if q.is_mcq:
            valid = list(q.options)
            r = await rt.client.chat(P.mcq_prompt(q.text, q.options, ctx), model=model,
                                     trace=rt.trace, name=name, max_tokens=4, logprobs=logprobs)
            dist = option_dist(r.top_logprobs, valid) if logprobs else None
            if dist and any(v > 0 for v in dist.values()):
                return max(dist, key=dist.get), dist
            return parse_letter(r.content, valid), None
        r = await rt.client.chat(P.open_prompt(q.text, ctx), model=model, trace=rt.trace,
                                 name=name, max_tokens=350)
        return r.content.strip() or None, None
    except LLMError:
        return None, None


async def prob_yes(rt: Runtime, prompt: str, name: str) -> float | None:
    try:
        r = await rt.client.chat(prompt, model=rt.probe_model, trace=rt.trace, name=name,
                                 max_tokens=2, logprobs=True)
        return yes_prob(r.top_logprobs, r.content)
    except LLMError:
        return None


async def movement(rt: Runtime, q: Question, cb: tuple, ctx: tuple) -> float:
    (a_cb, p_cb), (a_ctx, p_ctx) = cb, ctx
    if a_cb is None or a_ctx is None:
        return 0.0
    if q.is_mcq:
        if p_cb is not None and p_ctx is not None:
            return total_variation(p_cb, p_ctx)
        return 0.0 if a_cb == a_ctx else 1.0
    p_same = await prob_yes(rt, P.equivalence_prompt(q.text, a_cb, a_ctx), "movement(equivalence)")
    return 0.0 if p_same is None else 1.0 - p_same


def parse_evaluator(raw: str, pool: list[dict]) -> list[dict]:
    n = len(pool)
    keep: list[int] = []
    extracted: dict[int, str] = {}
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        obj = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        obj = None
    if obj is not None:
        for x in obj.get("keep", []) or []:
            try:
                if 0 <= int(x) < n:
                    keep.append(int(x))
            except (TypeError, ValueError):
                continue
        for k, v in (obj.get("decisive_sentence") or {}).items():
            try:
                if 0 <= int(k) < n and isinstance(v, str) and v.strip():
                    extracted[int(k)] = v.strip()
            except (TypeError, ValueError):
                continue
    else:
        keep = sorted({int(x) for x in re.findall(r"\b(\d+)\b", raw) if int(x) < n})
    out = []
    for i in dict.fromkeys(keep):
        c = dict(pool[i])
        if i in extracted:
            c["full_content"] = c["content"]
            c["content"] = extracted[i]
        out.append(c)
    return out


async def evaluate_filter(rt: Runtime, q: Question, pool: list[dict], speculative: bool = False) -> list[dict]:
    try:
        r = await rt.client.chat(P.evaluator_prompt(q.text, pool), model=rt.scout_model or rt.answer_model,
                                 trace=rt.trace,
                                 name="filter(marginal-utility)", max_tokens=500, speculative=speculative)
        return parse_evaluator(r.content, pool)
    except LLMError:
        return []


def parse_verdict(raw: str, context: list[dict]) -> dict:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        obj = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        obj = {}
    quote = str(obj.get("quote") or "").strip()
    # Support counts only with a quote that actually occurs in an admitted passage.
    norm = lambda t: re.sub(r"\W+", " ", t).lower().strip()
    ok = (obj.get("supported") is True and bool(norm(quote))
          and any(norm(quote)[:120] in norm(c.get("full_content", c["content"])) for c in context))
    return {"supported": ok, "doc": obj.get("doc"), "quote": quote if ok else ""}


async def verify_support(rt: Runtime, q: Question, ans: str | None, context: list[dict]) -> dict | None:
    """Independent reliability check. Returns None if the verifier call failed (caller treats as unverified)."""
    try:
        r = await rt.client.chat(P.verify_prompt(q.text, answer_text(q, ans), context), model=rt.verifier_model,
                                 trace=rt.trace, name="verify(support)", max_tokens=400)
        return parse_verdict(r.content, context)
    except LLMError:
        return None


async def _decision(rt: Runtime, label: str, **detail) -> None:
    s = rt.trace.start(label, "decision")
    await rt.trace.end(s, **detail)


# ------------------------------------------------------------------ gates
async def run_gate(gate: str, rt: Runtime, q: Question, pool_task: "asyncio.Future[list[dict]]") -> GateResult:
    if gate not in GATES:
        raise ValueError(f"unknown gate {gate!r}; choose from {GATES}")
    lp = q.is_mcq and gate in ("logit_shift", "grounded")

    if gate == "closed_book":
        a, _ = await answer(rt, q, [], "answer(closed-book)")
        return GateResult(gate, "closed_book", a, answer_text(q, a), False, closed_book_answer=a)

    if gate == "naive":
        pool = await pool_task
        pool_tok = ctx_tokens(pool)
        a, _ = await answer(rt, q, pool, "answer(full pool)")
        return GateResult(gate, "naive", a, answer_text(q, a), bool(pool), pool, context_answer=a,
                          retained_tokens=pool_tok, pool_tokens=pool_tok)

    pm = rt.scout_model
    # In cascade mode the answer model's closed-book answer is cheap (no evidence), so issue it up front.
    big_cb = asyncio.create_task(answer(rt, q, [], "answer(closed-book)")) if pm else None
    if rt.overlap_retrieval:
        cb_task = asyncio.create_task(answer(rt, q, [], "probe(closed-book)", logprobs=lp, model=pm))
        pool = await pool_task
    else:
        pool = await pool_task
        cb_task = asyncio.create_task(answer(rt, q, [], "probe(closed-book)", logprobs=lp, model=pm))
        if not rt.parallel_probes:
            await asyncio.wait({cb_task})
    pool_tok = ctx_tokens(pool)

    if not pool:
        a, _ = await (big_cb or cb_task)
        return GateResult(gate, "no_evidence", a, answer_text(q, a), False, closed_book_answer=a,
                          note="Retrieval returned no candidates; answered closed-book.")

    ctx_task = asyncio.create_task(answer(rt, q, pool, "probe(full pool)", logprobs=lp, model=pm))
    spec_task = None
    if gate == "flip" and rt.speculative_filter:
        spec_task = asyncio.create_task(evaluate_filter(rt, q, pool, speculative=True))
    cb, ctx = await asyncio.gather(cb_task, ctx_task)
    a_cb, a_ctx = cb[0], ctx[0]
    m = await movement(rt, q, cb, ctx)
    base = dict(closed_book_answer=a_cb, context_answer=a_ctx, movement=round(m, 4), pool_tokens=pool_tok)

    async def withheld() -> str | None:
        if big_cb is None:
            return a_cb if a_cb is not None else a_ctx
        a_big, _ = await big_cb
        return a_big if a_big is not None else a_cb

    async def on_pool() -> str | None:
        if pm is None:
            return a_ctx if a_ctx is not None else a_cb
        a_big, _ = await answer(rt, q, pool, "answer(admitted pool)")
        return a_big if a_big is not None else a_ctx

    if gate == "flip":
        moved = (a_ctx is not None and a_ctx != a_cb) if q.is_mcq else m >= rt.tau_sem
        await _decision(rt, "decide(movement)", moved=moved, movement=round(m, 4))
        if not moved:
            if spec_task:
                spec_task.cancel()
                now = rt.trace.now_ms()
                for s in rt.trace.spans:
                    if s.speculative:
                        s.used = False
                        if s.end_ms < s.start_ms:  # cancelled in flight
                            s.end_ms = now
            a = await withheld()
            return GateResult(gate, "closed_book", a, answer_text(q, a), False, **base,
                              note="Evidence did not move the answer; context withheld.")
        kept = await spec_task if spec_task else await evaluate_filter(rt, q, pool)
        mode = "flip_filtered"
        if not kept:
            kept, mode = pool, "flip_full"
        a, _ = await answer(rt, q, kept, "answer(filtered context)")
        if a is None:
            a = a_ctx
        if rt.verifier_model and a is not None:
            v = await verify_support(rt, q, a, kept)
            supported = bool(v and v["supported"])
            await _decision(rt, "decide(verified support)", supported=supported, quote=(v or {}).get("quote", ""))
            base["verification"] = dict(v or {"supported": False, "doc": None, "quote": "", "error": True},
                                        model=rt.verifier_model)
            if not supported:
                a = await withheld()
                return GateResult(gate, "unverified_keep", a, answer_text(q, a), False, **base,
                                  note="Evidence moved the answer, but an independent verifier found no passage "
                                       "that establishes it; kept the parametric answer.")
        quote = (base.get("verification") or {}).get("quote", "")
        return GateResult(gate, mode, a, answer_text(q, a), True, kept, **base,
                          retained_tokens=ctx_tokens(kept),
                          note="Evidence moved the answer; re-answered on the passages that carry it."
                               + (f' Verified: "{quote[:200]}"' if quote else ""))

    tau = rt.tau_m if q.is_mcq else rt.tau_sem
    if m < tau:
        await _decision(rt, "decide(movement)", moved=False, movement=round(m, 4), tau=tau)
        a = await withheld()
        return GateResult(gate, "closed_book", a, answer_text(q, a), False, **base,
                          note=f"Movement {m:.2f} < τ={tau}; context withheld.")

    if gate == "logit_shift":
        await _decision(rt, "decide(movement)", moved=True, movement=round(m, 4), tau=tau)
        a = await on_pool()
        return GateResult(gate, "shift", a, answer_text(q, a), True, pool, **base, retained_tokens=pool_tok)

    # grounded: influence x reliability
    g = await prob_yes(rt, P.entail_prompt(q.text, answer_text(q, a_ctx or a_cb), pool), "probe(groundedness)")
    g_val = 0.5 if g is None else g
    base["groundedness"] = round(g_val, 4)
    await _decision(rt, "decide(influence×reliability)", movement=round(m, 4), groundedness=round(g_val, 4))
    if g_val >= rt.tau_g:
        a = await on_pool()
        return GateResult(gate, "grounded_shift", a, answer_text(q, a), True, pool, **base,
                          retained_tokens=pool_tok, note="Evidence moved the answer and supports it.")
    a = await withheld()
    return GateResult(gate, "ungrounded_keep", a, answer_text(q, a), False, **base,
                      note="Evidence moved the answer but does not support it; kept the parametric answer.")
