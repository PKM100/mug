import asyncio

import pytest

from mug.gates import (Question, Runtime, option_dist, parse_evaluator, run_gate, total_variation,
                       yes_prob)
from mug.llm import LLMResponse
from mug.trace import Trace

OPTS = {"A": "Cystitis", "B": "Pyelonephritis", "C": "Stones"}
POOL = [{"id": "d0", "content": "Distractor about stones."},
        {"id": "d1", "content": "Flank pain with fever indicates pyelonephritis. More text."}]


class FakeClient:
    """Routes prompts to canned answers; records call order."""

    def __init__(self, cb="A", ctx="B", entail_yes=-0.1, entail_no=-3.0, same_yes=-5.0, same_no=-0.01,
                 keep='{"keep":[1],"decisive_sentence":{"1":"Flank pain with fever indicates pyelonephritis."}}',
                 delay=0.0):
        self.cb, self.ctx, self.keep, self.delay = cb, ctx, keep, delay
        self.entail = [{"token": "yes", "logprob": entail_yes}, {"token": "no", "logprob": entail_no}]
        self.same = [{"token": "yes", "logprob": same_yes}, {"token": "no", "logprob": same_no}]
        self.calls: list[str] = []
        self.verdict = '{"supported": true, "doc": 1, "quote": "Flank pain with fever indicates pyelonephritis."}'

    async def chat(self, prompt, *, model, trace=None, name="llm", speculative=False, **kw):
        span = trace.start(name, "llm", model=model, speculative=speculative) if trace else None
        try:
            self.calls.append(name)
            await asyncio.sleep(self.delay)
            if name.startswith("verify"):
                return LLMResponse(self.verdict, None, 10, 20)
            if name.startswith("filter"):
                return LLMResponse(self.keep, None, 10, 5)
            if "groundedness" in name:
                return LLMResponse("yes", self.entail, 10, 1)
            if "equivalence" in name:
                return LLMResponse("no", self.same, 10, 1)
            has_ctx = "Relevant references" in prompt
            letter = self.ctx if has_ctx else self.cb
            if "Options:" not in prompt:
                return LLMResponse(f"free text {letter}", None, 10, 20)
            top = [{"token": letter, "logprob": -0.05}, {"token": "C", "logprob": -3.5}]
            return LLMResponse(letter, top, 10, 1)
        finally:
            if span:
                await trace.end(span)


def rt(client, **kw):
    answer_model = "gpt-4.1" if kw.get("scout_model") else "gpt-4o-mini"
    return Runtime(client, Trace(), answer_model, "gpt-4o-mini", **kw)


async def pool_fut(pool=POOL, delay=0.0):
    await asyncio.sleep(delay)
    return [dict(c) for c in pool]


def run(gate, client, q=Question("Q?", OPTS), pool=POOL, delay=0.0, **kw):
    async def go():
        return await run_gate(gate, rt(client, **kw), q, asyncio.ensure_future(pool_fut(pool, delay)))
    return asyncio.run(go())


def test_option_dist_and_tv():
    d = option_dist([{"token": " A", "logprob": 0.0}, {"token": "B", "logprob": 0.0}, {"token": "x", "logprob": 0}],
                    ["A", "B", "C"])
    assert d == pytest.approx({"A": 0.5, "B": 0.5, "C": 0.0})
    assert total_variation({"A": 1.0}, {"B": 1.0}) == 1.0
    assert total_variation(d, d) == 0.0


def test_yes_prob_logprob_and_text_fallback():
    assert yes_prob([{"token": "Yes", "logprob": 0.0}, {"token": "No", "logprob": 0.0}], "") == pytest.approx(0.5)
    assert yes_prob([{"token": "no", "logprob": -0.1}], "") == 0.0
    assert yes_prob(None, "Yes.") == 1.0
    assert yes_prob(None, "unclear") == 0.5


def test_parse_evaluator_compresses_and_bounds():
    kept = parse_evaluator('{"keep":[1, 7, "x"],"decisive_sentence":{"1":"key fact"}}', POOL)
    assert [c["id"] for c in kept] == ["d1"]
    assert kept[0]["content"] == "key fact" and "full_content" in kept[0]
    assert parse_evaluator("not json, keep 0", POOL)[0]["id"] == "d0"


def test_flip_no_move_withholds_context():
    r = run("flip", FakeClient(cb="B", ctx="B"))
    assert (r.mode, r.answer, r.admitted, r.retained_tokens) == ("closed_book", "B", False, 0)


def test_flip_move_filters_and_reanswers():
    c = FakeClient(cb="A", ctx="B")
    r = run("flip", c)
    assert r.mode == "flip_filtered" and r.admitted and r.answer == "B"
    assert [k["id"] for k in r.kept_context] == ["d1"]
    assert 0 < r.retained_tokens < r.pool_tokens
    assert c.calls[-1] == "answer(filtered context)"


def test_flip_empty_filter_falls_back_to_full_pool():
    r = run("flip", FakeClient(cb="A", ctx="B", keep='{"keep":[]}'))
    assert r.mode == "flip_full" and r.retained_tokens == r.pool_tokens


def test_speculative_filter_is_marked_unused_when_no_move():
    client = FakeClient(cb="B", ctx="B", delay=0.01)
    runtime = rt(client, speculative_filter=True)

    async def go():
        return await run_gate("flip", runtime, Question("Q?", OPTS), asyncio.ensure_future(pool_fut()))
    r = asyncio.run(go())
    assert r.mode == "closed_book"
    spec = [s for s in runtime.trace.spans if s.speculative]
    assert spec and all(not s.used for s in spec)
    assert runtime.trace.summary()["wasted_prompt_tokens"] >= 0


class SlowFilterClient(FakeClient):
    async def chat(self, prompt, *, name="llm", **kw):
        if name.startswith("filter"):
            self.delay = 1.0
        return await super().chat(prompt, name=name, **kw)


def test_cancelled_speculative_span_is_closed_before_summary():
    runtime = rt(SlowFilterClient(cb="B", ctx="B"), speculative_filter=True)

    async def go():
        await run_gate("flip", runtime, Question("Q?", OPTS), asyncio.ensure_future(pool_fut()))
        return runtime.trace.summary(), list(runtime.trace.spans)
    summary, spans = asyncio.run(go())
    assert all(s.end_ms >= s.start_ms for s in spans)
    assert summary["sequential_ms"] >= 0


def test_closed_book_probe_overlaps_retrieval():
    client = FakeClient(cb="B", ctx="B", delay=0.05)
    runtime = rt(client, overlap_retrieval=True)

    async def go():
        return await run_gate("flip", runtime, Question("Q?", OPTS), asyncio.ensure_future(pool_fut(delay=0.05)))
    asyncio.run(go())
    cb = next(s for s in runtime.trace.spans if s.name == "probe(closed-book)")
    assert cb.start_ms < 20  # launched before the 50 ms retrieval completed


def test_grounded_keeps_parametric_when_ungrounded():
    r = run("grounded", FakeClient(cb="A", ctx="B", entail_yes=-4.0, entail_no=-0.02))
    assert r.mode == "ungrounded_keep" and r.answer == "A" and not r.admitted
    assert r.movement > 0.15 and r.groundedness < 0.5


def test_grounded_admits_when_supported():
    r = run("grounded", FakeClient(cb="A", ctx="B"))
    assert r.mode == "grounded_shift" and r.answer == "B" and r.admitted


def test_open_ended_semantic_movement_triggers_flip():
    r = run("flip", FakeClient(cb="A", ctx="B"), q=Question("Open question?"))
    assert r.mode == "flip_filtered" and r.movement > 0.9
    assert r.answer.startswith("free text")


def test_open_ended_equivalent_answers_withhold():
    r = run("flip", FakeClient(same_yes=-0.01, same_no=-5.0), q=Question("Open question?"))
    assert r.mode == "closed_book" and r.movement < 0.1


def test_no_evidence_falls_back_to_closed_book():
    r = run("flip", FakeClient(), pool=[])
    assert r.mode == "no_evidence" and r.answer == "A"


def test_pubmed_query_cleaning_and_relaxation():
    from mug.retrieval.pubmed import clean_query, keyword_query, relaxations
    assert clean_query('Query: "smoking cessation" AND relapse"') == "smoking cessation AND relapse"
    assert relaxations("a AND (b OR c) AND d AND e") == ["a AND (b OR c) AND d", "a AND (b OR c)",
                                                          "(a) OR ((b OR c)) OR (d) OR (e)"]
    assert relaxations("single concept") == []
    assert "year-old" not in keyword_query("A 26-year-old woman has a 2-day history of right flank pain")


class ModelLogClient(FakeClient):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.log: list[tuple[str, str]] = []

    async def chat(self, prompt, *, model, **kw):
        self.log.append((kw.get("name", "llm"), model))
        return await super().chat(prompt, model=model, **kw)


def test_cascade_keeps_full_pool_away_from_answer_model():
    def big_calls(c):
        return [n for n, m in c.log if m == "gpt-4.1"]
    c = ModelLogClient(cb="B", ctx="B")
    r = run("flip", c, scout_model="gpt-4o-mini")
    assert r.mode == "closed_book" and big_calls(c) == ["answer(closed-book)"]
    assert ("probe(full pool)", "gpt-4o-mini") in c.log
    c = ModelLogClient(cb="A", ctx="B")
    r = run("flip", c, scout_model="gpt-4o-mini")
    assert r.admitted and "answer(filtered context)" in big_calls(c) and "probe(full pool)" not in big_calls(c)
    assert ("filter(marginal-utility)", "gpt-4o-mini") in c.log


def test_verifier_admits_only_with_a_real_quote():
    c = FakeClient(cb="A", ctx="B")
    r = run("flip", c, verifier_model="claude-sonnet-5.5")
    assert r.admitted and r.answer == "B" and "Verified" in r.note
    for verdict in ('{"supported": false, "quote": ""}',
                    '{"supported": true, "quote": "Fever always means cystitis."}',  # fabricated: not in passages
                    '{"supported": true, "quote": ""}',  # claims support without pointing to it
                    "garbled"):
        c = FakeClient(cb="A", ctx="B")
        c.verdict = verdict
        r = run("flip", c, verifier_model="claude-sonnet-5.5")
        assert (r.mode, r.answer, r.admitted, r.retained_tokens) == ("unverified_keep", "A", False, 0)


def test_evidence_quality_levels_and_retraction():
    from mug.retrieval.pubmed import evidence_quality
    q = evidence_quality(["Journal Article", "Randomized Controlled Trial", "Review"], False, "2020", now_year=2026)
    assert (q["level"], q["label"], q["flags"]) == (2, "Randomized controlled trial", [])
    q = evidence_quality(["Case Reports"], True, "2001", now_year=2026)
    assert q["label"] == "Retracted" and q["flags"] == ["retracted", "older than 15 years"]
    assert evidence_quality([], False, "", now_year=2026)["level"] == 8
