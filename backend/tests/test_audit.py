from fastapi.testclient import TestClient

import app as appmod
from mug.audit import AuditStore, outcome
from mug.pipeline import Engine
from tests.test_gates import FakeClient, OPTS


def test_outcome_categories():
    assert outcome({"mode": "no_evidence"}) == "no_documents"
    assert outcome({"mode": "unverified_keep", "admitted": False}) == "unverified"
    assert outcome({"mode": "shift", "admitted": True, "verification": {"supported": True}}) == "supported"
    assert outcome({"mode": "shift", "admitted": True}) == "admitted"
    assert outcome({"mode": "closed_book", "admitted": False}) == "no_effect"


def test_proxy_logs_verified_and_unverified(tmp_path, monkeypatch):
    fake = FakeClient()
    eng = Engine.__new__(Engine)
    Engine.__init__(eng)
    eng.client = fake
    monkeypatch.setattr(appmod, "engine", eng)
    monkeypatch.setattr(appmod, "audit", AuditStore(tmp_path / "a.db"))
    c = TestClient(appmod.app)  # no context manager: lifespan would replace the fake engine
    body = {"messages": [{"role": "user", "content": "Flank pain and fever?"}], "options": OPTS,
            "documents": ["Distractor about stones.",
                          {"content": "Flank pain with fever indicates pyelonephritis. More text.", "title": "T"}],
            "verifier_model": "claude-sonnet-5.5", "user": "rag-1"}
    r = c.post("/v1/chat/completions", json=body).json()
    assert r["mug_audit"]["outcome"] == "supported" and r["mug_audit"]["rag_answer"] == "B"
    assert r["choices"][0]["message"]["content"].startswith("B")
    fake.verdict = '{"supported": false, "doc": null, "quote": ""}'
    r2 = c.post("/v1/chat/completions", json=body).json()
    assert r2["mug_audit"]["outcome"] == "unverified" and r2["choices"][0]["message"]["content"].startswith("A")
    lst = c.get("/api/audit").json()
    assert lst["stats"]["n"] == 2 and lst["stats"]["by_outcome"]["unverified"] == 1
    assert lst["items"][0]["client"] == "rag-1"
    full = c.get(f"/api/audit/{r['id']}").json()
    assert len(full["documents"]) == 2 and full["result"]["verification"]["supported"]
    assert c.post("/v1/chat/completions", json={"messages": [{"role": "system", "content": "x"}]}).status_code == 400


from pathlib import Path  # noqa: E402
import asyncio  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402

from mug.audit import export, load_cases, run_batch, summarize  # noqa: E402
from mug.llm import LLMError  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "audit_cases.jsonl"


def _engine(fake):
    eng = Engine.__new__(Engine)
    Engine.__init__(eng)
    eng.client = fake
    return eng


class FailingClient(FakeClient):
    """Every model call fails after 'retries', the way CopilotClient reports it on its span."""

    async def chat(self, prompt, *, model, trace=None, name="llm", speculative=False, **kw):
        span = trace.start(name, "llm", model=model, speculative=speculative) if trace else None
        if span:
            await trace.end(span, ok=False, error="HTTP 429")
        raise LLMError(f"{model} failed: HTTP 429")


def test_outcome_failed_beats_no_effect():
    spans = [{"name": "probe(closed-book)", "kind": "llm", "detail": {"ok": False}}]
    assert outcome({"mode": "closed_book", "admitted": False}, spans) == "failed"
    assert outcome({"mode": "closed_book"}, [{"kind": "llm", "detail": {"ok": True}}]) == "no_effect"


def test_batch_audit_frozen_pool(tmp_path):
    cases = load_cases(FIXTURE.read_text().splitlines())
    fake = FakeClient()
    rows = asyncio.run(run_batch(_engine(fake), cases, policies=["flip", "naive", "closed_book"],
                                 verifier_model="claude-sonnet-5.5"))
    by = {r["id"]: r["policies"] for r in rows}
    assert by["moves"]["flip"]["outcome"] == "supported" and by["moves"]["flip"]["moved"]
    assert by["moves"]["flip"]["correct"] and by["moves"]["closed_book"]["correct"] is False
    assert by["moves"]["naive"]["moved"] is None and by["moves"]["naive"]["answer"] == "B"
    assert by["no-docs"]["flip"]["outcome"] == "no_documents"
    s = summarize(rows)
    assert s["policies"]["flip"]["moved"] == 1 and s["policies"]["flip"]["verifier_rejected"] == 0
    assert s["policies"]["closed_book"]["accuracy"] == 0.0 and s["policies"]["naive"]["failed"] == 0
    paths = export(rows, s, tmp_path / "out")
    assert len(list(csv.DictReader(paths["csv"].open()))) == 6
    assert json.loads(paths["summary"].read_text())["n_cases"] == 2

    fake.verdict = '{"supported": false, "doc": null, "quote": ""}'
    rows = asyncio.run(run_batch(_engine(fake), cases[:1], policies=["flip"], verifier_model="claude-sonnet-5.5"))
    s = summarize(rows)["policies"]["flip"]
    assert s["verifier_rejected"] == 1 and s["moved"] == 1 and s["admitted"] == 0 and s["accuracy"] == 0.0


def test_batch_audit_counts_failures(tmp_path):
    rows = asyncio.run(run_batch(_engine(FailingClient()), load_cases(FIXTURE.read_text().splitlines())[:1],
                                 policies=["flip", "naive"]))
    s = summarize(rows)["policies"]
    assert s["flip"]["failed"] == 1 and s["flip"]["by_outcome"]["no_effect"] == 0 and "accuracy" not in s["flip"]
    assert s["naive"]["failed"] == 1


def test_load_cases_validates():
    import pytest
    with pytest.raises(ValueError):
        load_cases(['{"documents": []}'])
    with pytest.raises(ValueError):
        load_cases(['{"question": "Why?", "documents": [""]}'])


def test_proxy_shadow_mode_never_changes_answer(tmp_path, monkeypatch):
    fake = FakeClient()
    fake.verdict = '{"supported": false, "doc": null, "quote": ""}'  # gate mode would withhold and answer A
    monkeypatch.setattr(appmod, "engine", _engine(fake))
    monkeypatch.setattr(appmod, "audit", AuditStore(tmp_path / "a.db"))
    c = TestClient(appmod.app)
    body = {"messages": [{"role": "user", "content": "Flank pain and fever?"}], "options": OPTS,
            "documents": ["Distractor.", "Flank pain with fever indicates pyelonephritis."],
            "verifier_model": "claude-sonnet-5.5"}
    gated = c.post("/v1/chat/completions", json=body).json()
    assert gated["choices"][0]["message"]["content"].startswith("A") and gated["mug_audit"]["proxy_mode"] == "gate"
    shadow = c.post("/v1/chat/completions", json=body, headers={"X-MUG-Mode": "shadow"}).json()
    assert shadow["choices"][0]["message"]["content"].startswith("B")  # plain-RAG answer, untouched
    assert shadow["mug_audit"]["outcome"] == "unverified" and shadow["mug_audit"]["gate_answer"].startswith("A")
    items = c.get("/api/audit").json()["items"]
    assert {i["proxy_mode"] for i in items} == {"gate", "shadow"}
    assert c.post("/v1/chat/completions", json=body | {"mug_mode": "bogus"}).status_code == 400
    assert c.post("/v1/chat/completions", json=body | {"stream": True}).status_code == 400


def test_audit_store_migrates_old_schema(tmp_path):
    import sqlite3
    p = tmp_path / "old.db"
    db = sqlite3.connect(p)
    db.execute("""CREATE TABLE audit (id TEXT PRIMARY KEY, ts REAL, client TEXT, model TEXT, verifier TEXT,
        question TEXT, n_docs INTEGER, outcome TEXT, answer TEXT, rag_answer TEXT, closed_book_answer TEXT, kept TEXT,
        quote TEXT, movement REAL, prompt_tokens INTEGER, wall_ms REAL, record TEXT)""")
    db.execute("INSERT INTO audit (id, ts, outcome, kept) VALUES ('old', 1.0, 'no_effect', '[]')")
    db.commit()
    db.close()
    items = AuditStore(p).recent()
    assert items[0]["id"] == "old" and items[0]["proxy_mode"] == "gate"
