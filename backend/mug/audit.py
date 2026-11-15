"""Audit mode: measure whether a RAG system's own documents change its answers.

Two entry points share the labelling logic below.

* The audit proxy (``POST /v1/chat/completions`` in ``app.py``). A client that already retrieves its own documents
  sends an OpenAI-style chat request plus a non-standard ``documents`` list. In ``gate`` mode MUG answers through the
  flip gate (optionally with an evidence verifier); in ``shadow`` mode it returns the plain-RAG answer unchanged and
  only labels it. Every request is recorded in a local SQLite file.
* The batch auditor (``python -m mug.audit cases.jsonl``). It replays every policy on the *same frozen pool* per
  case and exports per-case JSONL/CSV plus aggregate counts. Nothing is logged to SQLite.

Nothing leaves the machine except the model calls the request itself makes.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

# Outcome categories, from the deployer's point of view.
#   no_effect       documents did not change the answer (plain RAG and closed-book agree)
#   supported       documents changed the answer, and the verifier model judged a kept passage supportive and
#                   returned a quote that occurs verbatim in it ("quote found + model judged supportive"; this is an
#                   LLM judgement with a string-match check, not a clinical-correctness check)
#   unverified      documents changed the answer but the verifier returned no verifiable quote; in gate mode MUG
#                   kept the closed-book answer, in shadow mode the label is recorded only
#   admitted        documents changed the answer and were admitted (no verifier configured)
#   no_documents    request carried no documents
#   failed          a model call failed after retries, so the label is unknown (never counted as no_effect)
OUTCOMES = ("no_effect", "supported", "unverified", "admitted", "no_documents", "failed")
PROXY_MODES = ("gate", "shadow")


def failed_calls(spans: Iterable[dict] | None) -> list[str]:
    return [s.get("name", "llm") for s in spans or []
            if s.get("kind") == "llm" and (s.get("detail") or {}).get("ok") is False]


def outcome(res: dict, spans: Iterable[dict] | None = None) -> str:
    if failed_calls(spans):
        return "failed"
    mode = res.get("mode")
    if mode == "no_evidence":
        return "no_documents"
    if mode == "unverified_keep":
        return "unverified"
    if res.get("admitted"):
        return "supported" if (res.get("verification") or {}).get("supported") else "admitted"
    return "no_effect"


def moved(res: dict) -> bool | None:
    """Did the documents move the answer? None for the fixed policies, which never measure movement."""
    mode = res.get("mode")
    if res.get("gate") in ("naive", "closed_book"):
        return None
    if mode in ("closed_book", "no_evidence"):
        return False
    return True


class AuditStore:
    _COLS = ("id", "ts", "client", "model", "verifier", "question", "n_docs", "outcome", "answer", "rag_answer",
             "closed_book_answer", "kept", "quote", "movement", "prompt_tokens", "wall_ms", "record", "proxy_mode",
             "gate_answer")

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS audit (
            id TEXT PRIMARY KEY, ts REAL, client TEXT, model TEXT, verifier TEXT, question TEXT, n_docs INTEGER,
            outcome TEXT, answer TEXT, rag_answer TEXT, closed_book_answer TEXT, kept TEXT, quote TEXT,
            movement REAL, prompt_tokens INTEGER, wall_ms REAL, record TEXT)""")
        have = {r[1] for r in self.db.execute("PRAGMA table_info(audit)")}
        if "proxy_mode" not in have:  # logs written before shadow mode existed are gate-mode logs
            self.db.execute("ALTER TABLE audit ADD COLUMN proxy_mode TEXT DEFAULT 'gate'")
        if "gate_answer" not in have:
            self.db.execute("ALTER TABLE audit ADD COLUMN gate_answer TEXT")
        self.db.commit()

    def record(self, *, client: str, model: str, verifier: str | None, question: str, documents: list[dict],
               out: dict, proxy_mode: str = "gate", returned: str | None = None) -> dict:
        """``returned`` is the text sent back to the client (defaults to the gated answer)."""
        res = out["result"]
        gate_answer = res.get("answer_text") or ""
        row = {
            "id": f"aud_{uuid.uuid4().hex[:12]}", "ts": time.time(), "client": client, "model": model,
            "verifier": verifier, "question": question, "n_docs": len(documents),
            "outcome": outcome(res, out.get("spans")),
            "answer": gate_answer if returned is None else returned, "rag_answer": res.get("context_answer"),
            "closed_book_answer": res.get("closed_book_answer"),
            "kept": [c.get("pool_index") for c in res.get("kept_context", [])],
            "quote": (res.get("verification") or {}).get("quote", ""), "movement": res.get("movement"),
            "prompt_tokens": out["summary"].get("prompt_tokens", 0), "wall_ms": out["summary"].get("wall_ms", 0),
            "proxy_mode": proxy_mode, "gate_answer": gate_answer,
        }
        record = {"result": res, "documents": documents, "summary": out["summary"], "proxy_mode": proxy_mode,
                  "failed_calls": failed_calls(out.get("spans"))}
        vals = {**row, "kept": json.dumps(row["kept"]), "record": json.dumps(record, default=str)}
        self.db.execute(f"INSERT INTO audit ({', '.join(self._COLS)}) VALUES ({', '.join('?' * len(self._COLS))})",
                        tuple(vals[c] for c in self._COLS))
        self.db.commit()
        return row

    def recent(self, limit: int = 100, outcome_: str | None = None) -> list[dict]:
        cols = [c for c in self._COLS if c != "record"]
        q = f"SELECT {', '.join(cols)} FROM audit"
        args: list = []
        if outcome_:
            q += " WHERE outcome = ?"
            args.append(outcome_)
        q += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        rows = [dict(zip(cols, r)) for r in self.db.execute(q, args)]
        for r in rows:
            r["kept"] = json.loads(r["kept"] or "[]")
            r["proxy_mode"] = r["proxy_mode"] or "gate"
        return rows

    def get(self, id_: str) -> dict | None:
        r = self.db.execute("SELECT record FROM audit WHERE id = ?", (id_,)).fetchone()
        return json.loads(r[0]) if r else None

    def stats(self) -> dict:
        n = self.db.execute("SELECT COUNT(*) FROM audit").fetchone()[0]
        by = dict(self.db.execute("SELECT outcome, COUNT(*) FROM audit GROUP BY outcome").fetchall())
        tok, wall = self.db.execute("SELECT AVG(prompt_tokens), AVG(wall_ms) FROM audit").fetchone()
        return {"n": n, "by_outcome": {o: by.get(o, 0) for o in OUTCOMES},
                "mean_prompt_tokens": tok or 0, "mean_wall_ms": wall or 0}


# ------------------------------------------------------------------ frozen-pool batch audit
BATCH_POLICIES = ("flip", "grounded", "logit_shift", "naive", "closed_book")


def load_cases(lines: Iterable[str]) -> list[dict]:
    """Parse JSONL ``{id?, question, options?, documents: [str | {content, title?, url?}], reference?}``."""
    cases = []
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        c = json.loads(line)
        if not isinstance(c.get("question"), str) or len(c["question"].strip()) < 3:
            raise ValueError(f"line {n}: 'question' is required")
        docs = []
        for i, d in enumerate(c.get("documents") or []):
            d = {"content": d} if isinstance(d, str) else dict(d)
            if not str(d.get("content") or "").strip():
                raise ValueError(f"line {n}: document {i} has no content")
            docs.append(d)
        cases.append({"id": str(c.get("id") or f"case-{len(cases) + 1}"), "question": c["question"].strip(),
                      "options": c.get("options") or None, "documents": docs, "reference": c.get("reference")})
    return cases


def frozen_pool(docs: list[dict]) -> list[dict]:
    return [{"id": f"doc:{i}", "source": "user", "title": d.get("title") or f"Document {i + 1}",
             "content": d["content"], "url": d.get("url")} for i, d in enumerate(docs)]


def _is_correct(res: dict, reference: Any) -> bool | None:
    if reference is None:
        return None
    ans = res.get("answer")
    if ans is None:
        return False
    return str(ans).strip().upper() == str(reference).strip().upper()


async def audit_case(engine, case: dict, *, policies: Iterable[str], model: str | None = None,
                     verifier_model: str | None = None) -> dict:
    """Run every policy on the case's frozen pool. The verifier applies to the flip gate only."""
    pool = frozen_pool(case["documents"])
    row: dict[str, Any] = {"id": case["id"], "question": case["question"], "n_docs": len(pool),
                           "reference": case["reference"], "policies": {}}
    for policy in policies:
        try:
            out = await engine.ask(question=case["question"], options=case["options"], gate=policy,
                                   answer_model=model, pool=pool,
                                   verifier_model=verifier_model if policy == "flip" else None)
        except Exception as e:  # noqa: BLE001
            row["policies"][policy] = {"outcome": "failed", "failed": True, "error": f"{type(e).__name__}: {e}"[:300]}
            continue
        res, summ = out["result"], out["summary"]
        fails = failed_calls(out["spans"])
        ver = res.get("verification") or {}
        row["policies"][policy] = {
            "answer": res.get("answer"), "answer_text": res.get("answer_text"), "mode": res.get("mode"),
            "outcome": outcome(res, out["spans"]), "moved": moved(res), "admitted": res.get("admitted"),
            "movement": res.get("movement"), "closed_book_answer": res.get("closed_book_answer"),
            "context_answer": res.get("context_answer"),
            "kept": [c.get("pool_index") for c in res.get("kept_context", [])],
            "verifier_supported": ver.get("supported") if ver else None, "verifier_quote": ver.get("quote", ""),
            "retained_tokens": res.get("retained_tokens"), "pool_tokens": res.get("pool_tokens"),
            "prompt_tokens": summ.get("prompt_tokens"), "llm_calls": summ.get("llm_calls"),
            "wall_ms": summ.get("wall_ms"), "failed": bool(fails), "failed_calls": fails,
            "correct": None if fails else _is_correct(res, case["reference"]),
        }
    return row


async def run_batch(engine, cases: list[dict], *, policies: Iterable[str] = ("flip", "naive", "closed_book"),
                    model: str | None = None, verifier_model: str | None = None, concurrency: int = 2) -> list[dict]:
    policies = list(policies)
    bad = [p for p in policies if p not in BATCH_POLICIES]
    if bad:
        raise ValueError(f"unknown policies {bad}; choose from {BATCH_POLICIES}")
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(c: dict) -> dict:
        async with sem:
            return await audit_case(engine, c, policies=policies, model=model, verifier_model=verifier_model)
    return list(await asyncio.gather(*(one(c) for c in cases)))


def summarize(rows: list[dict]) -> dict:
    """Aggregate counts per policy. Rates are descriptive; they are not clinical-correctness measures."""
    out: dict[str, Any] = {"n_cases": len(rows), "policies": {}}
    names = list(dict.fromkeys(p for r in rows for p in r["policies"]))
    for p in names:
        recs = [r["policies"][p] for r in rows if p in r["policies"]]
        ok = [x for x in recs if not x.get("failed")]
        graded = [x for x in ok if x.get("correct") is not None]
        s = {"n": len(recs), "failed": len(recs) - len(ok),
             "by_outcome": {o: sum(1 for x in recs if x.get("outcome") == o) for o in OUTCOMES}}
        if p not in ("naive", "closed_book"):
            s["moved"] = sum(1 for x in ok if x.get("moved"))
            s["admitted"] = sum(1 for x in ok if x.get("admitted"))
            s["verifier_rejected"] = s["by_outcome"]["unverified"]
        if graded:
            s["accuracy"] = round(sum(1 for x in graded if x["correct"]) / len(graded), 4)
            s["n_graded"] = len(graded)
        if ok:
            s["mean_prompt_tokens"] = round(sum(x.get("prompt_tokens") or 0 for x in ok) / len(ok), 1)
            s["mean_wall_ms"] = round(sum(x.get("wall_ms") or 0 for x in ok) / len(ok), 1)
        out["policies"][p] = s
    return out


CSV_FIELDS = ("id", "policy", "outcome", "moved", "admitted", "answer", "closed_book_answer", "context_answer",
              "reference", "correct", "movement", "kept", "verifier_supported", "verifier_quote", "retained_tokens",
              "pool_tokens", "prompt_tokens", "llm_calls", "wall_ms", "failed", "question")


def export(rows: list[dict], summary: dict, out_dir: Path) -> dict[str, Path]:
    """Write cases.jsonl, cases.csv (one row per case x policy) and summary.json. Document text is not copied,
    only kept-document indices and the verifier's quote."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"jsonl": out_dir / "cases.jsonl", "csv": out_dir / "cases.csv", "summary": out_dir / "summary.json"}
    with paths["jsonl"].open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    with paths["csv"].open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            for p, x in r["policies"].items():
                w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in
                            {**{f: x.get(f) for f in CSV_FIELDS}, "id": r["id"], "policy": p,
                             "reference": r["reference"], "question": r["question"]}.items()})
    paths["summary"].write_text(json.dumps(summary, indent=2))
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m mug.audit",
                                 description="Frozen-pool audit: does each case's own document pool change the answer?")
    ap.add_argument("cases", help="JSONL: {id?, question, options?, documents: [...], reference?} per line")
    ap.add_argument("--out", default="audit_out", help="output directory (cases.jsonl, cases.csv, summary.json)")
    ap.add_argument("--policies", default="flip,naive,closed_book", help=f"comma-separated subset of {BATCH_POLICIES}")
    ap.add_argument("--model", default=None, help="answer model (default: MUG_ANSWER_MODEL)")
    ap.add_argument("--verifier", default=None, help="optional verifier model for the flip gate")
    ap.add_argument("--concurrency", type=int, default=2)
    args = ap.parse_args(argv)

    from .pipeline import Engine  # imported here so the store can be used without model credentials

    cases = load_cases(Path(args.cases).read_text().splitlines())

    async def go() -> list[dict]:
        engine = Engine()
        try:
            return await run_batch(engine, cases, policies=args.policies.split(","), model=args.model,
                                   verifier_model=args.verifier, concurrency=args.concurrency)
        finally:
            await engine.client.aclose()

    rows = asyncio.run(go())
    summary = summarize(rows)
    paths = export(rows, summary, Path(args.out))
    json.dump(summary, sys.stdout, indent=2)
    print(f"\nwrote {', '.join(str(p) for p in paths.values())}", file=sys.stderr)


if __name__ == "__main__":
    main()
