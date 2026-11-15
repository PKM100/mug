"""Build the frozen replay cases shown by the UI's Replay mode (#/replay) from existing eval records.

No model calls. Reads recorded flip-gate traces from results/scale_*_d16_s1.jsonl (17-passage paper-protocol pools,
gpt-4o-mini), the evidence verifier's recorded verdicts from results/verifier/verify_claude-sonnet-55.jsonl, and the
passage texts from the local MedRAG textbook index / PubMedQA file. Writes frontend/public/replay/{cases,audit}.json.

Run: ../backend/.venv/bin/python build_replay_cases.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("MUG_DATA_DIR", HERE.parent / "data"))
OUT = HERE.parent / "frontend" / "public" / "replay"
VERIFIER = "claude-sonnet-5.5"
PMQA_OPTIONS = {"A": "yes", "B": "no", "C": "maybe"}

CASES = [
    ("no-movement", "scale_medqa_gpt4omini_d16_s1", "medqa:0359",
     "Evidence does not move the answer",
     "The closed-book and full-pool probes agree, so the flip gate withholds all 17 passages and answers closed-book "
     "after two parallel calls."),
    ("withheld", "scale_medqa_gpt4omini_d16_s1", "medqa:1040",
     "Tempting movement, verifier finds no support: evidence withheld",
     "The 17-passage pool flips a correct closed-book answer to a wrong one. The filter keeps two passages, but the "
     "evidence verifier finds no passage that establishes the moved answer, so MUG keeps the closed-book answer."),
    ("supported", "scale_pubmedqa_gpt4omini_d16_s1", "pubmedqa:11567820",
     "Supported movement: evidence admitted with a verbatim quote",
     "The study abstract flips a wrong closed-book answer to the correct one. The filter keeps only the abstract, and "
     "the verifier returns a quote that occurs verbatim in it."),
]


def tok(text: str) -> int:
    return max(1, len(text) // 4)


def load_record(run: str, qid: str) -> dict:
    for line in (HERE / "results" / f"{run}.jsonl").open():
        r = json.loads(line)
        if r["qid"] == qid and r["gate"] == "flip" and "error" not in r:
            return r
    raise KeyError((run, qid))


def load_verdict(run: str, qid: str) -> dict:
    for line in (HERE / "results" / "verifier" / "verify_claude-sonnet-55.jsonl").open():
        r = json.loads(line)
        if r["run"] == run and r["qid"] == qid:
            return r
    raise KeyError((run, qid))


def textbook_passages(ids: set[str]) -> dict[str, dict]:
    out = {}
    with (DATA / "index" / "meta.jsonl").open() as fh:
        for line in fh:
            if not any(i in line for i in ids):
                continue
            m = json.loads(line)
            if m["id"] in ids:
                out[m["id"]] = m
                if len(out) == len(ids):
                    break
    return out


def item(qid: str) -> dict:
    ds, key = qid.split(":", 1)
    if ds == "medqa":
        q = json.loads((DATA / "mirage_benchmark.json").read_text())["medqa"][key]
        return {"question": q["question"], "options": q["options"], "gold": q["answer"], "gold_context": None}
    q = json.loads((DATA / "pubmedqa_labeled.json").read_text())[key]
    return {"question": q["QUESTION"], "options": dict(PMQA_OPTIONS),
            "gold": {"yes": "A", "no": "B", "maybe": "C"}[q["final_decision"]], "gold_context": " ".join(q["CONTEXTS"])}


def span(s: dict, **detail) -> dict:
    return {"name": s["name"], "kind": s["kind"], "start_ms": round(s["start_ms"], 1), "end_ms": round(s["end_ms"], 1),
            "duration_ms": round(s["end_ms"] - s["start_ms"], 1), "model": s.get("model"),
            "prompt_tokens": s.get("prompt_tokens", 0), "completion_tokens": s.get("completion_tokens", 0),
            "speculative": s.get("speculative", False), "used": s.get("used", True), "detail": detail}


def build(key: str, run: str, qid: str, title: str, blurb: str, tb: dict[str, dict]) -> tuple[dict, dict]:
    rec, it = load_record(run, qid), item(qid)
    opts = it["options"]
    pool = []
    for i, pid in enumerate(rec["pool_ids"]):
        if pid == "GOLD":
            pool.append({"id": f"PMID:{qid.split(':')[1]}", "source": "pubmed", "title": "Study abstract (PubMedQA context)",
                         "content": it["gold_context"], "url": f"https://pubmed.ncbi.nlm.nih.gov/{qid.split(':')[1]}/",
                         "pool_index": i})
        else:
            m = tb[pid]
            pool.append({"id": pid, "source": "textbook", "title": m.get("title", pid).replace("_", " "),
                         "content": m["content"], "url": None, "pool_index": i})
    by_id = {c["id"] if c["id"] != f"PMID:{qid.split(':')[1]}" else "GOLD": c for c in pool}
    filter_kept = [by_id[k] for k in rec["kept_ids"]]
    moved = rec["mode"] != "closed_book"
    spans = []
    for s in rec["spans"]:
        d = {}
        if s["name"] == "decide(movement)":
            d = {"moved": moved, "movement": rec["movement"]}
        if s["name"] == "filter(marginal-utility)":
            d = {"kept": [c["pool_index"] + 1 for c in filter_kept]}
        spans.append(span(s, **d))
    verification = None
    mode, admitted, answer, kept, retained = rec["mode"], rec["admitted"], rec["pred"], filter_kept, rec["retained_tokens"]
    note = "Evidence did not move the answer; context withheld."
    if moved:
        v = load_verdict(run, qid)
        vd = v["verdict"] or {}
        t = max(s["end_ms"] for s in rec["spans"])
        untimed = "not recorded: verdict comes from a separate offline verifier pass over this trace"
        spans.append({"name": "verify(support)", "kind": "llm", "start_ms": round(t, 1), "end_ms": round(t, 1),
                      "duration_ms": 0, "model": VERIFIER, "prompt_tokens": v["prompt_tokens"],
                      "completion_tokens": v["completion_tokens"], "speculative": False, "used": True,
                      "detail": {"timing": untimed}})
        spans.append({"name": "decide(verified support)", "kind": "decision", "start_ms": round(t, 1),
                      "end_ms": round(t, 1), "duration_ms": 0, "model": None, "prompt_tokens": 0,
                      "completion_tokens": 0, "speculative": False, "used": True,
                      "detail": {"supported": bool(vd.get("supported")), "quote": vd.get("quote", "")}})
        verification = {"supported": bool(vd.get("supported")), "doc": vd.get("doc"), "quote": vd.get("quote", ""),
                        "model": VERIFIER}
        if vd.get("supported"):
            note = ("Evidence moved the answer; re-answered on the passages that carry it."
                    f' Verified: "{vd["quote"][:200]}"')
        else:
            mode, admitted, answer, kept, retained = "unverified_keep", False, rec["cb_pred"], [], 0
            note = ("Evidence moved the answer, but an independent verifier found no passage that establishes it; "
                    "kept the parametric answer. (The filter had kept passages "
                    + ", ".join(f"#{c['pool_index'] + 1}" for c in filter_kept) + ".)")
    summ = dict(rec["summary"])
    summ.setdefault("llm_calls_used", summ.get("llm_calls"))
    result = {"gate": "flip", "mode": mode, "answer": answer, "answer_text": f"{answer}. {opts.get(answer, '')}",
              "admitted": admitted, "kept_context": kept, "closed_book_answer": rec["cb_pred"],
              "context_answer": rec["ctx_pred"], "movement": rec["movement"], "groundedness": None,
              "retained_tokens": retained, "pool_tokens": rec["pool_tokens"], "note": note, "verification": verification}
    request = {"question": it["question"], "options": opts, "gate": "flip", "answer_model": rec["model"],
               "probe_model": rec["model"], "k_textbook": 0, "k_pubmed": 0, "rewrite_query": False,
               "user_passages": [], "speculative_filter": False, "overlap_retrieval": True, "scout_model": None,
               "verifier_model": VERIFIER if moved else None, "tau_m": 0.15, "tau_g": 0.5, "tau_sem": 0.5}
    case = {
        "key": key, "title": title, "blurb": blurb, "request": request,
        "result": {"type": "result", "question": it["question"], "options": opts, "question_type": "mcq",
                   "result": result, "pool": pool, "summary": summ, "spans": spans},
        "provenance": {"run": f"eval/results/{run}.jsonl", "qid": qid, "gold": it["gold"],
                       "recorded": "2026-11-13 (gpt-4o-mini via GitHub Copilot API)",
                       "verifier": "eval/results/verifier/verify_claude-sonnet-55.jsonl" if moved else None,
                       "differences": [
                           "Pool: 17-passage paper-protocol pool (top textbook chunks"
                           + (" plus the gold abstract" if qid.startswith("pubmedqa") else "") + "), not live retrieval.",
                           "Kept passages are shown in full; decisive-sentence compression was not recorded.",
                       ] + ([f"Verifier ({VERIFIER}) verdict was recorded in a separate pass; its latency is not "
                             "part of the trace."] if moved else [])},
    }
    docs = [{"content": c["content"], "title": c["title"], "url": c["url"]} for c in pool]
    audit_item = {"id": f"replay_{key}", "ts": 1794614400 + len(key), "client": "replay", "model": rec["model"],
                  "verifier": VERIFIER if moved else None, "question": it["question"], "n_docs": len(pool),
                  "outcome": {"closed_book": "no_effect", "unverified_keep": "unverified"}.get(mode, "supported"),
                  "answer": result["answer_text"], "gate_answer": result["answer_text"],
                  "rag_answer": f"{rec['ctx_pred']}. {opts.get(rec['ctx_pred'], '')}",
                  "closed_book_answer": f"{rec['cb_pred']}. {opts.get(rec['cb_pred'], '')}",
                  "kept": [c["pool_index"] for c in kept], "quote": (verification or {}).get("quote", ""),
                  "movement": rec["movement"], "prompt_tokens": summ["prompt_tokens"], "wall_ms": summ["wall_ms"],
                  "proxy_mode": "gate"}
    audit_rec = {"result": result, "documents": docs, "summary": summ, "proxy_mode": "gate", "failed_calls": []}
    return case, {"item": audit_item, "record": audit_rec}


def main() -> None:
    ids = set()
    for _, run, qid, *_ in CASES:
        ids |= {p for p in load_record(run, qid)["pool_ids"] if p != "GOLD"}
    tb = textbook_passages(ids)
    cases, audits = [], []
    for c in CASES:
        case, aud = build(*c, tb)
        cases.append(case)
        audits.append(aud)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"note": "Recorded cases, replayed without any model or network call.", "cases": cases}
    (OUT / "cases.json").write_text(json.dumps(meta, indent=1))
    items = [a["item"] for a in audits]
    by = {o: sum(1 for i in items if i["outcome"] == o)
          for o in ("no_effect", "supported", "unverified", "admitted", "no_documents", "failed")}
    audit = {"stats": {"n": len(items), "by_outcome": by,
                       "mean_prompt_tokens": sum(i["prompt_tokens"] for i in items) / len(items),
                       "mean_wall_ms": sum(i["wall_ms"] for i in items) / len(items)},
             "items": items, "records": {a["item"]["id"]: a["record"] for a in audits}}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=1))
    for c in cases:
        r = c["result"]["result"]
        print(c["key"], r["mode"], "cb", r["closed_book_answer"], "ctx", r["context_answer"], "final", r["answer"],
              "gold", c["provenance"]["gold"], "kept", [k["id"] for k in r["kept_context"]])


if __name__ == "__main__":
    main()
