"""Evaluation harness: runs MUG gates through the *serving* pipeline (Engine.ask).

Experiments
  fidelity : paper protocol pools (MedQA: top-5 textbook; PubMedQA: gold abstract + 4
             textbook distractors), MCQ, gates x {closed_book, naive, flip, logit_shift, grounded}
  open     : same pools, options hidden -> open-ended answers; semantic-movement gates;
             answers mapped back to a label by an independent judge model
  sched    : live hybrid retrieval (PubMed + textbooks), flip gate under four schedules
             (serial / parallel probes / overlap / overlap+speculative-filter), latency & cost
  live     : live hybrid retrieval, MCQ, gates -> end-to-end accuracy in the deployed setting
  scale    : paper-protocol pools grown to --n-distract passages (pool-size scaling)
  cascade  : paper-protocol pools; an expensive answer model (--model) with a cheap scout
             (--scout) running the probes and filter, vs. the answer model's fixed policies

Usage
  PYTHONPATH=backend backend/.venv/bin/python eval/run_eval.py fidelity --dataset medqa --n 150
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from mug.config import settings  # noqa: E402
from mug.llm import LLMError  # noqa: E402
from mug.pipeline import Engine  # noqa: E402

RESULTS = ROOT / "eval" / "results"
PMQA_OPTIONS = {"A": "yes", "B": "no", "C": "maybe"}
JUDGE_MODEL = "gpt-4.1"


# ------------------------------------------------------------------ data
def load_items(dataset: str, n: int, seed: int, start: int = 0) -> list[dict]:
    """The first n items of the seed shuffle, or items [start, start+n) with --start (E11 subsets keep E10's order)."""
    if dataset == "medqa":
        data = json.loads((settings.data_dir / "mirage_benchmark.json").read_text())["medqa"]
        items = [{"qid": f"medqa:{k}", "question": q["question"], "options": q["options"], "answer": q["answer"]}
                 for k, q in data.items()]
    elif dataset == "pubmedqa":
        data = json.loads((settings.data_dir / "pubmedqa_labeled.json").read_text())
        l2l = {"yes": "A", "no": "B", "maybe": "C"}
        items = [{"qid": f"pubmedqa:{k}", "question": q["QUESTION"], "options": dict(PMQA_OPTIONS),
                  "answer": l2l[q["final_decision"]], "gold_context": " ".join(q["CONTEXTS"])}
                 for k, q in data.items() if q["final_decision"] in l2l]
        _attach_neighbors(items)
        _attach_cf(items)
    else:
        raise ValueError(dataset)
    random.Random(seed).shuffle(items)
    return items[start:start + n]


_TOK = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of in and or to is are was were be for with on by does do did can that this as at from".split())


def _attach_neighbors(items: list[dict]) -> None:
    """E10 swap arm: for each PubMedQA item, the most question-similar other abstract (Jaccard over content
    words) whose study reached a different conclusion. Plausible, on-topic, and not support for the gold answer.
    E11 adds the same-label control: the most similar other abstract whose final_decision EQUALS the item's."""
    toks = [set(_TOK.findall(it["question"].lower())) - _STOP for it in items]
    for i, it in enumerate(items):
        best, bj = {False: None, True: None}, {False: -1.0, True: -1.0}
        for j, ot in enumerate(items):
            if j == i:
                continue
            same = ot["answer"] == it["answer"]
            u = len(toks[i] | toks[j]) or 1
            jac = len(toks[i] & toks[j]) / u
            if jac > bj[same] or (jac == bj[same] and ot["qid"] < best[same]["qid"]):
                best[same], bj[same] = ot, jac
        b, s = best[False], best[True]
        it["swap_context"], it["swap_qid"], it["swap_jaccard"] = b["gold_context"], b["qid"], round(bj[False], 3)
        it["swap_label"] = b["answer"]
        it["same_context"], it["same_qid"], it["same_jaccard"] = s["gold_context"], s["qid"], round(bj[True], 3)


CF_PATH = Path(os.environ.get("E11_CF_PATH") or RESULTS / "cf_abstracts.jsonl")  # env override: smoke tests


def _attach_cf(items: list[dict]) -> None:
    """E11 counterfactual arm: the source abstract minimally edited (make_cf.py) to support a different answer."""
    if not CF_PATH.exists():
        return
    l2l = {"yes": "A", "no": "B", "maybe": "C"}
    cf = {}
    for line in CF_PATH.read_text().splitlines():
        r = json.loads(line)
        if r.get("edited_abstract") and r.get("target_ok"):
            cf[r["qid"]] = r
    for it in items:
        r = cf.get(it["qid"])
        if r:
            it["cf_context"], it["cf_label"] = r["edited_abstract"], l2l[r["target_label"]]
            it["cf_valid"] = r.get("valid")


INJECT = {"gold": ("GOLD", "gold_context"), "swap": ("SWAP", "swap_context"), "cf": ("CF", "cf_context"),
          "samelabel": ("SAME", "same_context")}


def paper_pool(engine: Engine, it: dict, seed: int, n_distract: int = 4, gold: bool = True,
               swap: bool = False, arm: str | None = None) -> list[dict]:
    """Exact pool construction from the MUG paper's research code: the question's top-k textbook chunks,
    plus (PubMedQA) the gold abstract. gold=False drops the abstract and keeps the pool size (E8).
    arm replaces the abstract (swap: E10; cf, samelabel: E11) at the same pool position."""
    tb = engine.textbooks
    if it["qid"].startswith("medqa") or not gold:
        return tb.search(it["question"] + " " + " ".join(it["options"].values()), n_distract + 1)
    arm = arm or ("swap" if swap else "gold")
    pid, key = INJECT[arm]
    gold = {"id": pid, "source": arm, "title": "Study abstract", "content": it[key], "url": None,
            "is_gold": arm == "gold"}
    pool = [gold] + tb.search(it["question"], n_distract)
    random.Random(seed).shuffle(pool)
    return pool


# ------------------------------------------------------------------ judges (open-ended)
async def judge_label(engine: Engine, it: dict, answer: str | None) -> str | None:
    if not answer:
        return None
    opts = "\n".join(f"{k}. {v}" for k, v in it["options"].items())
    if it["qid"].startswith("pubmedqa"):
        prompt = (f"Question: {it['question']}\n\nFree-text answer: {answer}\n\n"
                  "Classify the answer's bottom-line conclusion to the question.\n"
                  f"{opts}\nReply with ONLY the letter.")
    else:
        prompt = (f"Question: {it['question']}\n\nCandidate options:\n{opts}\n\nFree-text answer: {answer}\n\n"
                  "Which single option does the free-text answer commit to? If it commits to none of them, "
                  "reply X. Reply with ONLY the letter.")
    try:
        r = await engine.client.chat(prompt, model=JUDGE_MODEL, max_tokens=2)
    except LLMError:
        return None
    up = r.content.strip().upper()
    return next((c for c in up if c in it["options"] or c == "X"), None)


# ------------------------------------------------------------------ runner
async def run_one(engine: Engine, it: dict, *, gate: str, open_mode: bool, pool: list[dict] | None,
                  model: str, variant: dict) -> dict:
    t = time.time()
    try:
        out = await engine.ask(question=it["question"], options=None if open_mode else it["options"], gate=gate,
                               answer_model=model, pool=pool, **variant)
    except Exception as e:  # noqa: BLE001
        return {"qid": it["qid"], "gate": gate, "error": f"{type(e).__name__}: {e}"}
    failed = [sp for sp in out["spans"] if sp["kind"] == "llm" and sp["detail"].get("ok") is False]
    if failed:  # an API failure would otherwise be scored as a wrong answer; record it so a re-run retries it
        return {"qid": it["qid"], "gate": gate, "error": f"{failed[0]['name']}: {failed[0]['detail'].get('error')}"}
    res = out["result"]
    rec = {"qid": it["qid"], "gate": gate, "gold": it["answer"], "mode": res["mode"], "admitted": res["admitted"],
           "movement": res["movement"], "groundedness": res["groundedness"],
           "retained_tokens": res["retained_tokens"], "pool_tokens": res["pool_tokens"],
           "summary": out["summary"], "spans": [{k: s[k] for k in ("name", "kind", "start_ms", "end_ms", "speculative",
                                                                      "used", "prompt_tokens", "completion_tokens",
                                                                      "model")} for s in out["spans"]],
           "kept_ids": [c["id"] for c in res["kept_context"]], "pool_ids": [c["id"] for c in out["pool"]],
           "pool_sources": [c.get("source") for c in out["pool"]], "client_s": round(time.time() - t, 3)}
    if open_mode:
        rec["cb_text"], rec["ctx_text"], rec["answer_text"] = (res["closed_book_answer"], res["context_answer"],
                                                               res["answer_text"])
        labels = await asyncio.gather(judge_label(engine, it, res["answer"]),
                                      judge_label(engine, it, res["closed_book_answer"]),
                                      judge_label(engine, it, res["context_answer"]))
        rec["pred"], rec["cb_pred"], rec["ctx_pred"] = labels
    else:
        rec["pred"], rec["cb_pred"], rec["ctx_pred"] = res["answer"], res["closed_book_answer"], res["context_answer"]
    rec["correct"] = rec["pred"] == it["answer"]
    return rec


async def run_experiment(args) -> Path:
    engine = Engine()
    engine.use_cache = args.exp != "sched"
    await engine.warm()
    assert engine.textbooks.ready, engine.textbooks.error
    items = load_items(args.dataset, args.n, args.seed, args.start)
    if args.cf_gold:
        missing = [it["qid"] for it in items if "cf_context" not in it]
        if missing:
            print(f"  cf: skipping {len(missing)} items without a usable counterfactual edit", flush=True)
        items = [it for it in items if "cf_context" in it]
    RESULTS.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.replace(".", "").replace("-", "")
    dtag = f"_d{args.n_distract}" if args.n_distract != 4 else ""
    if args.no_gold:
        dtag += "_nogold"
    if args.swap_gold:
        dtag += "_swap"
    if args.cf_gold:
        dtag += "_cf"
    if args.samelabel_gold:
        dtag += "_samelabel"
    if args.tag:
        dtag += f"_{args.tag}"
    out_path = RESULTS / f"{args.exp}_{args.dataset}_{model_tag}{dtag}_s{args.seed}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                done.add((r["qid"], r["gate"], r.get("variant", "")))

    if args.exp == "sched":
        plan = [("flip", "serial", dict(overlap_retrieval=False, parallel_probes=False, speculative_filter=False)),
                ("flip", "parallel", dict(overlap_retrieval=False, speculative_filter=False)),
                ("flip", "overlap", dict(overlap_retrieval=True, speculative_filter=False)),
                ("flip", "overlap+spec", dict(overlap_retrieval=True, speculative_filter=True))]
    elif args.exp == "cascade":
        scout = dict(scout_model=args.scout)
        plan = [("closed_book", "", {}), ("naive", "", {}),
                ("flip", "cascade", scout), ("grounded", "cascade", scout)]
        plan = [p for p in plan if p[0] in args.gates.split(",")]
    else:
        plan = [(g, "", {}) for g in args.gates.split(",")]
    open_mode = args.exp == "open"
    use_paper_pool = args.exp in ("fidelity", "open", "scale", "cascade")
    arm = "swap" if args.swap_gold else "cf" if args.cf_gold else "samelabel" if args.samelabel_gold else "gold"
    sem = asyncio.Semaphore(args.concurrency)
    fh = out_path.open("a")
    n_done = 0

    async def job(it: dict) -> None:
        nonlocal n_done
        async with sem:
            pool = (paper_pool(engine, it, args.seed + zlib.crc32(it["qid"].encode()) % 1000, args.n_distract,
                               gold=not args.no_gold, arm=arm)
                    if use_paper_pool else None)
            order = list(plan)
            if args.exp == "sched":  # interleave schedules per question to cancel network drift
                random.Random(it["qid"]).shuffle(order)
            for gate, vname, variant in order:
                if (it["qid"], gate, vname) in done:
                    continue
                rec = await run_one(engine, it, gate=gate, open_mode=open_mode, pool=pool, model=args.model,
                                    variant=variant)
                rec["variant"] = vname
                rec["exp"], rec["dataset"], rec["model"] = args.exp, args.dataset, args.model
                rec["n_distract"] = args.n_distract
                rec["gold_in_pool"] = not args.no_gold
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            n_done += 1
            if n_done % 10 == 0:
                print(f"  [{args.exp}/{args.dataset}] {n_done}/{len(items)}", flush=True)

    t0 = time.time()
    await asyncio.gather(*(job(it) for it in items))
    fh.close()
    await engine.client.aclose()
    print(f"done {out_path} in {time.time() - t0:.0f}s")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["fidelity", "open", "sched", "live", "scale", "cascade"])
    ap.add_argument("--dataset", choices=["medqa", "pubmedqa"], required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--gates", default="closed_book,naive,flip,logit_shift,grounded")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--n-distract", type=int, default=4, help="textbook passages added to each paper-protocol pool")
    ap.add_argument("--no-gold", action="store_true",
                    help="PubMedQA: leave the gold abstract out of the pool (same pool size), for E8")
    ap.add_argument("--swap-gold", action="store_true",
                    help="PubMedQA: replace the gold abstract with a similar study reaching a different conclusion (E10)")
    ap.add_argument("--cf-gold", action="store_true",
                    help="PubMedQA: replace the gold abstract with its counterfactual edit from make_cf.py (E11)")
    ap.add_argument("--samelabel-gold", action="store_true",
                    help="PubMedQA: replace the gold abstract with a similar study reaching the SAME conclusion (E11)")
    ap.add_argument("--start", type=int, default=0, help="skip the first START items of the shuffle (E11 subsets)")
    ap.add_argument("--tag", default="", help="suffix for the results file, so larger reruns do not overwrite")
    ap.add_argument("--scout", default="gpt-4o-mini", help="cascade: model that runs the probes and filter")
    args = ap.parse_args()
    if sum([args.no_gold, args.swap_gold, args.cf_gold, args.samelabel_gold]) > 1:
        ap.error("choose at most one of --no-gold/--swap-gold/--cf-gold/--samelabel-gold")
    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()
