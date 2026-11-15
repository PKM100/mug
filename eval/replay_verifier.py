"""E9: replay the independent support verifier on every question where a flip gate admitted context.

The verifier acts only after the flip gate has admitted evidence, and on rejection the gate returns the answer it
would have returned without evidence. Its effect on a finished run can therefore be computed exactly from the run's
log: we rebuild each admitted question's pool (same seed), hand the admitted passages and the final answer to the
verifier, and record the verdict. Analysis then swaps in the withheld answer wherever the verifier rejects.

  PYTHONPATH=../backend python replay_verifier.py --verifier claude-sonnet-5.5
  PYTHONPATH=../backend python replay_verifier.py --target naive   # verify every gpt-4.1 naive answer (no gate)

E11 (post hoc): --arms picks the tagged arms, --start/--n an item slice, --out-tag a separate output file, e.g.
  replay_verifier.py --target naive --tag e11 --arms _cf,_samelabel --start 150 --n 200
  replay_verifier.py --target naive --verifier gpt-4.1 --tag e10 --out-tag e10sub --start 150 --n 200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import zlib
from pathlib import Path

from mug.gates import Question, Runtime, verify_support  # noqa: E402
from mug.pipeline import Engine  # noqa: E402
from mug.trace import Trace  # noqa: E402
from run_eval import RESULTS, load_items, paper_pool  # noqa: E402

RUNS = ["scale_medqa_gpt4omini_d16_s1", "scale_pubmedqa_gpt4omini_d16_s1", "scale_pubmedqa_gpt4omini_d16_nogold_s1",
        "cascade_medqa_gpt41_d16_s1", "cascade_pubmedqa_gpt41_d16_s1", "cascade_pubmedqa_gpt41_d16_nogold_s1"]


async def main(args):
    engine = Engine()
    await engine.warm()
    assert engine.textbooks.ready, engine.textbooks.error
    prefix = "verify_naive_" if args.target == "naive" else "verify_"
    tag = f"_{args.tag}" if args.tag else ""
    otag = f"_{args.out_tag}" if args.out_tag else tag
    out_path = RESULTS / "verifier" / f"{prefix}{args.verifier.replace('.', '')}{otag}.jsonl"
    arms = [a if not a or a.startswith("_") else f"_{a}" for a in args.arms.split(",")]
    runs = ([f"{e}_pubmedqa_{m}_d16{arm}{tag}_s1" for e, m in (("scale", "gpt4omini"), ("cascade", "gpt41"))
             for arm in arms] if args.tag else RUNS)
    out_path.parent.mkdir(exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            if r.get("verdict") is not None:
                done.add((r["run"], r["qid"]))
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async def one(run, it, rec, gold, arm):
        async with sem:
            seed = args.seed + zlib.crc32(it["qid"].encode()) % 1000
            pool = paper_pool(engine, it, seed, 16, gold, arm=arm)
            by_id = {c["id"]: c for c in pool}
            kept = [by_id[i] for i in rec["kept_ids"] if i in by_id] or pool
            q = Question(it["question"], it["options"])
            rt = Runtime(engine.client, Trace(), "unused", "unused", verifier_model=args.verifier)
            v = await verify_support(rt, q, rec["pred"], kept)
            usage = [s for s in rt.trace.spans if s.kind == "llm"]
            row = {"run": run, "qid": rec["qid"], "verdict": v, "kept_gold": any(c.get("is_gold") for c in kept),
                   "prompt_tokens": sum(s.prompt_tokens for s in usage),
                   "completion_tokens": sum(s.completion_tokens for s in usage)}
            if v is None and row["completion_tokens"] == 0:
                return  # call failed (e.g. rate limit); leave for a later pass
            async with lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")

    tasks = []
    for run in runs:
        path = RESULTS / f"{run}.jsonl"
        if not path.exists():
            continue
        ds = run.split("_")[1]
        items = {it["qid"]: it for it in load_items(ds, args.n, args.seed, args.start)}
        gold = "nogold" not in run
        arm = next((a for a in ("swap", "cf", "samelabel") if f"_{a}_" in run), "gold")
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if "error" in rec or (run, rec["qid"]) in done or rec["qid"] not in items:
                continue
            if args.target == "naive":
                if not run.startswith("cascade") or rec["gate"] != "naive" or rec.get("variant"):
                    continue
            elif (rec["gate"] != "flip" or (run.startswith("cascade") and rec.get("variant") != "cascade")
                  or not rec["admitted"]):
                continue
            done.add((run, rec["qid"]))
            tasks.append(one(run, items[rec["qid"]], rec, gold, arm))
    print(f"verifying {len(tasks)} {args.target} answers with {args.verifier}")
    await asyncio.gather(*tasks)
    print("done", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", default="claude-sonnet-5.5")
    ap.add_argument("--target", choices=["flip", "naive"], default="flip")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="", help="replay the tagged PubMedQA runs (e.g. e10) instead of the n=150 runs")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--start", type=int, default=0, help="item slice [start, start+n) of the seed shuffle (E11)")
    ap.add_argument("--arms", default=",_nogold,_swap", help="tagged-run arm suffixes, comma-separated ('' = gold)")
    ap.add_argument("--out-tag", default="", help="output file tag if different from --tag (E11)")
    ap.add_argument("--concurrency", type=int, default=4)
    asyncio.run(main(ap.parse_args()))
