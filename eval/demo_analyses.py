"""Analyses for the MUG-Serve demo paper (round-1 revision), computed from existing result files only.

No model calls. Reads eval/results/{fidelity,live,open,sched}_*.jsonl and writes
eval/analysis/demo_analyses.json. Run: ../backend/.venv/bin/python demo_analyses.py
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))
from mug.audit import outcome  # noqa: E402  (pure function; imports no model client)

R = HERE / "results"
B = 4000
RNG = random.Random(0)


def load(name: str) -> list[dict]:
    return [r for r in (json.loads(l) for l in (R / f"{name}.jsonl").open()) if "error" not in r]


def by_gate(rows: list[dict]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for r in rows:
        out.setdefault(r["gate"] + (f":{r['variant']}" if r.get("variant") else ""), {})[r["qid"]] = r
    return out


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def paired(a: dict[str, dict], b: dict[str, dict], qids: list[str] | None = None) -> dict:
    """Accuracy difference a-b on shared questions, with a paired percentile bootstrap CI."""
    qs = sorted(set(a) & set(b)) if qids is None else [q for q in qids if q in a and q in b]
    x = [int(a[q]["correct"]) - int(b[q]["correct"]) for q in qs]
    n = len(x)
    bs = sorted(sum(x[RNG.randrange(n)] for _ in range(n)) / n for _ in range(B))
    bb, cc = sum(1 for v in x if v > 0), sum(1 for v in x if v < 0)
    return {"n": n, "diff": round(sum(x) / n, 3), "ci": [round(bs[int(0.025 * B)], 3), round(bs[int(0.975 * B) - 1], 3)],
            "b": bb, "c": cc, "p": round(mcnemar(bb, cc), 4)}


def acc(g: dict[str, dict], qids: list[str] | None = None) -> dict:
    qs = list(g) if qids is None else [q for q in qids if q in g]
    k = sum(1 for q in qs if g[q]["correct"])
    return {"n": len(qs), "acc": round(k / len(qs), 3), "wilson": wilson(k, len(qs))}


def ctx_frac(gate: dict[str, dict], naive: dict[str, dict]) -> float:
    kept = sum(r["retained_tokens"] for r in gate.values())
    full = sum(naive[q]["retained_tokens"] for q in gate if q in naive)
    return round(kept / full, 3) if full else float("nan")


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def pearson(x: list[float], y: list[float]) -> float:
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")


def corr_ci(x: list[float], y: list[float]) -> dict:
    n = len(x)
    bs = []
    for _ in range(2000):
        idx = [RNG.randrange(n) for _ in range(n)]
        r = pearson([x[i] for i in idx], [y[i] for i in idx])
        if not math.isnan(r):
            bs.append(r)
    bs.sort()
    return {"n": n, "r": round(pearson(x, y), 3),
            "ci": [round(bs[int(0.025 * len(bs))], 2), round(bs[int(0.975 * len(bs)) - 1], 2)]}


# ---------------------------------------------------------------- Table 1 (fidelity) + live + open
def accuracy_tables() -> dict:
    out = {}
    for run in ["fidelity_medqa_gpt4omini_s1", "fidelity_pubmedqa_gpt4omini_s1", "fidelity_medqa_claudehaiku45_s1",
                "fidelity_pubmedqa_claudehaiku45_s1", "live_medqa_gpt4omini_s3", "live_pubmedqa_gpt4omini_s3",
                "open_medqa_gpt4omini_s1", "open_pubmedqa_gpt4omini_s1"]:
        g = by_gate(load(run))
        cell = {"policies": {k: acc(v) | {"ctx": ctx_frac(v, g["naive"])} for k, v in g.items()}}
        cell["diffs"] = {f"{k}-{ref}": paired(g[k], g[ref]) for k in g if k not in ("naive", "closed_book")
                         for ref in ("naive", "closed_book")}
        cell["diffs"]["naive-closed_book"] = paired(g["naive"], g["closed_book"])
        if "flip" in g:
            f = list(g["flip"].values())
            cell["flip_moved"] = sum(1 for r in f if r["mode"] not in ("closed_book", "no_evidence"))
            cell["flip_calls_per_q"] = round(statistics.fmean(r["summary"]["llm_calls"] for r in f), 2)
            cell["naive_open_text_only"] = None
        out[run] = cell
    return out


def leakage() -> dict:
    """PubMedQA questions are paper titles; a live pool 'leaks' if it contains the question's own PMID."""
    g = by_gate(load("live_pubmedqa_gpt4omini_s3"))
    leaked = sorted(q for q, r in g["naive"].items() if f"PMID:{q.split(':', 1)[1]}" in r.get("pool_ids", []))
    clean = sorted(q for q in g["naive"] if q not in set(leaked))
    res = {"n": len(g["naive"]), "n_leaked": len(leaked), "n_clean": len(clean), "method":
           "naive-policy pool_ids contains PMID:<question PMID>"}
    for name, qs in (("leaked", leaked), ("clean", clean)):
        res[name] = {k: acc(v, qs) for k, v in g.items()}
        res[name + "_diffs"] = {f"{k}-closed_book": paired(g[k], g["closed_book"], qs) for k in g if k != "closed_book"}
    # do the flip pools carry the same leaks? (pools are rebuilt per policy, PubMed is cached across them)
    res["flip_pool_leaks"] = sum(1 for q, r in g["flip"].items() if f"PMID:{q.split(':', 1)[1]}" in r.get("pool_ids", []))
    return res


def correlations() -> dict:
    """Movement vs the signed context effect (ctx correct - closed-book correct) for the flip records."""
    out = {}
    for run in ["open_medqa_gpt4omini_s1", "open_pubmedqa_gpt4omini_s1", "fidelity_medqa_gpt4omini_s1",
                "fidelity_pubmedqa_gpt4omini_s1"]:
        g = by_gate(load(run))
        gate = "flip" if run.startswith("open") else "logit_shift"
        rs = [r for r in g[gate].values() if r.get("movement") is not None and r.get("ctx_pred") is not None]
        x = [float(r["movement"]) for r in rs]
        y = [float(r["ctx_pred"] == r["gold"]) - float(r["cb_pred"] == r["gold"]) for r in rs]
        out[f"{run}:{gate}"] = corr_ci(x, y)
    return out


# ---------------------------------------------------------------- audit-proxy label replay
def audit_labels() -> dict:
    """The proxy calls Engine.ask(gate='flip', pool=documents) with default scheduling; these flip records used the
    same code path, so outcome() reproduces the label the proxy would have logged (no verifier configured)."""
    out = {}
    for run in ["fidelity_medqa_gpt4omini_s1", "fidelity_pubmedqa_gpt4omini_s1", "fidelity_medqa_claudehaiku45_s1",
                "fidelity_pubmedqa_claudehaiku45_s1", "live_medqa_gpt4omini_s3", "live_pubmedqa_gpt4omini_s3",
                "open_medqa_gpt4omini_s1", "open_pubmedqa_gpt4omini_s1"]:
        f = [r for r in load(run) if r["gate"] == "flip"]
        c = Counter(outcome(r, r.get("spans")) for r in f)
        adm = [r for r in f if outcome(r, r.get("spans")) == "admitted"]
        cb_ok = lambda r: r["cb_pred"] == r["gold"]  # noqa: E731
        ok = lambda r: r["pred"] == r["gold"]  # noqa: E731
        out[run] = {"n": len(f), "labels": dict(c),
                    "admitted_fixed": sum(1 for r in adm if not cb_ok(r) and ok(r)),
                    "admitted_broke": sum(1 for r in adm if cb_ok(r) and not ok(r)),
                    "admitted_both_right": sum(1 for r in adm if cb_ok(r) and ok(r)),
                    "admitted_both_wrong": sum(1 for r in adm if not cb_ok(r) and not ok(r)),
                    "no_effect_correct": sum(1 for r in f if outcome(r) == "no_effect" and ok(r))}
    return out


# ---------------------------------------------------------------- Table 2 (scheduling) and baselines
def scheduling() -> dict:
    out = {}
    for ds in ("medqa", "pubmedqa"):
        g = by_gate(load(f"sched_{ds}_gpt4omini_s2"))
        ser = g["flip:serial"]
        res = {}
        for v in ("serial", "parallel", "overlap", "overlap+spec"):
            gv = g[f"flip:{v}"]
            w = [r["summary"]["wall_ms"] / 1000 for r in gv.values()]
            res[v] = {"n": len(w), "p50": round(pct(w, .5), 2), "p95": round(pct(w, .95), 2),
                      "calls_per_q": round(statistics.fmean(r["summary"]["llm_calls"] for r in gv.values()), 2),
                      "faster_than_serial": round(sum(1 for q, r in gv.items()
                                                      if r["summary"]["wall_ms"] < ser[q]["summary"]["wall_ms"]) / len(gv), 3)}
        # Baselines reconstructed from the serial trace of the same request (not separately measured runs):
        # closed-book = duration of the closed-book probe (identical prompt to the closed-book policy);
        # naive = time the pool became ready + duration of the full-pool probe (identical to the naive answer call).
        cb, nv = [], []
        for r in ser.values():
            sp = {s["name"]: s for s in r["spans"]}
            p_cb, p_full = sp.get("probe(closed-book)"), sp.get("probe(full pool)")
            if p_cb:
                cb.append((p_cb["end_ms"] - p_cb["start_ms"]) / 1000)
            if p_full:
                ready = max((s["end_ms"] for s in r["spans"] if s["kind"] == "retrieval"), default=p_full["start_ms"])
                nv.append((ready + p_full["end_ms"] - p_full["start_ms"]) / 1000)
        res["closed_book_reconstructed"] = {"n": len(cb), "p50": round(pct(cb, .5), 2), "p95": round(pct(cb, .95), 2)}
        res["naive_reconstructed"] = {"n": len(nv), "p50": round(pct(nv, .5), 2), "p95": round(pct(nv, .95), 2)}
        live = by_gate(load(f"live_{ds}_gpt4omini_s3"))
        for k in ("closed_book", "naive"):
            w = [r["summary"]["wall_ms"] / 1000 for r in live[k].values()]
            res[f"{k}_live_E4"] = {"n": len(w), "p50": round(pct(w, .5), 2), "p95": round(pct(w, .95), 2),
                                   "tok_per_q": round(statistics.fmean(r["summary"]["prompt_tokens"]
                                                                       for r in live[k].values()))}
        res["question_overlap_with_E4"] = len(set(ser) & set(live["naive"]))
        if (R / f"live_{ds}_gpt4omini_latbase_s2.jsonl").exists():
            lb = by_gate(load(f"live_{ds}_gpt4omini_latbase_s2"))
            for k in ("closed_book", "naive"):
                w = [r["summary"]["wall_ms"] / 1000 for r in lb[k].values()]
                res[f"{k}_measured"] = {"n": len(w), "p50": round(pct(w, .5), 2), "p95": round(pct(w, .95), 2),
                                        "tok_per_q": round(statistics.fmean(r["summary"]["prompt_tokens"]
                                                                            for r in lb[k].values())),
                                        "same_questions_as_sched": len(set(lb[k]) & set(ser))}
        out[ds] = res
    return out


def main() -> None:
    res = {"accuracy": accuracy_tables(), "leakage": leakage(), "correlations": correlations(),
           "audit_labels": audit_labels(), "scheduling": scheduling()}
    p = HERE / "analysis" / "demo_analyses.json"
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
