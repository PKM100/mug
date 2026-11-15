"""Aggregate eval/results/*.jsonl into paper tables (LaTeX), figures and summary.json."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)

GATE_ORDER = ["closed_book", "naive", "flip", "logit_shift", "grounded"]
GATE_TEX = {"closed_book": "Closed-book", "naive": "Naive (full pool)", "flip": "MUG flip+filter",
            "logit_shift": "MUG logit-shift", "grounded": "MUG grounded"}
FIXED = ("closed_book", "naive")


def load() -> list[dict]:
    rows = []
    for f in sorted(RES.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                rows.append(r)
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(a: dict, b: dict) -> tuple[int, int, float]:
    """Exact two-sided McNemar on qids answered under both. b=a right/b wrong."""
    common = set(a) & set(b)
    bb = sum(1 for q in common if a[q] and not b[q])
    cc = sum(1 for q in common if b[q] and not a[q])
    n = bb + cc
    if n == 0:
        return bb, cc, 1.0
    p = sum(math.comb(n, i) for i in range(0, min(bb, cc) + 1)) / 2 ** n
    return bb, cc, min(1.0, 2 * p)


def cluster_diff(a: dict, b: dict, qs, halves=(True, False), n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired accuracy difference a-b when every question is asked once per half. The question is the unit:
    per-question mean-over-halves differences, a percentile bootstrap CI over questions, and a sign-flip
    permutation p-value (exact over the nonzero questions when there are at most 20 of them)."""
    qs = sorted(qs)
    d = np.array([np.mean([int(a[(g, q)]) - int(b[(g, q)]) for g in halves]) for q in qs])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    nz = d[d != 0]
    obs = abs(d.sum())
    if len(nz) == 0:
        p_ = 1.0
    elif len(nz) <= 20:
        import itertools
        sums = [abs(sum(s * x for s, x in zip(signs, nz))) for signs in itertools.product((1, -1), repeat=len(nz))]
        p_ = float(np.mean([s_ >= obs - 1e-12 for s_ in sums]))
    else:
        flips = rng.choice((1, -1), (n_boot, len(nz)))
        p_ = float((np.sum(np.abs((flips * nz).sum(axis=1)) >= obs - 1e-12) + 1) / (n_boot + 1))
    return {"n": len(qs), "diff": float(d.mean()), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p": p_, "q_better": int((d > 0).sum()), "q_worse": int((d < 0).sum())}


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


def group(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["exp"], r["dataset"], r["model"], r["gate"], r.get("variant", ""))].append(r)
    return g


def gate_stats(rs: list[dict]) -> dict:
    n = len(rs)
    k = sum(r["correct"] for r in rs)
    lo, hi = wilson(k, n)
    walls = [r["summary"]["wall_ms"] for r in rs]
    pool = np.mean([r["pool_tokens"] for r in rs]) if rs else 0
    return {
        "n": n, "acc": k / n if n else float("nan"), "ci": [lo, hi],
        "admit_rate": float(np.mean([r["admitted"] for r in rs])),
        "retained_tokens": float(np.mean([r["retained_tokens"] for r in rs])),
        "pool_tokens": float(pool),
        "calls": float(np.mean([r["summary"]["llm_calls"] for r in rs])),
        "prompt_tokens": float(np.mean([r["summary"]["prompt_tokens"] for r in rs])),
        "wall_p50": pct(walls, 50), "wall_p95": pct(walls, 95),
        "seq_p50": pct([r["summary"]["sequential_ms"] for r in rs], 50),
        "wasted_tokens": float(np.mean([r["summary"].get("wasted_prompt_tokens", 0) for r in rs])),
        "none_rate": float(np.mean([r["pred"] is None for r in rs])),
    }


def accuracy_table(g, exp: str, model: str, datasets=("medqa", "pubmedqa"), gates=GATE_ORDER,
                   latency: bool = True) -> tuple[str, dict]:
    # live runs share a PubMed cache across gates (same pool for every gate), so their wall times are not comparable
    ncol = 8 if latency else 7
    summ: dict = {}
    lines = []
    for ds in datasets:
        cells = {gt: g.get((exp, ds, model, gt, ""), []) for gt in gates}
        if not any(cells.values()):
            continue
        corr = {gt: {r["qid"]: r["correct"] for r in rs} for gt, rs in cells.items()}
        fixed_acc = {f: (np.mean(list(corr[f].values())) if corr.get(f) else -1) for f in FIXED}
        best_fixed = max(fixed_acc, key=fixed_acc.get)
        summ[ds] = {"best_fixed": best_fixed}
        lines.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{{'MedQA' if ds == 'medqa' else 'PubMedQA'}}}}} \\")
        naive_tok = None
        for gt in gates:
            rs = cells[gt]
            if not rs:
                continue
            s = gate_stats(rs)
            if gt == "naive":
                naive_tok = s["retained_tokens"]
            b, c, p = mcnemar(corr[gt], corr[best_fixed]) if gt != best_fixed else (0, 0, float("nan"))
            s.update(vs_best_fixed={"b": b, "c": c, "p": p})
            summ[ds][gt] = s
        for gt in gates:
            if gt not in summ[ds]:
                continue
            s = summ[ds][gt]
            ret = s["retained_tokens"] / naive_tok * 100 if naive_tok else float("nan")
            s["retained_pct_of_naive"] = ret
            vb = s["vs_best_fixed"]
            ptxt = "--" if gt == best_fixed else (f"{vb['p']:.2f}" if vb["p"] >= 0.01 else f"{vb['p']:.0e}")
            dtxt = "--" if gt == best_fixed else f"{vb['b']}/{vb['c']}"
            lines.append(f"{GATE_TEX[gt]} & {s['acc']:.3f} & [{s['ci'][0]:.2f},{s['ci'][1]:.2f}] & {dtxt} & {ptxt} & "
                         f"{s['retained_tokens']:.0f} ({ret:.0f}\\%) & {s['calls']:.2f}"
                         + (f" & {s['wall_p50'] / 1000:.2f}" if latency else "") + " \\\\")
        lines.append(r"\midrule")
    if lines:
        lines.pop()
    body = "\n".join(lines)
    tex = (r"\begin{tabular}{l" + "c" * (ncol - 1) + "}\n" r"\toprule" "\n"
           r"Policy & Acc. & 95\% CI & $b/c$ & $p$ & Retained ctx (tok, \% naive) & Calls/q"
           + (r" & p50 (s)" if latency else "") + r" \\" "\n"
           r"\midrule" "\n" + body + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    return tex, summ


def signed_effect_analysis(rows, exp="open", gate="flip", thresh=0.5):
    """Paper's core finding, re-tested for open-ended answers: does semantic movement
    predict *help* (evidence-hungry) or *harm* (distractor-prone)?"""
    out = {}
    for ds in ("medqa", "pubmedqa"):
        rs = [r for r in rows if r["exp"] == exp and r["dataset"] == ds and r["gate"] == gate
              and r["cb_pred"] is not None and r["ctx_pred"] is not None and r["movement"] is not None]
        if len(rs) < 10:
            continue
        m = np.array([r["movement"] for r in rs])
        eff = np.array([(r["ctx_pred"] == r["gold"]) - (r["cb_pred"] == r["gold"]) for r in rs], float)
        rho = float(np.corrcoef(m, eff)[0, 1]) if m.std() > 0 and eff.std() > 0 else float("nan")
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(2000):
            i = rng.integers(0, len(rs), len(rs))
            if m[i].std() > 0 and eff[i].std() > 0:
                boots.append(np.corrcoef(m[i], eff[i])[0, 1])
        moved = m >= thresh
        out[ds] = {"n": len(rs), "r": rho, "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                   "moved_frac": float(moved.mean()),
                   "effect_when_moved": float(eff[moved].mean()) if moved.any() else float("nan"),
                   "effect_when_still": float(eff[~moved].mean()) if (~moved).any() else float("nan")}
    return out


def sched_analysis(rows):
    out = {}
    for ds in ("medqa", "pubmedqa"):
        rs = [r for r in rows if r["exp"] == "sched" and r["dataset"] == ds]
        if not rs:
            continue
        by = defaultdict(dict)
        for r in rs:
            by[r["variant"]][r["qid"]] = r
        common = set.intersection(*[set(v) for v in by.values()]) if by else set()
        d = {"n_paired": len(common)}
        for v, m in by.items():
            walls = [m[q]["summary"]["wall_ms"] for q in common]
            moved = [m[q]["admitted"] for q in common]
            d[v] = {"p50": pct(walls, 50), "p95": pct(walls, 95), "mean": float(np.mean(walls)),
                    "prompt_tokens": float(np.mean([m[q]["summary"]["prompt_tokens"] for q in common])),
                    "wasted_tokens": float(np.mean([m[q]["summary"].get("wasted_prompt_tokens", 0) for q in common])),
                    "admit_rate": float(np.mean(moved)), "walls": walls,
                    "retrieval_ms": float(np.mean([max((s["end_ms"] for s in m[q]["spans"] if s["kind"] == "retrieval"),
                                                       default=0) for q in common]))}
        base = "serial" if "serial" in d else "sequential"
        if base in d:
            ref = np.array(d[base]["walls"])
            rng = np.random.default_rng(0)
            moved = np.array([by[base][q]["admitted"] for q in common]) if common else np.array([])
            for v in [k for k in by if k != base]:
                w = np.array(d[v]["walls"])
                ratio = w / ref
                boots = [np.median(rng.choice(ratio, len(ratio))) for _ in range(2000)]
                d[v]["paired_median_speedup_ms"] = float(np.median(ref - w))
                d[v]["paired_median_ratio"] = float(np.median(ratio))
                d[v]["ratio_ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
                d[v]["win_rate"] = float(np.mean(w < ref))
            # latency split by whether the answer moved (in any schedule's own run)
            for v in by:
                mv = np.array([by[v][q]["admitted"] for q in common])
                w = np.array(d[v]["walls"])
                d[v]["p50_moved"] = pct(list(w[mv]), 50)
                d[v]["p50_still"] = pct(list(w[~mv]), 50)
                out[ds] = d
    return out


def leakage(rows):
    """Live PubMed retrieval can surface the very paper a PubMedQA question was built from."""
    rs = [r for r in rows if r["exp"] == "live" and r["dataset"] == "pubmedqa"]
    if not rs:
        return {}
    leaked = {r["qid"] for r in rs if r["gate"] == "naive" and f"PMID:{r['qid'].split(':')[1]}" in r["pool_ids"]}
    qids = {r["qid"] for r in rs if r["gate"] == "naive"}
    out = {"n": len(qids), "source_paper_retrieved": len(leaked) / max(1, len(qids)), "by_gate": {}}
    for g in GATE_ORDER:
        gr = [r for r in rs if r["gate"] == g]
        if not gr:
            continue
        a = [r["correct"] for r in gr if r["qid"] in leaked]
        b = [r["correct"] for r in gr if r["qid"] not in leaked]
        out["by_gate"][g] = {"acc_leaked": float(np.mean(a)) if a else None, "n_leaked": len(a),
                             "acc_clean": float(np.mean(b)) if b else None, "n_clean": len(b)}
    # McNemar on the clean subset vs closed-book
    cb = {r["qid"]: r["correct"] for r in rs if r["gate"] == "closed_book" and r["qid"] not in leaked}
    for g in ("naive", "flip", "grounded"):
        gm = {r["qid"]: r["correct"] for r in rs if r["gate"] == g and r["qid"] not in leaked}
        if gm and g in out["by_gate"]:
            out["by_gate"][g]["clean_vs_cb"] = dict(zip(("b", "c", "p"), mcnemar(gm, cb)))
    return out


def figures(rows, sched, summaries):
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    # Latency CDFs
    if sched:
        fig, axes = plt.subplots(1, len(sched), figsize=(3.4 * len(sched), 2.6), squeeze=False)
        for ax, (ds, d) in zip(axes[0], sched.items()):
            for v, ls in (("serial", "-"), ("parallel", "-."), ("overlap", "--"), ("overlap+spec", ":")):
                if v in d:
                    w = np.sort(np.array(d[v]["walls"]) / 1000)
                    ax.plot(w, np.arange(1, len(w) + 1) / len(w), ls, label=v)
            ax.set_title({"medqa": "MedQA", "pubmedqa": "PubMedQA"}[ds] + " (live hybrid retrieval)")
            ax.set_xlabel("end-to-end latency (s)")
            ax.set_ylabel("CDF")
            ax.grid(alpha=.3)
            ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUT / "latency_cdf.pdf")
        fig.savefig(OUT / "latency_cdf.png", dpi=200)
        plt.close(fig)
    # Accuracy vs retained context
    panels = [(k, v) for k, v in summaries.items() if v]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 2.8), squeeze=False)
        for ax, (title, summ) in zip(axes[0], panels):
            for ds, mk in (("medqa", "o"), ("pubmedqa", "s")):
                for gt in GATE_ORDER:
                    s = summ.get(ds, {}).get(gt)
                    if not s:
                        continue
                    ax.errorbar(s["retained_tokens"], s["acc"], yerr=[[max(0.0, s["acc"] - s["ci"][0])], [max(0.0, s["ci"][1] - s["acc"])]],
                                fmt=mk, ms=5, capsize=2, color=f"C{GATE_ORDER.index(gt)}",
                                label=GATE_TEX[gt] if ds == "medqa" else None)
            ax.set_xlabel("retained final-answer context (tokens)")
            ax.set_ylabel("accuracy")
            ax.set_title(title)
            ax.grid(alpha=.3)
        axes[0][0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "acc_vs_context.pdf")
        fig.savefig(OUT / "acc_vs_context.png", dpi=200)
        plt.close(fig)


# ------------------------------------------------------------------ where gating adds value (E5-E7)
# OpenAI list prices, USD per 1M tokens (input, output).
# Claude prices are assumed list prices (the Copilot API does not bill per token); stated in the paper.
PRICES = {"gpt-4.1": (2.00, 8.00), "gpt-4o-mini": (0.15, 0.60), "claude-sonnet-5.5": (3.00, 15.00),
          "claude-haiku-5.5": (1.00, 5.00)}
# verifier model -> variant suffix
VERIFIERS = {"claude-sonnet-5.5": "+verify", "claude-haiku-5.5": "+verifyH", "gpt-4.1": "+verifyG"}


def naive_verified_rows(rows, verifier: str = "claude-sonnet-5.5", disagree_only: bool = False) -> list[dict]:
    """Advisor's control: no gate. gpt-4.1 always reads the pool, the verifier checks every answer, and on rejection
    gpt-4.1's closed-book answer is returned. The closed-book call is charged on every question (it would be issued
    in parallel, as in the cascade). With disagree_only, the verifier runs only where the evidence-reading and
    closed-book answers differ: elsewhere a rejection would return the same answer, so accuracy is unchanged."""
    vf = RES / "verifier" / f"verify_naive_{verifier.replace('.', '')}.jsonl"
    if not vf.exists():
        return []
    ver = {}
    for line in vf.read_text().splitlines():
        v = json.loads(line)
        if v.get("verdict") is not None or v.get("prompt_tokens"):
            ver[(v["run"], v["qid"])] = v
    cb = {(r["dataset"], r["qid"]): r for r in rows
          if r.get("exp") == "cascade" and r["gate"] == "closed_book" and r.get("model") == "gpt-4.1" and "error" not in r}
    out = []
    for r in rows:
        if (r.get("exp") != "cascade" or r.get("gate") != "naive" or r.get("variant") or r.get("model") != "gpt-4.1"
                or r.get("n_distract") != 16 or "error" in r):
            continue
        run = f"cascade_{r['dataset']}_gpt41_d16{'' if r.get('gold_in_pool', True) else '_nogold'}_s1"
        v, c = ver.get((run, r["qid"])), cb.get((r["dataset"], r["qid"]))
        if v is None or c is None:
            continue
        n = dict(r, variant="+vdis" if disagree_only else "+vnaive",
                 spans=list(r.get("spans", [])) + list(c.get("spans", [])))
        if disagree_only and r["pred"] == c["pred"]:
            out.append(n)
            continue
        n["spans"].append({"kind": "llm", "model": verifier, "prompt_tokens": v["prompt_tokens"],
                           "completion_tokens": v["completion_tokens"], "name": "verify(support)"})
        if not (v["verdict"] or {}).get("supported"):
            n.update(pred=c["pred"], admitted=False, retained_tokens=0, correct=c["pred"] == r["gold"])
        out.append(n)
    return out


def verified_rows(rows, verifier: str = "claude-sonnet-5.5") -> list[dict]:
    """E9: apply the replayed support verifier (replay_verifier.py) to admitted flip rows. On rejection the gate
    returns its withheld answer: the probe's closed-book answer, or in a cascade the answer model's closed-book
    answer (taken from the paired closed-book row; same prompt, temperature 0). Verifier tokens are charged. A
    verifier that returns nothing parseable counts as a rejection, as in the live gate."""
    suffix = VERIFIERS[verifier]
    vf = RES / "verifier" / f"verify_{verifier.replace('.', '')}.jsonl"
    if not vf.exists():
        return []
    ver = {}
    for line in vf.read_text().splitlines():
        v = json.loads(line)
        if v.get("verdict") is not None or v.get("prompt_tokens"):
            ver[(v["run"], v["qid"])] = v
    big_cb = {(r["dataset"], r["qid"]): r["pred"] for r in rows
              if r.get("exp") == "cascade" and r["gate"] == "closed_book" and r.get("model") == "gpt-4.1"}
    out = []
    for r in rows:
        if r.get("gate") != "flip" or r.get("exp") not in ("scale", "cascade") or r.get("n_distract") != 16:
            continue
        casc = r["exp"] == "cascade"
        if casc != (r.get("variant") == "cascade"):
            continue
        run = (f"{r['exp']}_{r['dataset']}_{r['model'].replace('.', '').replace('-', '')}_d16"
               f"{'' if r.get('gold_in_pool', True) else '_nogold'}_s1")
        n = dict(r, variant=(r.get("variant") or "") + suffix, spans=list(r.get("spans", [])))
        if r["admitted"]:
            v = ver.get((run, r["qid"]))
            if v is None:
                continue  # not yet verified; leave the policy incomplete rather than guess
            n["spans"].append({"kind": "llm", "model": verifier, "prompt_tokens": v["prompt_tokens"],
                               "completion_tokens": v["completion_tokens"], "name": "verify(support)"})
            if not (v["verdict"] or {}).get("supported"):
                n["pred"] = big_cb.get((r["dataset"], r["qid"])) if casc else r["cb_pred"]
                if n["pred"] is None:
                    continue
                n.update(admitted=False, retained_tokens=0, mode="unverified_keep", correct=n["pred"] == r["gold"])
        out.append(n)
    return out


def q_cost(r: dict) -> dict:
    """Per-question tokens and dollars, split by model, from the span log (speculative calls included)."""
    out = defaultdict(float)
    for s in r.get("spans", []):
        m = s.get("model")
        if s.get("kind") != "llm" or not m:
            continue
        pin, pout = PRICES.get(m, (0.0, 0.0))
        out[f"in:{m}"] += s.get("prompt_tokens", 0)
        out["usd"] += (s.get("prompt_tokens", 0) * pin + s.get("completion_tokens", 0) * pout) / 1e6
    return out


def mixed_workload(g) -> dict:
    """E5: pool MedQA and PubMedQA into one workload, as a deployment would see it."""
    res = {}
    for exp, model in [("fidelity", "gpt-4o-mini"), ("fidelity", "claude-haiku-4.5"),
                       ("open", "gpt-4o-mini"), ("live", "gpt-4o-mini")]:
        corr = {}
        for gt in GATE_ORDER:
            c = {(d, r["qid"]): r["correct"] for d in ("medqa", "pubmedqa") for r in g.get((exp, d, model, gt, ""), [])}
            if c:
                corr[gt] = c
        if "naive" not in corr:
            continue
        cell = {}
        for gt, c in corr.items():
            e = {"acc": float(np.mean(list(c.values()))), "n": len(c)}
            for f in ("closed_book", "naive"):
                if gt != f:
                    b, cc, p = mcnemar(c, corr[f])
                    e[f"vs_{f}"] = {"b": b, "c": cc, "p": p}
            cell[gt] = e
        res[f"{exp}/{model}"] = cell
    return res


def by_distract(rows, exp: str, model: str, gold: bool = True) -> dict:
    g = defaultdict(list)
    for r in rows:
        if (r.get("exp") == exp and r.get("model") == model and "error" not in r
                and r.get("gold_in_pool", True) == gold):
            g[(r["dataset"], r.get("n_distract", 4), r["gate"], r.get("variant", ""))].append(r)
    return g


def pool_scaling(rows) -> tuple[str, dict]:
    """E6: accuracy and retained context as the paper-protocol pool grows (4 -> 16 -> 32 added passages)."""
    sc = by_distract(rows, "scale", "gpt-4o-mini")
    fid = by_distract(rows, "fidelity", "gpt-4o-mini")
    out: dict = {}
    lines = []
    for ds in ("medqa", "pubmedqa"):
        qids = {r["qid"] for (d, _, _, _), rs in sc.items() if d == ds for r in rs}
        if not qids:
            continue
        lines.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{'MedQA' if ds == 'medqa' else 'PubMedQA'}}}}} \\")
        for nd in (4, 16, 32):
            src = fid if nd == 4 else sc
            cells = {gt: {r["qid"]: r for r in src.get((ds, nd, gt, ""), []) if r["qid"] in qids}
                     for gt in ("closed_book", "naive", "flip", "grounded")}
            if not all(cells.values()):
                continue
            common = set.intersection(*(set(c) for c in cells.values()))
            corr = {gt: {q: c[q]["correct"] for q in common} for gt, c in cells.items()}
            acc = {gt: float(np.mean(list(v.values()))) for gt, v in corr.items()}
            pool_tok = float(np.mean([cells["naive"][q]["pool_tokens"] for q in common]))
            ret = {gt: float(np.mean([cells[gt][q]["retained_tokens"] for q in common])) for gt in ("flip", "grounded")}
            tests = {gt: dict(zip("bcp", mcnemar(corr[gt], corr["naive"]))) for gt in ("closed_book", "flip", "grounded")}
            out.setdefault(ds, {})[nd] = {"n": len(common), "acc": acc, "pool_tokens": pool_tok, "retained": ret,
                                          "vs_naive": tests}
            def pv(p):
                return f"{p:.2f}" if p >= 0.01 else f"{p:.0e}"
            lines.append(
                f"{nd + 1} & {pool_tok:.0f} & {acc['closed_book']:.3f} & {acc['naive']:.3f} & "
                f"{acc['flip']:.3f} ({tests['flip']['b']}/{tests['flip']['c']}, {pv(tests['flip']['p'])}) & "
                f"{acc['grounded']:.3f} ({tests['grounded']['b']}/{tests['grounded']['c']}, {pv(tests['grounded']['p'])}) & "
                f"{ret['flip'] / pool_tok * 100:.0f}\\% & {ret['grounded'] / pool_tok * 100:.0f}\\% \\\\")
        lines.append(r"\midrule")
    if lines:
        lines.pop()
    tex = (r"\begin{tabular}{rrcccccc}" "\n" r"\toprule" "\n"
           r"Pool & Pool tok & Closed-book & Naive & Flip ($b/c$, $p$) & Grounded ($b/c$, $p$) & "
           r"Flip ctx & Grnd ctx \\" "\n" r"\midrule" "\n" + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    return tex, out


CASCADE_ROWS = [("closed_book", "", "gpt-4.1", "gpt-4.1 closed-book"),
                ("naive", "", "gpt-4.1", "gpt-4.1 naive"),
                ("flip", "cascade", "gpt-4.1", "Cascade flip (4o-mini scout)"),
                ("grounded", "cascade", "gpt-4.1", "Cascade grounded (4o-mini scout)"),
                ("naive", "", "gpt-4o-mini", "gpt-4o-mini naive"),
                ("flip", "", "gpt-4o-mini", "gpt-4o-mini flip"),
                ("flip", "cascade+verify", "gpt-4.1", "Cascade flip + verifier"),
                ("flip", "+verify", "gpt-4o-mini", "gpt-4o-mini flip + verifier"),
                ("flip", "+verifyH", "gpt-4o-mini", "gpt-4o-mini flip + Haiku verifier"),
                ("flip", "+verifyG", "gpt-4o-mini", "gpt-4o-mini flip + gpt-4.1 verifier"),
                ("naive", "+vnaive", "gpt-4.1", "gpt-4.1 naive + verifier"),
                ("naive", "+vdis", "gpt-4.1", "gpt-4.1 naive + verifier (disagreements)")]
ROUTER = "Format router (oracle per dataset)"


def paired_diff(a: dict, b: dict, z: float = 1.96) -> dict:
    """Paired accuracy difference a-b over common items, with a Wald CI on the per-item difference."""
    common = set(a) & set(b)
    n = len(common)
    d = [int(a[q]) - int(b[q]) for q in common]
    m = sum(d) / n
    se = math.sqrt(max(sum(x * x for x in d) / n - m * m, 0) / n)
    return {"diff": m, "lo": m - z * se, "hi": m + z * se, "n": n}


def cascade(rows) -> tuple[str, dict]:
    """E7: an expensive answer model whose gate is run by a cheap scout. Accuracy vs. dollars."""
    cg = by_distract(rows, "cascade", "gpt-4.1")
    pooled: dict = {}
    sg = by_distract(rows, "scale", "gpt-4o-mini")
    out: dict = {}
    lines = []
    for ds in ("medqa", "pubmedqa"):
        cells = {}
        for gt, var, model, label in CASCADE_ROWS:
            src = cg if model == "gpt-4.1" else sg
            cells[label] = {r["qid"]: r for r in src.get((ds, 16, gt, var), [])}
        if not cells["gpt-4.1 naive"]:
            continue
        common = set.intersection(*(set(c) for c in cells.values() if c))
        ref = {q: cells["gpt-4.1 naive"][q]["correct"] for q in common}
        ref_usd = np.mean([q_cost(cells["gpt-4.1 naive"][q])["usd"] for q in common])
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{'MedQA' if ds == 'medqa' else 'PubMedQA'}}}, $n{{=}}{len(common)}$, pool $\approx${np.mean([cells['gpt-4.1 naive'][q]['pool_tokens'] for q in common]):.0f} tok}} \\")
        for _, _, _, label in CASCADE_ROWS:
            c = cells[label]
            if not c:
                continue
            corr = {q: c[q]["correct"] for q in common}
            k = sum(corr.values())
            lo, hi = wilson(k, len(common))
            costs = [q_cost(c[q]) for q in common]
            usd = float(np.mean([x["usd"] for x in costs]))
            big_in = float(np.mean([x.get("in:gpt-4.1", 0) for x in costs]))
            b, cc, p = mcnemar(corr, ref) if label != "gpt-4.1 naive" else (0, 0, float("nan"))
            ret = float(np.mean([c[q]["retained_tokens"] for q in common]))
            pooled.setdefault(label, {"corr": {}, "usd": {}})
            pooled[label]["corr"].update({(ds, q): v for q, v in corr.items()})
            pooled[label]["usd"].update({(ds, q): x["usd"] for q, x in zip(common, costs)})
            out.setdefault(ds, {})[label] = {"acc": k / len(common), "ci": [lo, hi], "usd_per_1k": usd * 1000,
                                             "cost_vs_naive": usd / ref_usd, "big_model_in_tok": big_in,
                                             "retained_tokens": ret, "vs_41_naive": {"b": b, "c": cc, "p": p}}
            ptxt = "--" if label == "gpt-4.1 naive" else (f"{b}/{cc}, {p:.2f}" if p >= 0.01 else f"{b}/{cc}, {p:.0e}")
            if "verifier" not in label:
                lines.append(f"{label} & {k / len(common):.3f} & [{lo:.2f},{hi:.2f}] & {ptxt} & {big_in:.0f} & "
                             f"\\${usd * 1000:.2f} & {usd / ref_usd:.2f} \\\\")
        lines.append(r"\midrule")
    if len({ds for ds, _ in pooled.get("gpt-4.1 naive", {"corr": {}})["corr"]}) > 1:
        # Mixed workload: a deployment doesn't know which dataset a question resembles.
        # A question-only router: MedQA (4 options) and PubMedQA (yes/no/maybe) are told apart by format alone,
        # so it can send each to that dataset's cheapest near-best fixed policy (chosen with hindsight).
        route = {"medqa": "gpt-4.1 closed-book", "pubmedqa": "gpt-4o-mini naive"}
        pooled[ROUTER] = {"corr": {k: v for lab_ in set(route.values()) for k, v in pooled[lab_]["corr"].items()
                                   if route[k[0]] == lab_},
                          "usd": {k: v for lab_ in set(route.values()) for k, v in pooled[lab_]["usd"].items()
                                  if route[k[0]] == lab_}}
        ref = pooled["gpt-4.1 naive"]
        ref_usd = float(np.mean(list(ref["usd"].values())))
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{Mixed (both pooled)}}, $n{{=}}{len(ref['corr'])}$}} \\")
        for label in [r[3] for r in CASCADE_ROWS] + [ROUTER]:
            if label not in pooled:
                continue
            corr = pooled[label]["corr"]
            k, n = sum(corr.values()), len(corr)
            lo, hi = wilson(k, n)
            usd = float(np.mean(list(pooled[label]["usd"].values())))
            b, cc, p = mcnemar(corr, ref["corr"]) if label != "gpt-4.1 naive" else (0, 0, float("nan"))
            vs_cb = mcnemar(corr, pooled["gpt-4.1 closed-book"]["corr"]) if label != "gpt-4.1 closed-book" else None
            out.setdefault("mixed", {})[label] = {"acc": k / n, "ci": [lo, hi], "usd_per_1k": usd * 1000,
                                                  "cost_vs_naive": usd / ref_usd,
                                                  "vs_41_naive": {"b": b, "c": cc, "p": p},
                                                  "vs_41_cb": dict(zip("bcp", vs_cb)) if vs_cb else None,
                                                  "diff_vs_41_naive": paired_diff(corr, ref["corr"])}
            ptxt = "--" if label == "gpt-4.1 naive" else (f"{b}/{cc}, {p:.2f}" if p >= 0.01 else f"{b}/{cc}, {p:.0e}")
            if "verifier" not in label:
                lines.append(f"{label} & {k / n:.3f} & [{lo:.2f},{hi:.2f}] & {ptxt} & -- & "
                             f"\\${usd * 1000:.2f} & {usd / ref_usd:.2f} \\\\")
        lines.append(r"\midrule")
        cf = "Cascade flip (4o-mini scout)"
        out["mixed_extra"] = {
            "cascade_flip_vs": {lab_: {"diff": paired_diff(pooled[cf]["corr"], pooled[lab_]["corr"]),
                                       "mcnemar": dict(zip("bcp", mcnemar(pooled[cf]["corr"], pooled[lab_]["corr"])))}
                                for lab_ in pooled if lab_ != cf},
            # Mixture sensitivity: w = fraction of PubMedQA-like questions in the workload.
            "mixture": {f"{w:.2f}": {lab_: {"acc": (1 - w) * out["medqa"][lab_]["acc"] + w * out["pubmedqa"][lab_]["acc"],
                                            "usd_per_1k": (1 - w) * out["medqa"][lab_]["usd_per_1k"]
                                            + w * out["pubmedqa"][lab_]["usd_per_1k"]}
                                     for lab_ in out["medqa"] if lab_ in out["pubmedqa"]}
                        for w in np.linspace(0, 1, 11)}}
    if lines:
        lines.pop()
    tex = (r"\begin{tabular}{lcccrrr}" "\n" r"\toprule" "\n"
           r"Policy & Acc. & 95\% CI & vs.\ 4.1 naive & 4.1 in tok & \$/1k q & Cost \\" "\n"
           r"\midrule" "\n" + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    return tex, out


E8_ROWS = [("cascade", "gpt-4.1", "closed_book", "", "gpt-4.1 closed-book"),
           ("cascade", "gpt-4.1", "naive", "", "gpt-4.1 naive"),
           ("cascade", "gpt-4.1", "flip", "cascade", "Cascade flip (4o-mini scout)"),
           ("scale", "gpt-4o-mini", "naive", "", "gpt-4o-mini naive"),
           ("scale", "gpt-4o-mini", "flip", "", "gpt-4o-mini flip"),
           ("cascade", "gpt-4.1", "flip", "cascade+verify", "Cascade flip + verifier"),
           ("scale", "gpt-4o-mini", "flip", "+verify", "gpt-4o-mini flip + verifier"),
           ("scale", "gpt-4o-mini", "flip", "+verifyH", "gpt-4o-mini flip + Haiku verifier"),
           ("scale", "gpt-4o-mini", "flip", "+verifyG", "gpt-4o-mini flip + gpt-4.1 verifier"),
           ("cascade", "gpt-4.1", "naive", "+vnaive", "gpt-4.1 naive + verifier"),
           ("cascade", "gpt-4.1", "naive", "+vdis", "gpt-4.1 naive + verifier (disagreements)")]
E8_FIXED = ("gpt-4.1 closed-book", "gpt-4.1 naive", "gpt-4o-mini naive")


def same_format(rows) -> tuple[str, dict]:
    """E8: every PubMedQA question twice, with and without its gold abstract in a 17-passage pool. A question-only
    router sees identical inputs in both halves, so any routing gain must come from the evidence itself."""
    halves = {}
    for gold in (True, False):
        src = {"cascade": by_distract(rows, "cascade", "gpt-4.1", gold), "scale": by_distract(rows, "scale", "gpt-4o-mini", gold)}
        halves[gold] = {lab: {r["qid"]: r for r in src[exp].get(("pubmedqa", 16, gt, var), [])}
                        for exp, _, gt, var, lab in E8_ROWS}
    # Closed-book never sees the pool, so its no-gold half is the gold half's answers.
    halves[False]["gpt-4.1 closed-book"] = halves[True]["gpt-4.1 closed-book"]
    if not all(halves[False].values()):
        return "", {}
    common = set.intersection(*(set(c) for h in halves.values() for c in h.values()))
    pol = {}
    for _, _, _, _, lab in E8_ROWS:
        pol[lab] = {"corr": {(g, q): halves[g][lab][q]["correct"] for g in halves for q in common},
                    "usd": {(g, q): q_cost(halves[g][lab][q])["usd"] for g in halves for q in common},
                    "adm": {(g, q): bool(halves[g][lab][q].get("admitted")) for g in halves for q in common}}
    acc = {lab: float(np.mean(list(p_["corr"].values()))) for lab, p_ in pol.items()}
    best = max(E8_FIXED, key=lambda lab: acc[lab])
    ref_usd = float(np.mean(list(pol["gpt-4.1 naive"]["usd"].values())))
    out: dict = {"n_questions": len(common), "best_fixed": best, "policies": {}}
    lines = []
    for _, _, _, _, lab in E8_ROWS:
        c = pol[lab]["corr"]
        k, n = sum(c.values()), len(c)
        lo, hi = wilson(k, n)
        half = {g: float(np.mean([c[(g, q)] for q in common])) for g in halves}
        adm = {g: float(np.mean([pol[lab]["adm"][(g, q)] for q in common])) for g in halves}
        usd = float(np.mean(list(pol[lab]["usd"].values())))
        b, cc, p = mcnemar(c, pol[best]["corr"]) if lab != best else (0, 0, float("nan"))
        out["policies"][lab] = {"acc": k / n, "ci": [lo, hi], "acc_gold": half[True], "acc_nogold": half[False],
                                "admit_gold": adm[True], "admit_nogold": adm[False], "usd_per_1k": usd * 1000,
                                "cost_vs_41_naive": usd / ref_usd, "vs_best_fixed": {"b": b, "c": cc, "p": p},
                                "diff_vs_best_fixed": paired_diff(c, pol[best]["corr"]) if lab != best else None}
        cd = cluster_diff(c, pol["gpt-4.1 naive"]["corr"], common) if lab != "gpt-4.1 naive" else None
        out["policies"][lab]["clustered_vs_41_naive"] = cd
        ptxt = "--" if cd is None else (f"{cd['diff']:+.3f} [{cd['ci'][0]:+.2f},{cd['ci'][1]:+.2f}], "
                                        + (f"{cd['p']:.2f}" if cd['p'] >= 0.01 else f"{cd['p']:.0e}"))
        admtxt = f"{adm[True] * 100:.0f}\\% / {adm[False] * 100:.0f}\\%" if ("flip" in lab or "verifier" in lab) else "--"
        if "flip" in lab or "verifier" in lab:
            ag, an = pol[lab]["adm"], pol[lab]["adm"]
            out["policies"][lab]["admit_paired"] = dict(zip(("only_gold", "only_nogold", "p"), mcnemar(
                {q: ag[(True, q)] for q in common}, {q: an[(False, q)] for q in common})))
        if "verifier" not in lab:
            lines.append(f"{lab} & {half[True]:.3f} & {half[False]:.3f} & {k / n:.3f} & {ptxt} & "
                         f"{admtxt} & \\${usd * 1000:.2f} \\\\")
    # Does the scout's admission track whether the evidence helps gpt-4.1 (naive right vs. closed-book right)?
    cf = pol["Cascade flip (4o-mini scout)"]
    help_ = {k: pol["gpt-4.1 naive"]["corr"][k] and not pol["gpt-4.1 closed-book"]["corr"][k] for k in cf["corr"]}
    hurt = {k: pol["gpt-4.1 closed-book"]["corr"][k] and not pol["gpt-4.1 naive"]["corr"][k] for k in cf["corr"]}
    out["scout_vs_41_effect"] = {
        "admit_when_ctx_helps": float(np.mean([cf["adm"][k] for k in help_ if help_[k]])) if any(help_.values()) else None,
        "admit_when_ctx_hurts": float(np.mean([cf["adm"][k] for k in hurt if hurt[k]])) if any(hurt.values()) else None,
        "admit_when_neutral": float(np.mean([cf["adm"][k] for k in help_ if not help_[k] and not hurt[k]])),
        "n_helps": sum(help_.values()), "n_hurts": sum(hurt.values())}
    # Reference only (not a deployable policy): gpt-4.1 told which half it is in.
    og, ong = out["policies"]["gpt-4.1 naive"]["acc_gold"], out["policies"]["gpt-4.1 closed-book"]["acc_nogold"]
    out["evidence_oracle"] = {"acc": (og + ong) / 2, "acc_gold": og, "acc_nogold": ong}
    oracle = {(g, q): pol["gpt-4.1 naive" if g else "gpt-4.1 closed-book"]["corr"][(g, q)] for g in halves for q in common}
    out["evidence_oracle"]["clustered_vs_41_naive"] = cluster_diff(oracle, pol["gpt-4.1 naive"]["corr"], common)
    for lab in pol:
        out["policies"][lab]["clustered_vs_oracle"] = cluster_diff(pol[lab]["corr"], oracle, common)
    out["cascver_vs_nvv"] = cluster_diff(pol["Cascade flip + verifier"]["corr"], pol["gpt-4.1 naive + verifier"]["corr"], common)
    out["miniver_vs_nvv"] = cluster_diff(pol["gpt-4o-mini flip + verifier"]["corr"], pol["gpt-4.1 naive + verifier"]["corr"], common)
    lines.append(r"\midrule" "\n" f"\\textit{{Condition-aware reference (gpt-4.1)}} & {og:.3f} & {ong:.3f} & {(og + ong) / 2:.3f} "
                 + (lambda cd: f"& {cd['diff']:+.3f} [{cd['ci'][0]:+.2f},{cd['ci'][1]:+.2f}], {cd['p']:.2f} ")(
                     out["evidence_oracle"]["clustered_vs_41_naive"]) + r"& 100\% / 0\% & -- \\")
    tex = (r"\begin{tabular}{lcccccr}" "\n" r"\toprule" "\n"
           r"Policy & Gold & No gold & Mixed & $\Delta$ vs.\ 4.1 naive [95\% CI], $p$ & Admit (g/ng) & \$/1k q \\" "\n"
           r"\midrule" "\n" + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    return tex, out


def verifier_stats() -> dict:
    """E9 discrimination: what each verifier does to admissions of known provenance."""
    out = {}
    for m in VERIFIERS:
        vf = RES / "verifier" / f"verify_{m.replace('.', '')}.jsonl"
        if not vf.exists():
            continue
        d = {}
        for line in vf.read_text().splitlines():
            v = json.loads(line)
            if v.get("verdict") is not None or v.get("prompt_tokens"):
                d[(v["run"], v["qid"])] = v
        sup = lambda v: bool((v["verdict"] or {}).get("supported"))
        groups = {"gold": [v for (run, _), v in d.items() if "pubmedqa" in run and "nogold" not in run and v["kept_gold"]],
                  "nogold": [v for (run, _), v in d.items() if "nogold" in run],
                  "medqa": [v for (run, _), v in d.items() if "_medqa_" in run]}
        out[m] = {g: {"n": len(vs), "supported": sum(map(sup, vs))} for g, vs in groups.items()}
        out[m]["no_verdict"] = sum(v["verdict"] is None for v in d.values())
        out[m]["mean_in"] = float(np.mean([v["prompt_tokens"] for v in d.values()]))
        out[m]["mean_out"] = float(np.mean([v["completion_tokens"] for v in d.values()]))
    return out


def verifier_table(summary: dict) -> str:
    """E9: verifier policies next to their unverified counterparts on E7 and E8."""
    casc, sf = summary.get("cascade", {}), summary.get("same_format", {}).get("policies", {})
    rows = [("gpt-4.1 naive", "gpt-4.1 naive"),
            ("gpt-4.1 naive + verifier", "\\quad + Sonnet verifier (no gate)"),
            ("gpt-4.1 naive + verifier (disagreements)", "\\quad + Sonnet verifier (disagreements only)"),
            ("gpt-4o-mini flip", "4o-mini flip"),
            ("gpt-4o-mini flip + verifier", "\\quad + Sonnet verifier"),
            ("gpt-4o-mini flip + Haiku verifier", "\\quad + Haiku verifier"),
            ("gpt-4o-mini flip + gpt-4.1 verifier", "\\quad + gpt-4.1 verifier"),
            ("Cascade flip (4o-mini scout)", "Cascade flip"),
            ("Cascade flip + verifier", "\\quad + Sonnet verifier")]
    f = lambda d, k: f"{d[k]:.3f}" if d and k in d else "--"
    lines = []
    for key, lab in rows:
        e = sf.get(key, {})
        lines.append(f"{lab} & {f(casc.get('medqa', {}).get(key), 'acc')} & {f(casc.get('pubmedqa', {}).get(key), 'acc')} & "
                     f"{f(casc.get('mixed', {}).get(key), 'acc')} & \\${casc.get('mixed', {}).get(key, {}).get('usd_per_1k', float('nan')):.2f} & "
                     f"{f(e, 'acc_gold')} & {f(e, 'acc_nogold')} & {f(e, 'acc')} & "
                     + (f"{e['admit_gold'] * 100:.0f}\\% / {e['admit_nogold'] * 100:.0f}\\%" if e and ("flip" in key or "verifier" in key) else "--")
                     + " \\\\")
    o = summary.get("same_format", {}).get("evidence_oracle")
    if o:
        lines.append(r"\midrule" "\n" f"\\textit{{Condition-aware reference (gpt-4.1)}} & -- & -- & -- & -- & {o['acc_gold']:.3f} & "
                     f"{o['acc_nogold']:.3f} & {o['acc']:.3f} & 100\\% / 0\\% \\\\")
    return (r"\begin{tabular}{lccccccccc}" "\n" r"\toprule" "\n"
            r" & \multicolumn{4}{c}{E7 (gold in pool)} & \multicolumn{4}{c}{E8 (PubMedQA, same questions)} \\" "\n"
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}" "\n"
            r"Policy & MedQA & PubMedQA & Mixed & \$/1k q & Gold & No gold & Mixed & Admit (g/ng) \\" "\n"
            r"\midrule" "\n" + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


def paper_macros(summary: dict) -> str:
    """E8/E9 numbers quoted in the prose, so the text cannot drift from the tables."""
    sf, ver = summary.get("same_format", {}), summary.get("verifier", {})
    P = sf.get("policies", {})
    if not P or not ver:
        return ""
    pct = lambda x: f"${x * 100:.0f}\\%$"
    acc = lambda x: f"${x:.3f}$"
    frac = lambda d: f"${d['supported']}/{d['n']}$"
    casc, nv, mini = P["Cascade flip (4o-mini scout)"], P["gpt-4.1 naive"], P["gpt-4o-mini flip + verifier"]
    sg = lambda x: (lambda t: "{" + t[0] + "}" + t[1:])(f"{x:+.3f}")
    cl = lambda d: (f"$\\Delta{{=}}{sg(d['diff'])}$, 95\\% CI $[{sg(d['ci'][0])},{sg(d['ci'][1])}]$, "
                    f"$p{{=}}{d['p']:.{2 if d['p'] >= 0.01 else 3}f}$")
    eff = sf["scout_vs_41_effect"]
    S, H = ver["claude-sonnet-5.5"], ver["claude-haiku-5.5"]
    m = {"Eightadmitg": pct(casc["admit_gold"]), "Eightadmitng": pct(casc["admit_nogold"]),
         "Eightcascg": acc(casc["acc_gold"]), "Eightnaiveg": acc(nv["acc_gold"]),
         "Eightcascng": acc(casc["acc_nogold"]), "Eightcbng": acc(P["gpt-4.1 closed-book"]["acc_nogold"]),
         "Eightcasc": acc(casc["acc"]), "Eightnaive": acc(nv["acc"]), "Eightcasccost": pct(casc["cost_vs_41_naive"]),
         "Eightmininaive": acc(P["gpt-4o-mini naive"]["acc"]),
         "Eightmininaivecost": pct(P["gpt-4o-mini naive"]["cost_vs_41_naive"]),
         "Eightminiflip": acc(P["gpt-4o-mini flip"]["acc"]),
         "Eighthelps": f"{pct(eff['admit_when_ctx_helps'])} (of {eff['n_helps']})",
         "Eighthurts": f"{pct(eff['admit_when_ctx_hurts'])} (of {eff['n_hurts']})",
         "Eightminiver": acc(mini["acc"]), "Eightminivervs": cl(mini["clustered_vs_41_naive"]),
         "Eightminiveror": cl(mini["clustered_vs_oracle"]), "Eightcascvs": cl(casc["clustered_vs_41_naive"]),
         "Eightminiflipvs": cl(P["gpt-4o-mini flip"]["clustered_vs_41_naive"]),
         "Eightoraclevs": cl(sf["evidence_oracle"]["clustered_vs_41_naive"]),
         "Eightadmpair": f"${casc['admit_paired']['only_gold']}$ vs.\\ ${casc['admit_paired']['only_nogold']}$ questions, "
                         f"$p{{=}}{casc['admit_paired']['p']:.2f}$",
         "Eightminiverp": f"$p{{=}}{mini['clustered_vs_41_naive']['p']:.2f}$",
         "Eightoracle": acc(sf["evidence_oracle"]["acc"]), "Eightminivercost": pct(mini["cost_vs_41_naive"]),
         "Eightminiveradmitg": pct(mini["admit_gold"]),
         "Eightcascverng": acc(P["Cascade flip + verifier"]["acc_nogold"]),
         "Eightn": f"${sf['n_questions']}$",
         "Ninehaikunone": f"${H['no_verdict']}$", "Ninesngn": f"${S['nogold']['n']}$",
         "Ninesgold": frac(S["gold"]), "Ninesgoldpct": pct(S["gold"]["supported"] / S["gold"]["n"]),
         "Ninesmed": frac(S["medqa"]), "Ninehgold": frac(H["gold"]),
         "Ninehgoldpct": pct(H["gold"]["supported"] / H["gold"]["n"]), "Ninehngn": frac(H["nogold"])}
    G = ver.get("gpt-4.1")
    if G and "gpt-4o-mini flip + gpt-4.1 verifier" in P:
        g41 = P["gpt-4o-mini flip + gpt-4.1 verifier"]
        m.update(Ninefgold=frac(G["gold"]), Ninefngn=frac(G["nogold"]), Ninefmed=frac(G["medqa"]),
                 Ninefnone=f"${G['no_verdict']}$", Eightminiverg=acc(g41["acc"]),
                 Eightminivergng=acc(g41["acc_nogold"]), Eightminivergvs=cl(g41["clustered_vs_41_naive"]))
    if "gpt-4.1 naive + verifier" in P:
        nvv = P["gpt-4.1 naive + verifier"]
        m.update(Eightnvv=acc(nvv["acc"]), Eightnvvg=acc(nvv["acc_gold"]), Eightnvvng=acc(nvv["acc_nogold"]),
                 Eightnvvvs=cl(nvv["clustered_vs_41_naive"]), Eightnvvcost=pct(nvv["cost_vs_41_naive"]),
                 Eightnvvadm=f"{pct(nvv['admit_gold'])} / {pct(nvv['admit_nogold'])}",
                 Eightminivsnvv=cl(sf["miniver_vs_nvv"]), Eightcascvsnvv=cl(sf["cascver_vs_nvv"]),
                 Eightcascver=acc(P["Cascade flip + verifier"]["acc"]),
                 Eightcascvercost=pct(P["Cascade flip + verifier"]["cost_vs_41_naive"]))
    if "gpt-4.1 naive + verifier (disagreements)" in P:
        d = P["gpt-4.1 naive + verifier (disagreements)"]
        m.update(Eightdis=acc(d["acc"]), Eightdiscost=pct(d["cost_vs_41_naive"]), Eightdisvs=cl(d["clustered_vs_41_naive"]))
    return "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items())


def value_figures(scale: dict, casc: dict):
    """E6: accuracy vs. pool size. E7: accuracy vs. dollar cost per policy (cascade)."""
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    if scale:
        fig, axes = plt.subplots(1, len(scale), figsize=(3.4 * len(scale), 2.6), squeeze=False)
        for ax, (ds, cells) in zip(axes[0], scale.items()):
            ds_ = sorted(cells, key=int)
            x = [int(d) + 1 for d in ds_]
            for gt, ls in (("closed_book", ":"), ("naive", "-"), ("flip", "-"), ("grounded", "--")):
                y = [cells[d]["acc"].get(gt) for d in ds_]
                if all(v is not None for v in y):
                    ax.plot(x, y, ls, marker="o", ms=4, color=f"C{GATE_ORDER.index(gt)}", label=GATE_TEX[gt])
            ax.set_xscale("log", base=2)
            ax.set_xticks(x, [str(v) for v in x])
            ax.set_xlabel("passages in pool")
            ax.set_ylabel("accuracy")
            ax.set_title({"medqa": "MedQA", "pubmedqa": "PubMedQA"}[ds])
            ax.grid(alpha=.3)
        axes[0][0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "acc_vs_pool.pdf")
        fig.savefig(OUT / "acc_vs_pool.png", dpi=200)
        plt.close(fig)
    panels = [k for k in ("mixed", "medqa", "pubmedqa") if k in casc]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 2.7), squeeze=False)
        style = {"gpt-4.1 closed-book": ("C0", "o"), "gpt-4.1 naive": ("C1", "o"),
                 "Cascade flip (4o-mini scout)": ("C2", "*"), "Cascade grounded (4o-mini scout)": ("C4", "*"),
                 "gpt-4o-mini naive": ("C1", "s"), "gpt-4o-mini flip": ("C2", "s"),
                 "Cascade flip + verifier": ("C5", "P"), "gpt-4o-mini flip + verifier": ("C5", "s"), "gpt-4o-mini flip + Haiku verifier": ("C6", "s"),
                 ROUTER: ("C3", "D")}
        for ax, k in zip(axes[0], panels):
            for label, r in casc[k].items():
                if "verifier" in label:
                    continue
                c, mk = style.get(label, ("k", "o"))
                ax.errorbar(r["usd_per_1k"], r["acc"], yerr=[[r["acc"] - r["ci"][0]], [r["ci"][1] - r["acc"]]],
                            fmt=mk, ms=9 if mk == "*" else 5, capsize=2, color=c, label=label)
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.1, 0.3, 1, 3, 10]))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlim(0.1, 12)
            ax.set_xlabel("\\$ per 1k questions (log)")
            ax.set_ylabel("accuracy")
            ax.set_title({"mixed": "Mixed workload", "medqa": "MedQA", "pubmedqa": "PubMedQA"}[k])
            ax.grid(alpha=.3, which="both")
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, fontsize=7.5, loc="lower center", ncol=(len(l) + 1) // 2)
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        fig.savefig(OUT / "cascade_cost.pdf")
        fig.savefig(OUT / "cascade_cost.png", dpi=200)
        plt.close(fig)


def main():
    rows = load()
    rows += [v for m in VERIFIERS for v in verified_rows(rows, m)] + naive_verified_rows(rows) + naive_verified_rows(rows, disagree_only=True)
    g = group(rows)
    summary: dict = {"n_rows": len(rows)}
    tabs = {}
    for name, exp, model, gates in [
        ("fidelity_4omini", "fidelity", "gpt-4o-mini", GATE_ORDER),
        ("fidelity_claude", "fidelity", "claude-haiku-4.5", ["closed_book", "naive", "flip"]),
        ("open_4omini", "open", "gpt-4o-mini", ["closed_book", "naive", "flip", "grounded"]),
        ("live_4omini", "live", "gpt-4o-mini", ["closed_book", "naive", "flip", "grounded"]),
    ]:
        tex, s = accuracy_table(g, exp, model, gates=gates, latency=exp != "live")
        if s:
            (OUT / f"table_{name}.tex").write_text(tex)
            summary[name] = s
            tabs[name] = s
    summary["open_signed_effect"] = signed_effect_analysis(rows)
    summary["fidelity_signed_effect_tv"] = signed_effect_analysis(rows, exp="fidelity", gate="logit_shift", thresh=0.15)
    sched = sched_analysis(rows)
    summary["sched"] = {ds: {v: ({k: x for k, x in d.items() if k != "walls"} if isinstance(d, dict) else d)
                             for v, d in dd.items()} for ds, dd in sched.items()}
    summary["leakage"] = leakage(rows)
    summary["mixed_workload"] = mixed_workload(g)
    for name, fn in (("scale", pool_scaling), ("cascade", cascade), ("same_format", same_format)):
        tex, s_ = fn(rows)
        if s_:
            (OUT / f"table_{name}.tex").write_text(tex)
            summary[name] = s_
    figures(rows, sched, {"Paper protocol (MCQ)": tabs.get("fidelity_4omini"),
                          "Open-ended (semantic)": tabs.get("open_4omini"),
                          "Live hybrid retrieval": tabs.get("live_4omini")})
    value_figures(summary.get("scale", {}), summary.get("cascade", {}))
    summary["verifier"] = verifier_stats()
    (OUT / "table_verifier.tex").write_text(verifier_table(summary))
    (OUT / "macros_e8e9.tex").write_text(paper_macros(summary))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
