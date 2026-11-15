# Replay mode, evaluation and paper

## Replay mode (no credentials, no network)

`#/replay` renders three **recorded** cases in the normal UI (decision card, trace timeline, evidence pool) and
`#/replay/audit` renders them as audit-log entries. Nothing is sent to a model or to PubMed; the page reads
`replay/cases.json` and `replay/audit.json`, and is labelled "Recorded · not live".

```bash
cd frontend && npm install && npm run build
cd dist && python3 -m http.server 8765    # open http://127.0.0.1:8765/#/replay (no backend needed)
```

The cases are (a) `medqa:0359`, where evidence does not move the answer; (b) `medqa:1040`, where the evidence moves a
correct closed-book answer to a wrong one, the verifier finds no supporting quote, and the evidence is withheld; and (c)
`pubmedqa:11567820`, where the study abstract moves a wrong answer to the right one and the verifier returns a verbatim
quote. They are taken from `eval/results/scale_*_gpt4omini_d16_s1.jsonl` (17-passage paper-protocol pools, gpt-4o-mini)
and the verifier verdicts in `eval/results/verifier/verify_claude-sonnet-55.jsonl`, which were recorded in a separate
pass, so the verifier span is shown as untimed. Rebuild with
`backend/.venv/bin/python eval/build_replay_cases.py`.

## Reproduce the paper's experiments

```bash
cd eval && ./run_all.sh            # resumable; writes eval/results/*.jsonl
./run_live.sh                      # re-runs only the live-PubMed experiments (E2 scheduling, E4)
./run_value.sh scale|cascade       # E6 pool scaling, E7 cascade (gpt-4.1 answers, gpt-4o-mini scouts)
./resume_value.sh                  # finishes E6/E7, waiting out Copilot rate limits between passes
../backend/.venv/bin/python analyze.py   # writes eval/analysis/{summary.json,table_*.tex,*.pdf}
```

The harness reads `mirage_benchmark.json` (the MIRAGE `benchmark.json`) and `pubmedqa_labeled.json` from
`MUG_DATA_DIR`. `analyze.py` also contains analyses for a follow-up study whose result files are not included here,
so those parts are skipped or fail. Results files are resumable: a run skips rows already present, so move stale `results/*.jsonl` to
`results/archive/` before re-running after a code change.
Rows whose LLM calls failed (e.g. rate limits) are recorded with an `error` field and retried on the next run.

To use the cascade from the API, pass `"scout_model": "gpt-4o-mini"` with an expensive `model` in `/api/ask`:
the scout runs the probes and filter, and the answer model only sees the admitted context.

## Paper

The demo paper is in `paper_demo/` (`tectonic -X compile main.tex`). Its extra statistics (paired bootstrap and
Wilson CIs, leakage split, audit-label tabulation, latency baselines reconstructed from spans) come from
`eval/demo_analyses.py` → `eval/analysis/demo_analyses.json`, computed from existing result files without LLM calls.
Fig. 1 is composed by `eval/make_fig1.py` from Replay-mode screenshots in `paper_demo/figs/shots/`.
