# MUG · Marginal-Utility Gated Medical QA

A research prototype that serves the gates from *Marginal-Utility Gating: Context Engineering by
Answer-Movement, Not Intrinsic Relevance* as an interactive medical Q&A
application. The app retrieves evidence (MedRAG textbooks + live PubMed). It admits that evidence
into the final answer only when the evidence **moves** the model's answer, and optionally only
when the evidence also **supports** the moved answer.

> ⚠️ Research prototype. Not medical advice. Do not use for clinical decisions.

## When is this useful?

MUG asks one question about retrieved evidence: *did it change the model's answer?* Evidence that changes nothing is
dropped. Evidence that changes the answer is kept only if, optionally, a verifier model finds a quote in it that
supports the new answer. That makes the app useful in these cases:

- **Auditing an existing RAG system.** Point your RAG system's generation call at the audit proxy in *shadow* mode.
  Your users still get plain-RAG answers, and every request is labelled: the documents had no effect, they changed
  the answer with a supporting quote, or they changed it with no quote found. The Audit page then shows where your
  retriever is steering answers without support. This is the main intended use.
- **Comparing retrieval policies offline.** The batch auditor runs naive RAG, closed-book and the gates on the same
  frozen documents (`python -m mug.audit cases.jsonl`). With reference answers, you get accuracy, token cost and
  latency per policy before changing anything in production.
- **Noisy or partly irrelevant retrieval.** When most retrieved passages are off-topic, the gate keeps few of them.
  Under the original study's protocol (`gpt-4o-mini`, 17-passage pools) it matched naive RAG's accuracy while
  putting only 5-9% of naive RAG's evidence tokens in the final prompt.
- **Inspecting a single answer.** For one question, the trace shows what was retrieved, whether it moved the
  answer, which passages were kept or withheld, the verifier's quote, and what each step cost in tokens and time.
  This is useful for debugging, teaching, and explaining why a RAG answer changed.
- **Cost-sensitive answering with a strong model.** In cascade mode a cheap scout decides whether evidence matters
  before the expensive model reads it.
- **Studying gating itself.** Gates, thresholds, schedules (serial or parallel probes) and models are all
  configurable, and the evaluation harness reproduces the demo paper's numbers.

Where it does **not** help, or needs care:

- **It is not a correctness check.** "No effect" means the evidence did not change the answer, not that the answer
  is right. "Supported" is an LLM judgement that a verbatim quote backs the answer, not a check against clinical
  truth.
- **When the evidence argues against the right answer.** A gate that only measures whether the answer moved cannot
  tell good evidence from misleading evidence. The verifier helps when the right source is missing, but
  plausible-looking evidence that argues against the right answer can also fool it.
- **When the evidence is general background.** When the evidence needs reasoning rather than stating the finding
  (textbook background, for example), there is often no single quote to find, so the verifier tends to withhold it.
- **Some model and retriever combinations.** With `claude-haiku-4.5` the gate scored below closed-book on MedQA.
  Check your own setup with the batch auditor before relying on it.
- **Latency.** The gate adds model calls. Parallel probes help, but it is still slower than naive RAG.
- **Clinical use.** Not for clinical decisions.

## Layout

| Path | What |
| --- | --- |
| `backend/mug/gates.py` | Gates: `flip` (answer-flip + marginal-utility filter), `logit_shift` (TV movement), `grounded` (movement × entailment), `naive`, `closed_book`. MCQ and open-ended (semantic movement). Overlapped and speculative scheduling. |
| `backend/mug/pipeline.py` | `Engine`: hybrid retrieval → gate → result + per-span trace |
| `backend/mug/llm.py` | Async chat client for the OpenAI API, the Anthropic API and the GitHub Copilot endpoint (logprobs, retries, token accounting) |
| `backend/mug/retrieval/` | MedRAG textbook dense index (reuses the paper's MiniLM index) and NCBI E-utilities PubMed client |
| `backend/app.py` | FastAPI server: `/api/ask`, `/api/ask/stream` (SSE), `/api/config`, `/api/health`, the audit proxy `/v1/chat/completions` and `/api/audit`; serves `frontend/dist` |
| `backend/mug/audit.py` | Audit log (SQLite), outcome labels, and the frozen-pool batch auditor `python -m mug.audit` |
| `frontend/` | React + TypeScript UI: live pipeline timeline, decision card, evidence panel, gate comparison, Audit and Replay pages |
| `frontend/public/replay/` | Three recorded cases for Replay mode (built by `eval/build_replay_cases.py` from stored traces) |
| `eval/` | Evaluation harness (`run_eval.py`, `run_all.sh`) and analysis (`analyze.py`) for the systems paper |
| `paper_demo/` | LaTeX source of the demo paper |
| `scripts/download_index.py` | Downloads the MedRAG textbooks and builds the retrieval index |
| `Dockerfile` | One image with the backend and the built UI |

## Requirements

- Python 3.12 and Node 20 (or Docker)
- A model provider. Set **one or more** of:
  - `OPENAI_API_KEY`: OpenAI models (`gpt-4o-mini`, `gpt-4o`, `gpt-4.1*`); these return token logprobs.
  - `ANTHROPIC_API_KEY`: Claude models (`claude-haiku-4-5`, `claude-sonnet-4-5`, ...). Anthropic's API does not
    return logprobs, so the gates parse the text answer and the logprob-based gates fall back accordingly.
  - Neither: the app uses GitHub Copilot's endpoint with your GitHub token (`gh auth login` or `GITHUB_TOKEN`);
    this needs a Copilot subscription and is what the paper's runs used.

  With `MUG_PROVIDER=auto` (default) each model goes to the provider whose key is set (`claude-*` to Anthropic,
  everything else to OpenAI) and falls back to Copilot. Force one with `MUG_PROVIDER=openai|anthropic|copilot`.
  If only `ANTHROPIC_API_KEY` is set, the default model is `claude-haiku-4-5`. Add other model ids with
  `MUG_EXTRA_MODELS=a,b`.
- The MedRAG textbook index (optional; ~500 MB). Build it once:

  ```bash
  cd backend && .venv/bin/python ../scripts/download_index.py   # -> ../data/index, ~10-30 min on CPU
  ```

  It downloads the 18 chunked textbooks from the Hugging Face dataset
  [`MedRAG/textbooks`](https://huggingface.co/datasets/MedRAG/textbooks) and embeds 125,847 snippets with
  all-MiniLM-L6-v2. The result is identical to the index used in the paper. The corpus is not redistributed
  here; check the dataset card for its terms. Point elsewhere with `MUG_DATA_DIR`, or set
  `MUG_DISABLE_TEXTBOOKS=1` to run with PubMed only.

## Run

```bash
cd backend
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
export OPENAI_API_KEY=...        # and/or ANTHROPIC_API_KEY; omit both to use GitHub Copilot
PYTHONPATH=. .venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000

cd ../frontend && npm install
npm run dev      # http://localhost:5173 (proxies /api to :8000)
npm run build    # or build once; the backend then serves the UI at http://127.0.0.1:8000
```

Tests: `cd backend && .venv/bin/pytest -q` (no network or keys needed).

## Docker

```bash
docker build -t mug .
docker run --rm -v mug-data:/data mug python ../scripts/download_index.py --data-dir /data   # once
docker run --rm -p 127.0.0.1:8000:8000 -v mug-data:/data -e OPENAI_API_KEY mug              # http://127.0.0.1:8000
```

Pass `-e ANTHROPIC_API_KEY` for Claude, or `-e GITHUB_TOKEN` for Copilot. The volume holds the index and the audit
log (`/data/audit.db`). Keep the port bound to `127.0.0.1` (or put an authenticating proxy in front): the server
has no authentication, and anyone who can reach it spends your API keys.

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

## Audit proxy

`POST /v1/chat/completions` accepts an OpenAI chat-completions request **plus a non-standard `documents` field**
(strings or `{content, title?, url?}`), i.e. the passages your RAG system retrieved. Optional extra fields: `options`
(`{"A": ..., ...}` for multiple choice), `verifier_model`, and `mug_mode`. A stock OpenAI SDK must send these via
`extra_body`. The response is a chat completion with an added `mug_audit` object (outcome, plain-RAG answer,
closed-book answer, kept documents, verifier quote), and every request is logged to `MUG_AUDIT_DB`
(default `~/.mug/audit.db`) and shown on the Audit page.

- **Gate mode** (default) returns MUG's gated answer.
- **Shadow mode** (`"mug_mode": "shadow"`, header `X-MUG-Mode: shadow`, or `MUG_PROXY_MODE=shadow`) never changes
  the answer: it returns what plain RAG would have answered (the answer with all documents) and only labels the
  request. It still adds MUG's latency, because the labelling runs before the response is returned.

Outcome labels: *no effect* (the documents did not move the answer; **not** an endorsement of the answer),
*moved, no verifiable quote* (withheld), *quote found + judged supportive* (a verifier model returned a quote that
occurs verbatim in a kept passage and judged it supportive; this is an LLM judgement, not a check of clinical
correctness), *moved, admitted (no verifier)*, *no documents*, and *failed* (an LLM call failed, so the label is
unreliable).

Limitations: only the **last user message** is read (no system prompt, chat history, or the client's own prompt
template; MUG answers with its own prompt); `stream: true` is rejected; at most 40 documents per request.

## Batch audit on a frozen pool

```bash
cd backend
PYTHONPATH=. .venv/bin/python -m mug.audit cases.jsonl --out audit_out \
    --policies flip,naive,closed_book [--model gpt-4o-mini] [--verifier claude-sonnet-5.5] [--concurrency 2]
```

Each line of `cases.jsonl` is `{"id"?, "question", "options"?, "documents": [...], "reference"?}`
(see `backend/tests/fixtures/audit_cases.jsonl`). All policies see the same frozen documents. The command writes
`audit_out/cases.jsonl` (full results and traces), `audit_out/cases.csv` (one row per case × policy) and
`audit_out/summary.json` (per policy: outcome counts, moved, admitted, verifier-rejected, failed, accuracy when
`reference` is given, mean tokens and wall time). It makes live LLM calls.

## License

MIT, see [LICENSE](LICENSE). The MedRAG textbooks, PubMed and the benchmarks keep their own terms.

## Configuration (env)

| Variable | Default | Meaning |
| --- | --- | --- |
| `MUG_PROVIDER` | `auto` | `auto`, `openai`, `anthropic` or `copilot` (see Requirements) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | – | provider keys |
| `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` | official APIs | e.g. an Azure OpenAI or other OpenAI-compatible endpoint |
| `MUG_ANSWER_MODEL` | `gpt-4o-mini` (`claude-haiku-4-5` if only Anthropic) | answerer |
| `MUG_PROBE_MODEL` | as above | probe for equivalence/groundedness; best with logprobs (OpenAI `gpt-4o*`, `gpt-4.1*`) |
| `MUG_EXTRA_MODELS` | – | extra model ids to offer |
| `MUG_DATA_DIR` | `<repo>/data` | directory holding `index/` |
| `MUG_K_TEXTBOOK` / `MUG_K_PUBMED` | 4 / 3 | candidate pool size per source |
| `MUG_TAU_M` / `MUG_TAU_G` | 0.15 / 0.5 | paper thresholds |
| `MUG_MAX_CONCURRENCY` | 8 | concurrent LLM requests |
| `MUG_REWRITE_DEADLINE_S` | 3.0 | deadline on the LLM PubMed query rewrite; falls back to a keyword query |
| `MUG_VERIFIER_MODEL` | – | default verifier model for the audit proxy |
| `MUG_PROXY_MODE` | `gate` | audit proxy default mode (`gate` or `shadow`) |
| `MUG_AUDIT_DB` | `~/.mug/audit.db` | audit-log location |
| `NCBI_API_KEY` | – | optional; raises the PubMed rate limit from 3 to 10 req/s (the client paces itself and retries on 429) |

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
