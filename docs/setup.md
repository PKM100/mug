# Setup and configuration

## Model providers

Requires Python 3.12 and Node 20, or Docker.

Set **one or more** of:

- `OPENAI_API_KEY`: OpenAI models (`gpt-4o-mini`, `gpt-4o`, `gpt-4.1*`); these return token logprobs.
- `ANTHROPIC_API_KEY`: Claude models (`claude-haiku-4-5`, `claude-sonnet-4-5`, ...). Anthropic's API does not
  return logprobs, so the gates parse the text answer and the logprob-based gates fall back accordingly.
- Neither: the app uses GitHub Copilot's endpoint with your GitHub token (`gh auth login` or `GITHUB_TOKEN`);
  this needs a Copilot subscription and is what the paper's runs used.

With `MUG_PROVIDER=auto` (default) each model goes to the provider whose key is set (`claude-*` to Anthropic,
everything else to OpenAI) and falls back to Copilot. Force one with `MUG_PROVIDER=openai|anthropic|copilot`.
If only `ANTHROPIC_API_KEY` is set, the default model is `claude-haiku-4-5`. Add other model ids with
`MUG_EXTRA_MODELS=a,b`.

## Textbook index

Optional (~500 MB). Build it once:

```bash
cd backend && .venv/bin/python ../scripts/download_index.py   # -> ../data/index, ~10-30 min on CPU
```

It downloads the 18 chunked textbooks from the Hugging Face dataset
[`MedRAG/textbooks`](https://huggingface.co/datasets/MedRAG/textbooks) and embeds 125,847 snippets with
all-MiniLM-L6-v2. The result is identical to the index used in the paper. The corpus is not redistributed
here; check the dataset card for its terms. Point elsewhere with `MUG_DATA_DIR`, or set
`MUG_DISABLE_TEXTBOOKS=1` to run with PubMed only.

## Run locally

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

## Environment variables

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
