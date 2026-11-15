# Auditing a RAG system

## Audit proxy (live traffic)

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
