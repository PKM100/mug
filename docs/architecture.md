# Code layout

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
