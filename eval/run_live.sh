#!/usr/bin/env bash
# Re-runs the live-retrieval experiments (E2 scheduling, E4 live accuracy).
set -u
cd "$(dirname "$0")"
export PYTHONPATH=../backend
PY=../backend/.venv/bin/python
run() { echo "=== $(date +%T) $*"; $PY run_eval.py "$@" 2>&1 | grep -vi warn; }
run sched --dataset pubmedqa --n 60 --seed 2 --concurrency 2
run sched --dataset medqa    --n 60 --seed 2 --concurrency 2
run live --dataset medqa    --n 150 --seed 3 --gates closed_book,naive,flip,grounded --concurrency 4
run live --dataset pubmedqa --n 150 --seed 3 --gates closed_book,naive,flip,grounded --concurrency 4
echo "=== LIVE DONE $(date +%T)"
