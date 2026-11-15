#!/usr/bin/env bash
# Runs the full evaluation suite for the MUG-Serve paper. Resumable (skips done rows).
set -u
cd "$(dirname "$0")"
PY="../backend/.venv/bin/python"
export PYTHONPATH=../backend
run() { echo "=== $(date +%T) $*"; $PY run_eval.py "$@" 2>&1 | grep -vi warn; }

# E1 fidelity: paper protocol, MCQ, gpt-4o-mini, all gates
run fidelity --dataset medqa    --n 200 --seed 1
run fidelity --dataset pubmedqa --n 200 --seed 1
# E1b cross-model: Claude (no logprobs -> flip only meaningful)
run fidelity --dataset medqa    --n 200 --seed 1 --model claude-haiku-4.5 --gates closed_book,naive,flip
run fidelity --dataset pubmedqa --n 200 --seed 1 --model claude-haiku-4.5 --gates closed_book,naive,flip
# E3 open-ended semantic movement
run open --dataset pubmedqa --n 150 --seed 1 --gates closed_book,naive,flip,grounded
run open --dataset medqa    --n 150 --seed 1 --gates closed_book,naive,flip,grounded
# E2 scheduling (live hybrid retrieval), low concurrency to avoid queueing distortion
run sched --dataset pubmedqa --n 60 --seed 2 --concurrency 2  # serial|parallel|overlap|overlap+spec
run sched --dataset medqa    --n 60 --seed 2 --concurrency 2
# E4 live hybrid retrieval end-to-end
run live --dataset medqa    --n 150 --seed 3 --gates closed_book,naive,flip,grounded --concurrency 4
run live --dataset pubmedqa --n 150 --seed 3 --gates closed_book,naive,flip,grounded --concurrency 4
echo "=== ALL DONE $(date +%T)"
