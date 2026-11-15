#!/usr/bin/env bash
# Experiments for "where does gating add value": pool-size scaling (E6) and the cascade (E7).
# Usage: ./run_value.sh scale | cascade
set -u
cd "$(dirname "$0")"
export PYTHONPATH=../backend
PY=../backend/.venv/bin/python
run() { echo "=== $(date +%T) $*"; $PY run_eval.py "$@" 2>&1 | grep -vi warn; }
case "${1:-}" in
  scale)
    for d in 16 32; do for ds in pubmedqa medqa; do
      run scale --dataset $ds --n 150 --seed 1 --n-distract $d --gates closed_book,naive,flip,grounded --concurrency 5
    done; done ;;
  cascade)
    for ds in pubmedqa medqa; do
      run cascade --dataset $ds --n 150 --seed 1 --model gpt-4.1 --scout gpt-4o-mini --n-distract 16 --concurrency 3
    done ;;
  *) echo "usage: $0 scale|cascade"; exit 2 ;;
esac
echo "=== $1 DONE $(date +%T)"
