#!/usr/bin/env bash
# Finishes the E6/E7 runs: waits out Copilot's utility-model rate limit, then resumes (runs are resumable;
# rows that hit an API error are recorded as errors and retried on the next pass).
set -u
cd "$(dirname "$0")"
export PYTHONPATH=../backend
PY=../backend/.venv/bin/python
wait_limit() {
  while :; do
    ra=$(curl -s -D - -o /dev/null https://api.githubcopilot.com/chat/completions \
      -H "Authorization: Bearer $(gh auth token)" -H "Copilot-Integration-Id: vscode-chat" \
      -H "Editor-Version: vscode/1.95.0" -H "Content-Type: application/json" \
      -d "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":2}" \
      | awk 'tolower($1)=="retry-after:"{print $2+0}')
    [ -z "$ra" ] && return
    echo "$(date +%T) $1 rate-limited; sleeping $((ra + 60))s"; sleep $((ra + 60))
  done
}
run() { echo "=== $(date +%T) $*"; $PY run_eval.py "$@" 2>&1 | grep -vi warn; }
for pass in 1 2 3; do
  wait_limit gpt-4o-mini
  run scale --dataset medqa --n 150 --seed 1 --n-distract 32 --gates closed_book,naive,flip,grounded --concurrency 2
  for ds in pubmedqa medqa; do
    wait_limit gpt-4.1
    run cascade --dataset $ds --n 150 --seed 1 --model gpt-4.1 --scout gpt-4o-mini --n-distract 16 --concurrency 2
  done
done
echo "=== RESUME DONE $(date +%T)"
