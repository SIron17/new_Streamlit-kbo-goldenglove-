#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "[pre_pr_check] 1) git status"
git status --short

echo "[pre_pr_check] 2) conflict marker scan"
if rg -n '^(<<<<<<<|=======|>>>>>>>)' README.md AGENTS.md app src tests requirements.txt; then
  echo "Conflict markers found. Resolve before PR."
  exit 1
fi

echo "[pre_pr_check] 3) CLI help smoke"
python -m src.pipeline.build_dataset --help >/dev/null
python -m src.train.train_model --help >/dev/null
python -m src.train.predict --help >/dev/null

echo "[pre_pr_check] done"
