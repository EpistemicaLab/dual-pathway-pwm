#!/bin/bash
# Experiment D4: launcher.
# Defaults assume a venv at $REPO/.venv. Override REPO / VENV / OUT via env vars.
# Example:
#   REPO=$PWD VENV=$PWD/.venv ./experiments/D4/run.sh
# Or from any cwd:
#   ./experiments/D4/run.sh

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)}"
EXP_DIR="$REPO/experiments/D4"
OUT="${OUT:-./results-D4.json}"
VENV="${VENV:-$REPO/.venv}"

cd "$REPO"

if [[ ! -d "$VENV" ]]; then
  echo "[D4] FATAL: venv not found at $VENV" >&2
  echo "[D4] Hint: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 2
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

cd "$EXP_DIR"
exec python run.py "$@" --out "$OUT"
