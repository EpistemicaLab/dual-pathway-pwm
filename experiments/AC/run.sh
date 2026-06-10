#!/bin/bash
# Experiment AC: launcher.
# Defaults assume a venv at $REPO/.venv. Override REPO / VENV / OUT via env vars.
# Example:
#   REPO=$PWD VENV=$PWD/.venv ./experiments/AC/run.sh
# Or from any cwd:
#   ./experiments/AC/run.sh

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)}"
EXP_DIR="$REPO/experiments/AC"
OUT="${OUT:-./results-AC.json}"
VENV="${VENV:-$REPO/.venv}"

cd "$REPO"

if [[ ! -d "$VENV" ]]; then
  echo "[AC] FATAL: venv not found at $VENV" >&2
  echo "[AC] Hint: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 2
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

cd "$EXP_DIR"
exec python run.py "$@" --out "$OUT"
