#!/usr/bin/env bash
# Unified test runner. Mode is chosen with TEST_MODE (.env or env var):
#   TEST_MODE=smoke  -> fast, no GPU, no 2B download (default)
#   TEST_MODE=full   -> venv -> install -> data (Zenodo + local) -> real-model eval
#
# Usage:
#   bash scripts/test.sh                 # smoke
#   TEST_MODE=full bash scripts/test.sh  # full
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${TEST_MODE:-smoke}"
case "$MODE" in
  smoke) bash scripts/smoke_test.sh ;;
  full)  bash scripts/full_test.sh ;;
  *) echo "TEST_MODE must be 'smoke' or 'full' (got '$MODE')"; exit 1 ;;
esac