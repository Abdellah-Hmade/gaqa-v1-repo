#!/usr/bin/env bash
# Full end-to-end test of the GAQA v1 reproduction repository:
#   venv creation -> pip install -> dataset WITH Zenodo -> tiny training smoke
#   -> data-creation modules -> LOCAL dataset build (NO Zenodo) -> evaluation
#   data on the real BitNet model (small sample, CPU-safe).
#
# Usage: bash scripts/full_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv-test/bin/python"

echo "================================================================"
echo "[1/6] Creating venv (.venv-test) + installing requirements"
echo "================================================================"
if [ ! -x "$PY" ]; then
  python3 -m venv .venv-test
fi
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r requirements.txt
"$PY" -c "import torch, transformers, sklearn, requests; print('deps OK: torch', torch.__version__, '| transformers', transformers.__version__)"

echo "================================================================"
echo "[2/6] Dataset WITH Zenodo (auto-download)"
echo "================================================================"
"$PY" -c "from data_utils import load_jsonl; print('knowledge test:', len(load_jsonl('knowledge')), '| heldout:', len(load_jsonl('heldout')))"

echo "================================================================"
echo "[3/6] Tiny end-to-end pipeline smoke (offline)"
echo "================================================================"
"$PY" smoke_test.py

echo "================================================================"
echo "[4/6] Data-creation modules smoke (original scripts)"
echo "================================================================"
"$PY" data_pipeline/smoke_pipeline.py

echo "================================================================"
echo "[5/6] Build LOCAL dataset WITHOUT Zenodo (original generators)"
echo "================================================================"
"$PY" scripts/build_local_dataset.py
"$PY" -c "from data_utils import load_jsonl; print('local train:', len(load_jsonl('train', 'outputs/gaqa_v1_local')), '| heldout:', len(load_jsonl('heldout', 'outputs/gaqa_v1_local')))"

echo "================================================================"
echo "[6/6] Evaluation data — real BitNet b1.58 2B (small sample, CPU)"
echo "================================================================"
echo "-- heldout FROM ZENODO --"
"$PY" eval.py --split heldout --max-samples 6 --batch-size 2 --per-family --device cpu
echo "-- heldout FROM LOCAL BUILD (no Zenodo) --"
"$PY" eval.py --split heldout --max-samples 6 --batch-size 2 --per-family --device cpu --data-dir outputs/gaqa_v1_local

echo "================================================================"
echo "FULL TEST PASSED"
echo "================================================================"