#!/usr/bin/env bash
# Smoke test for all reproduction components (data, train, eval, benchmark entry).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== smoke: dataset loading (Zenodo download path) =="
python -c "from data_utils import load_jsonl; r=load_jsonl('knowledge'); print('knowledge test rows:', len(r))"

echo "== smoke: end-to-end pipeline (tiny model) =="
python smoke_test.py

echo "== smoke: data-creation pipeline (original modules: conditions/manifest/render) =="
python data_pipeline/smoke_pipeline.py

echo "== smoke: CLI wiring (train/eval/benchmark help) =="
python train.py --help >/dev/null && echo "train.py OK"
python eval.py --help >/dev/null && echo "eval.py OK"
python benchmark.py --help >/dev/null && echo "benchmark.py OK"

echo
echo "ALL SMOKE TESTS PASSED"