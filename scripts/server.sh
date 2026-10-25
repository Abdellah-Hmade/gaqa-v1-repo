#! /usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

echo "=== STEP 0: git pull ==="
git pull origin experiment/gaqa-bitnet-assembly

echo ""
echo "=== STEP 1: setup ==="
REALVAL_DATA_DIR=data/realval bash scripts/realval_pipeline.sh --stage setup

echo ""
echo "=== STEP 2: download reads + references ==="
REALVAL_DATA_DIR=data/realval bash scripts/realval_pipeline.sh --stage download

echo ""
echo "=== STEP 3: generate conditions (subsample + spike) ==="
python3 scripts/realval_conditions.py \
    --data-dir data/realval \
    --runner "conda run -n realval"

echo ""
echo "=== STEP 4: assemble (longest: ~2-4h) ==="
REALVAL_DATA_DIR=data/realval bash scripts/realval_pipeline.sh --stage assemble

echo ""
echo "=== STEP 5: QC (QUAST + BUSCO + CheckM, ~1h) ==="
REALVAL_DATA_DIR=data/realval bash scripts/realval_pipeline.sh --stage qc

echo ""
echo "=== Pipeline complete ==="
echo "Now sync back: rsync qc_results/ + conditions.csv + data/gaqa_knowledge/"
