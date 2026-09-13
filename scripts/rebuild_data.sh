#!/usr/bin/env bash
# Rebuild the GAQA v1 dataset from raw sequencing reads using the original
# data_pipeline scripts. After a successful rebuild, set DATA_SOURCE=rebuild
# in .env so train.py / eval.py / benchmark.py read the rebuilt dataset.
#
# Prerequisites: micromamba env from data_pipeline/env/realval.yml (seqtk,
# SPAdES, MEGAHIT, QUAST, BUSCO, CheckM, Flye) + ENA network access.
# The knowledge stage additionally needs a vLLM endpoint serving
# Qwen2.5-32B-Instruct (set KNOWLEDGE_ENDPOINT in .env).
#
# Flow (single data dir: data_pipeline/data/realval_v3/):
#   1. manifest   build_v3_manifest.py            -> manifest.csv + manifest_pipeline.csv
#   2. download   realval_pipeline.sh --stage download   -> reads/ + references/
#   3. conditions realval_v3_conditions.py        -> conditions.csv + subsampled/
#   4. assemble+qc realval_pipeline.sh --stage assemble --stage qc -> qc_results/
#   5. knowledge  chunk + generate (optional, needs vLLM)
#   6. build      build_gaqa_dataset_v3.py        -> processed_gaqa_v3 + heldout
#
# Usage: bash scripts/rebuild_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
DP="data_pipeline"
V3="$DP/data/realval_v3"
RUNNER="${REALVAL_RUNNER:-micromamba run -n realval}"

echo "=================================================================="
echo "Rebuilding GAQA v1 from raw reads (original data_pipeline scripts)"
echo "  data dir: $V3"
echo "=================================================================="

echo "[1/6] Organism manifest (12 organisms) -> manifest.csv + manifest_pipeline.csv"
"$PY" "$DP/src/build_v3_manifest.py"

echo "[2/6] Download reads + references (ENA/RefSeq)"
REALVAL_DATA_DIR="$V3" bash "$DP/scripts/realval_pipeline.sh" --stage download

echo "[3/6] Condition matrix (C01-C11 per organism) -> conditions.csv + subsampled/"
"$PY" "$DP/scripts/realval_v3_conditions.py" --data-dir "$V3" --runner "$RUNNER"

echo "[4/6] Assemble + QC (SPAdes/MEGAHIT -> QUAST/BUSCO/CheckM)"
REALVAL_DATA_DIR="$V3" bash "$DP/scripts/realval_pipeline.sh" --stage assemble --stage qc

echo "[5/6] Knowledge questions (needs KNOWLEDGE_ENDPOINT / vLLM Qwen2.5-32B)"
"$PY" -c "import sys; sys.path.insert(0,'$DP'); from src import generate_knowledge_qa as g; print('knowledge module OK', g.__file__)"
echo "      (run src.chunk_knowledge_sources + src.generate_knowledge_qa manually if not done)"

echo "[6/6] Build GAQA dataset (label + render + jitter + splits)"
(cd "$DP" && "$PY" -m src.build_gaqa_dataset_v3 --data-dir data/realval_v3 --include-knowledge)

echo "=================================================================="
echo "Rebuild done. To use the rebuilt dataset:"
echo "  - edit .env: DATA_SOURCE=rebuild"
echo "  - then run: python train.py / python eval.py ... / python benchmark.py ..."
echo "Expected outputs: data_pipeline/data/processed_gaqa_v3/{train,val}.jsonl,"
echo "  data_pipeline/data/realval_v3_heldout/test.jsonl,"
echo "  data_pipeline/data/gaqa_knowledge/test.jsonl"
echo "=================================================================="