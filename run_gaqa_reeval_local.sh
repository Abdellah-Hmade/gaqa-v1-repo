#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source venv/bin/activate

export DATA_PROCESSED_DIR=./data/processed_gaqa
export EVAL_NUM_SAMPLES=2000
export SEED=42
export HF_CACHE_DIR=./models/hf_cache

RESULTS_DIR="./results"
EVAL_LOG_DIR="logs/evaluation"
ADAPTER_PATH="models-gaqa/bitnet-ste-adapter-seed42"

mkdir -p "$EVAL_LOG_DIR"

backup_results() {
    local tag="$1"
    local src_pred="$RESULTS_DIR/gaqa_scored_predictions_bitnet-seed42.json"
    local src_sum="$RESULTS_DIR/gaqa_summary_bitnet-seed42.json"
    local dst_pred="$RESULTS_DIR/gaqa_scored_predictions_${tag}-seed42.json"
    local dst_sum="$RESULTS_DIR/gaqa_summary_${tag}-seed42.json"

    if [ -f "$src_pred" ]; then
        cp "$src_pred" "$dst_pred"
        echo "  [backup] predictions -> $(basename $dst_pred)"
    fi
    if [ -f "$src_sum" ]; then
        cp "$src_sum" "$dst_sum"
        echo "  [backup] summary -> $(basename $dst_sum)"
    fi
}

echo "=============================================="
echo "  GAQA Re-evaluation (2,000 samples, RTX 5060)"
echo "=============================================="
echo ""

echo "▸ (1/2) BitNet zero-shot (packed)"
export EVAL_BATCH_SIZE=8
python -m src.eval \
    --model-type bitnet --packed \
    --benchmarks gaqa \
    --log-dir "$EVAL_LOG_DIR"
backup_results "bitnet-zero"
echo ""

echo "▸ (2/2) BitNet fine-tuned (packed + adapter)"
python -m src.eval \
    --model-type bitnet --packed \
    --adapter "$ADAPTER_PATH" \
    --benchmarks gaqa \
    --log-dir "$EVAL_LOG_DIR"
backup_results "bitnet-ft"
echo ""

echo "=============================================="
echo "  Evaluations complete"
echo "=============================================="
echo ""
echo "Eval logs:"
ls -la "$EVAL_LOG_DIR"/
echo ""
echo "Prediction/summary files:"
ls -la "$RESULTS_DIR"/gaqa_*.json
