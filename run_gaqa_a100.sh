#!/usr/bin/env bash

set -euo pipefail

RUN_START=$(date +%s)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

LOGDIR="logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/gaqa_a100_run.log"

exec > >(tee -a "${LOGFILE}") 2>&1

echo ""
echo "========================================================================"
echo " GAQA A100 Run — $(date)"
echo " Host: $(hostname)"
echo " PWD:  ${SCRIPT_DIR}"
echo "========================================================================"
echo ""

if [ -d "venv/bin" ]; then
    echo "[PREFLIGHT] Activating venv..."
    source venv/bin/activate
    echo "  Python: $(python --version)"
    echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
else
    echo "[PREFLIGHT] No venv found — creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "  Created. Python: $(python --version)"

    echo "[PREFLIGHT] Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install bitsandbytes  # for optional adamw_8bit optimizer patch

    echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
    echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
fi

set -a
source .env.gaqa
set +a

: "${SEED:?}.env.gaqa must define SEED"
: "${DATA_PROCESSED_DIR:?}.env.gaqa must define DATA_PROCESSED_DIR"
: "${BATCH_SIZE:?}.env.gaqa must define BATCH_SIZE"

echo ""
echo "[CONFIG] SEED=${SEED}  BATCH_SIZE=${BATCH_SIZE}  GRAD_ACCUM=${GRAD_ACCUM}"
echo "         MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES}  MAX_LENGTH=${MAX_LENGTH}"
echo "         BITNET_NUM_EPOCHS=${BITNET_NUM_EPOCHS}  BITNET_LORA_R=${BITNET_LORA_R}"
echo ""

ADAPTER_DIR="models/bitnet-ste-adapter-seed${SEED}"
PACKED_DIR="models/bitnet-packed-ste"
TRAIN_LOG_DIR="logs/training"
EVAL_LOG_DIR="logs/evaluation"
DONE_MARKER="logs/gaqa_a100_DONE"

mkdir -p "${TRAIN_LOG_DIR}" "${EVAL_LOG_DIR}" "${PACKED_DIR}"

echo "══════════════════════════════════════════════════════════════════════════"
echo " [PREFLIGHT] GPU check"
echo "══════════════════════════════════════════════════════════════════════════"
nvidia-smi || {
    echo "FATAL: nvidia-smi failed. Ensure CUDA is available."
    exit 1
}
echo ""

echo "══════════════════════════════════════════════════════════════════════════"
echo " [PREFLIGHT] Data check"
echo "══════════════════════════════════════════════════════════════════════════"
for f in train.jsonl val.jsonl test.jsonl; do
    path="${DATA_PROCESSED_DIR}/${f}"
    if [ ! -f "${path}" ]; then
        echo "FATAL: Missing data file ${path}"
        echo "  Run the preprocessing phase first, or rsync the processed data."
        exit 1
    fi
    echo "  OK: ${path} ($(wc -l < "${path}") lines)"
done
echo ""

echo "══════════════════════════════════════════════════════════════════════════"
echo " [PREFLIGHT] Model cache check"
echo "══════════════════════════════════════════════════════════════════════════"
BITNET_CACHED=false

if [ -d "${HF_CACHE_DIR}" ]; then
    if ls "${HF_CACHE_DIR}/hub/models--microsoft--bitnet-b1.58-2B-4T-bf16"*/snapshots/*/ 1>/dev/null 2>&1; then
        BITNET_CACHED=true
    fi
fi

echo "  BitNet b1.58 2B:    $([ "${BITNET_CACHED}" = true ] && echo 'CACHED' || echo 'NOT CACHED — will download (~2.1 GB)')"
echo ""

echo "══════════════════════════════════════════════════════════════════════════"
echo " STEP 1/4: BitNet STE LoRA fine-tuning (the ONLY training step)"
echo "══════════════════════════════════════════════════════════════════════════"

TRAIN_DONE="${ADAPTER_DIR}/lora_weights.pt"

if [ -f "${TRAIN_DONE}" ]; then
    echo "  [SKIP] ${TRAIN_DONE} exists — training already complete."
    echo "  Remove this file to re-run training."
else
    echo "  Output adapter: ${ADAPTER_DIR}"
    echo "  Log dir:        ${TRAIN_LOG_DIR}"
    echo "  Args:           $@"
    echo ""

    TRAIN_ARGS=("$@")

    python -m src.train_bitnet_ste \
        --log-dir "${TRAIN_LOG_DIR}" \
        "${TRAIN_ARGS[@]}"

    if [ -f "${TRAIN_DONE}" ]; then
        echo "  ✅ Training complete: ${TRAIN_DONE}"
    else
        echo "  ⚠️  Training finished but ${TRAIN_DONE} not found."
        echo "     Check logs in ${TRAIN_LOG_DIR}"
    fi
fi
echo ""

echo "══════════════════════════════════════════════════════════════════════════"
echo " STEP 2/4: Pack ternary base weights"
echo "══════════════════════════════════════════════════════════════════════════"

PACK_DONE="${PACKED_DIR}/packed_weights.pt"

if [ -f "${PACK_DONE}" ]; then
    echo "  [SKIP] ${PACK_DONE} exists — packing already done."
else
    echo "  Output: ${PACKED_DIR}/"
    python -m src.pack_ste_base --output "${PACKED_DIR}"
    echo "  ✅ Pack complete: ${PACK_DONE}"
fi
echo ""

echo "══════════════════════════════════════════════════════════════════════════"
echo " STEP 3/4: BitNet zero-shot eval (packed, no adapter)"
echo "══════════════════════════════════════════════════════════════════════════"

BITNET_ZERO_LOG="${EVAL_LOG_DIR}/bitnet-2b_packed-seed${SEED}_zero.json"

if [ -f "${BITNET_ZERO_LOG}" ]; then
    echo "  [SKIP] ${BITNET_ZERO_LOG} exists."
else
    python -m src.eval \
        --model-type bitnet \
        --packed \
        --benchmarks gaqa \
        --log-dir "${EVAL_LOG_DIR}"
    echo "  ✅ BitNet zero-shot eval done"
fi
echo ""

echo "══════════════════════════════════════════════════════════════════════════"
echo " STEP 4/4: BitNet fine-tuned eval (packed + LoRA adapter)"
echo "══════════════════════════════════════════════════════════════════════════"

BITNET_FT_LOG="${EVAL_LOG_DIR}/bitnet-2b_packed-seed${SEED}_ft.json"

if [ -f "${BITNET_FT_LOG}" ]; then
    echo "  [SKIP] ${BITNET_FT_LOG} exists."
else
    if [ ! -d "${ADAPTER_DIR}" ]; then
        echo "  WARNING: Adapter dir ${ADAPTER_DIR} not found."
        echo "  The fine-tuned eval will likely fail without trained weights."
        echo "  If training was skipped (restored from cache), this is expected."
    fi
    python -m src.eval \
        --model-type bitnet \
        --packed \
        --adapter "${ADAPTER_DIR}" \
        --benchmarks gaqa \
        --log-dir "${EVAL_LOG_DIR}"
    echo "  ✅ BitNet fine-tuned eval done"
fi
echo ""

echo "========================================================================"
echo " GAQA A100 Run — COMPLETE"
echo "========================================================================"

touch "${DONE_MARKER}"

RUN_END=$(date +%s)
DURATION=$((RUN_END - RUN_START))
echo "  Total wall time: $((DURATION / 60)) min $((DURATION % 60)) sec"
echo ""

echo "────────────────────────────────────────────────────────────────────────"
echo "  Produced files:"
echo "────────────────────────────────────────────────────────────────────────"
echo "  Training logs:       ${TRAIN_LOG_DIR}/"
ls -1 "${TRAIN_LOG_DIR}"/*.json 2>/dev/null | sed 's/^/    /' || echo "    (none)"
echo ""
echo "  Adapter:             ${ADAPTER_DIR}/"
ls -1 "${ADAPTER_DIR}"/*.pt 2>/dev/null | sed 's/^/    /' || echo "    (none)"
echo ""
echo "  Packed base:         ${PACKED_DIR}/"
ls -1 "${PACKED_DIR}"/*.pt 2>/dev/null | sed 's/^/    /' || echo "    (none)"
echo ""
echo "  Eval logs:           ${EVAL_LOG_DIR}/"
for f in "${EVAL_LOG_DIR}"/*.json; do
    [ -f "${f}" ] && echo "    $(basename "${f}")"
done
echo ""
echo "  Run log:             ${LOGFILE}"
echo "  Done marker:         ${DONE_MARKER}"
echo ""
