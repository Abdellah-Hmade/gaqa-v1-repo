#!/bin/bash
set -e

echo "=========================================="
echo "BitNet STE Training Pipeline"
echo "=========================================="

echo ""
echo "[Step 1/3] Training LoRA with STE quantization..."
python -m src.train_bitnet_ste

echo ""
echo "[Step 2/3] Packing base weights..."
python -m src.pack_ste_base

echo ""
echo "[Step 3/3] Evaluating packed model with STE-trained LoRA..."
python -m src.eval --model-type bitnet --packed --adapter models/bitnet-ste-adapter --benchmarks gaqa

echo ""
echo "=========================================="
echo "STE training pipeline complete!"
echo "=========================================="
