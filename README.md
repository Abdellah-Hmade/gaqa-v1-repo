# GAQA v1 — Reproduction Guide

Step-by-step reproduction of every quantitative result in **GAQA v1**
(Hmade & Guelzim): *"Fine-Tuning Ternary BitNet b1.58 for Genome Assembly
Quality Assessment"*. The outputs are the trained
LoRA adapter, console metrics, and a CSV results table.

| | |
|---|---|
| Dataset (GAQA v1) | Zenodo [10.5281/zenodo.21762025](https://doi.org/10.5281/zenodo.21762025) (auto-downloaded, CC BY 4.0) |
| Base model | `microsoft/bitnet-b1.58-2B-4T` (Hugging Face) |
| Method | LoRA (r=32, α=64, dropout 0.05) over frozen ternary layers; STE; **completion-only loss**; **logit-based forced-choice MCQ** |
| Paper hardware | Single A100 80 GB |

---

## Table of contents

1. [What you will reproduce](#what-you-will-reproduce)
2. [Quick start (7 steps)](#quick-start-7-steps)
3. [Step 1 — Install](#step-1--install)
4. [Step 2 — Configure (`.env`)](#step-2--configure-env)
5. [Step 3 — Verify the code](#step-3--verify-the-code)
6. [Step 4 — Get the data](#step-4--get-the-data)
7. [Step 5 — Fine-tune BitNet](#step-5--fine-tune-bitnet)
8. [Step 6 — Evaluate a single model](#step-6--evaluate-a-single-model)
9. [Step 7 — Full multi-model benchmark](#step-7--full-multi-model-benchmark)
10. [Expected numbers (paper tables)](#expected-numbers-paper-tables)
11. [Troubleshooting](#troubleshooting)
12. [Repo layout](#repo-layout)

---

## What you will reproduce

Each step below emits a number that appears in the paper. The mapping:

| Paper result | Command (step) | Expected number |
|---|---|---|
| Dataset statistics | `load_jsonl` counts (Step 4) | 11,043 train / 124 val / 114 heldout / 181 knowledge |
| Training dynamics | `python train.py` (Step 5) | 210 layers, 372M params (15.15%), 4,143 steps, loss 0.007 / val 0.040 |
| Eval — zero-shot BitNet | `python eval.py --split heldout` (Step 6) | 34.2% heldout, 68.5% knowledge |
| Eval — **BitNet v1** (ours) | `python eval.py --packed --adapter ...` (Step 6) | **69.3%** heldout, **84.0%** knowledge |
| Eval — baselines | `python eval.py --model ...` (Step 6) | Mistral 39.5% / 84.5%; Qwen 54.4% / 94.5% |
| Per-family + AUC | `python eval.py --per-family` (Step 6) | tool_sel 100%, qc AUC 0.959, remediation 39.5% |
| Full comparison table | `python benchmark.py` (Step 7) | `outputs/multi_model_comparison.csv` |

---

## Quick start (7 steps)

```bash
# 1. Install
cd gaqa-v1-repro && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env          # edit if you have a GPU / HF token

# 3. Verify (fast, no GPU)
bash scripts/test.sh

# 4. Get the data (auto-download from Zenodo on first run)
python train.py --max-steps 2 --max-train-samples 8 --batch-size 1 --device cpu

# 5. Fine-tune (paper numbers — A100 80 GB, ~35 min)
python train.py

# 6. Evaluate (packed model → the paper's exact 69.3% / 84.0%)
python eval.py --split heldout --packed --packed-weights packed_weights.pt \
    --adapter outputs/gaqa-v1-lora
python eval.py --split knowledge --packed --packed-weights packed_weights.pt \
    --adapter outputs/gaqa-v1-lora

# 7. Full benchmark table
python benchmark.py --adapter outputs/gaqa-v1-lora/lora_weights.pt
```

Steps 1–4 work on any machine. Steps 5–7 need the hardware listed in
[Step 5](#step-5--fine-tune-bitnet).

---

## Step 1 — Install

**Python** 3.10+ (tested 3.10). **Hardware**:

| Task | Minimum | Recommended (paper) |
|---|---|---|
| Smoke tests / quick sanity | CPU | any GPU |
| Fine-tune (3 epochs) | 16 GB GPU* | **A100 80 GB** (~35 min) |
| BitNet eval (114 + 181) | 16 GB GPU (or CPU, slow) | A100 80 GB |
| 7B baselines (Mistral/Qwen) | ≥16 GB GPU | A100 80 GB |

\* The full config (2B bf16 model + 372M-trainable LoRA + AdamW8bit fp32 master
weights) exceeds 8 GB VRAM — verified OOM. See [Troubleshooting](#troubleshooting).

```bash
cd gaqa-v1-repro
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with torch 2.14 + transformers 5.17 (torch ≥2.3, transformers ≥4.47 work).

> If `pip install` fails with a `Read timed out` while downloading the torch
> wheel (large download, flaky networks), just re-run it — or use
> `pip install --default-timeout=200 -r requirements.txt`.

**Rebuilding the dataset from raw reads** ([Step 4.3](#43-rebuild-from-raw-sequencing-reads))
additionally needs the `data_pipeline/` toolchain (conda/micromamba): seqtk,
SPAdES, MEGAHIT, QUAST, BUSCO, CheckM, Flye (see `data_pipeline/env/realval.yml`).

---

## Step 2 — Configure (`.env`)

All parameters live in **`.env`** (template: `.env.example`) — edit without
touching code. CLI flags override `.env`; environment variables override both.

| Variable | Default | Meaning |
|---|---|---|
| `TEST_MODE` | `smoke` | `smoke` (fast, no GPU) or `full` (venv → data → real eval) for `scripts/test.sh` |
| `DATA_SOURCE` | `zenodo` | `zenodo` \| `local` \| `rebuild` — where `train.py`/`eval.py`/`benchmark.py` read data |
| `ZENODO_RECORD` | `21762025` | Dataset release on Zenodo |
| `GAQA_DATA_DIR` | *(empty)* | Local data dir (used when `DATA_SOURCE=local`) |
| `REBUILD_DATA_DIR`, `REBUILD_HELDOUT_DIR`, `REBUILD_KNOWLEDGE_DIR` | `data_pipeline/data/...` | Rebuilt-dataset locations (used when `DATA_SOURCE=rebuild`) |
| `BITNET_MODEL` | `microsoft/bitnet-b1.58-2B-4T` | Base model |
| `MISTRAL_MODEL`, `QWEN_MODEL` | Mistral-7B / Qwen2.5-7B | Baselines |
| `HF_TOKEN` | *(empty)* | HF token, **required for gated models (Mistral)** |
| `SEED` | `42` | Random seed |
| `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT` | `32`, `64`, `0.05` | LoRA config |
| `NUM_EPOCHS`, `BATCH_SIZE`, `GRAD_ACCUM`, `MAX_LENGTH` | `3`, `8`, `1`, `1024` | Training schedule |
| `LEARNING_RATE`, `WARMUP_RATIO`, `WEIGHT_DECAY`, `GRAD_CLIP` | `0.0002`, `0.05`, `0.01`, `1.0` | Optimizer |
| `MAX_TRAIN_SAMPLES`, `MAX_STEPS` | `0`, `0` | `0` = full run; set for quick checks |
| `EVAL_SAMPLES` | `256` | Val samples during training eval |
| `EVAL_BATCH_SIZE`, `DEVICE` | `16`, `cuda` | Evaluation |
| `OUTPUT_DIR`, `BENCHMARK_OUTPUT` | `outputs/...` | Output paths |
| `REALVAL_DATA_DIR`, `REALVAL_JOBS` | `data/realval`, `12` | Data-pipeline dir / parallel jobs |
| `KNOWLEDGE_ENDPOINT`, `JITTER_COPIES` | `http://localhost:8000/v1`, `60` | Knowledge gen / jitter |

---

## Step 3 — Verify the code

Fast, no GPU, no 2B download:

```bash
bash scripts/test.sh            # uses TEST_MODE from .env (default: smoke)
TEST_MODE=full bash scripts/test.sh   # full: venv -> data -> real-model eval
```

**smoke** (`scripts/smoke_test.sh`): Zenodo download check (181 knowledge rows)
→ tiny end-to-end pipeline (STE/LoRA → completion-only training → MCQ eval →
AUC → adapter save/load) → data-creation modules (22 checks) → CLI wiring.

**full** (`scripts/full_test.sh`): fresh `.venv-test`, installs
`requirements.txt`, downloads data from Zenodo, builds a small local dataset
**without** Zenodo, and produces evaluation data on the real BitNet b1.58 2B
model (small sample, CPU-safe) for both data sources.

---

## Step 4 — Get the data

The dataset is available three ways, selected with `DATA_SOURCE` in `.env`.
Pick the one that fits your setup — they all produce the same GAQA v1 dataset.

### 4.1 Download from Zenodo (recommended)

The released dataset (CC BY 4.0) is auto-downloaded on first use — no extra
setup, no bioinformatics tools.

```bash
# .env
DATA_SOURCE=zenodo
```

```bash
python train.py --max-steps 20   # first run downloads into data_cache/
```

> **On an 8 GB GPU** the command above still OOMs (it uses the default
> `DEVICE=cuda`). To just download the data and sanity-check the code on a small
> GPU, add `--device cpu`. Alternatively run the count check at the end of this
> step, which needs no GPU at all.

### 4.2 Use your own local copy

If you already have the released files (or are on an offline machine), point the
code at the directory containing `gaqa_v1_*.jsonl`. To obtain those files:
download once from Zenodo (Step 4.1 leaves them in `data_cache/`), or generate
a small synthetic set with `python scripts/build_local_dataset.py` (writes
`outputs/gaqa_v1_local/`; all four splits, for offline smoke-testing).

```bash
# .env
DATA_SOURCE=local
GAQA_DATA_DIR=/path/to/release/data
```

```bash
python train.py
```

### 4.3 Rebuild from raw sequencing reads

Recreate the dataset from scratch using the `data_pipeline/` toolchain. This
needs the bioinformatics environment (micromamba; see
[`data_pipeline/README.md`](data_pipeline/README.md)) plus network access to
ENA/RefSeq.

```bash
# one-shot (see data_pipeline/README.md for prerequisites)
bash scripts/rebuild_data.sh
```

Then point the code at the rebuilt files:

```bash
# .env
DATA_SOURCE=rebuild
```

The stage-by-stage steps — and the C01–C11 condition matrix — are documented in
[`data_pipeline/README.md`](data_pipeline/README.md). In short:

```bash
# Verify the modules (no bioinformatics tools needed)
python data_pipeline/smoke_pipeline.py

# Full rebuild environment
micromamba create -f data_pipeline/env/realval.yml && micromamba activate realval

# Manifest (12 organisms) + download reads/references
python data_pipeline/src/build_v3_manifest.py
REALVAL_DATA_DIR=data_pipeline/data/realval_v3 \
  bash data_pipeline/scripts/realval_pipeline.sh --stage download

# Condition matrix (C01–C11) + assemble + QC
python data_pipeline/scripts/realval_v3_conditions.py \
  --data-dir data_pipeline/data/realval_v3 --runner "micromamba run -n realval"
REALVAL_DATA_DIR=data_pipeline/data/realval_v3 \
  bash data_pipeline/scripts/realval_pipeline.sh --stage assemble --stage qc

# Build the GAQA dataset (label + render + jitter + splits), from data_pipeline/:
cd data_pipeline && python -m src.build_gaqa_dataset_v3 --data-dir data/realval_v3 --include-knowledge
```

> The Zenodo release is the authoritative artifact for the paper's numbers; the
> pipeline reproduces it. This stage needs the bioinformatics toolchain and
> 30–50 GB of free disk — see [`data_pipeline/README.md`](data_pipeline/README.md).

### Verify the split counts

Whichever source you chose, confirm you have the expected splits. The dataset
has **11,043 train** (4,740 qc_diagnosis + 4,740 remediation + 159 tool_selection
+ 1,404 knowledge), **124 val**, **114 heldout** (38 per family), and
**181 knowledge test** rows:

```bash
python -c "from data_utils import load_jsonl; \
  print({s: len(load_jsonl(s)) for s in ['train','val','heldout','knowledge']})"
# -> {'train': 11043, 'val': 124, 'heldout': 114, 'knowledge': 181}
```

---

## Step 5 — Fine-tune BitNet

```bash
# Full run (paper numbers; A100 80 GB)
python train.py

# Smaller GPU / quick sanity
python train.py --max-steps 20                                   # code-path check (≥16 GB GPU)
python train.py --batch-size 2 --grad-accum 4 --max-train-samples 1000
python train.py --max-steps 2 --max-train-samples 8 --batch-size 1 --device cpu
```

> **Note on hardware:** the full 3-epoch run needs a ≥16 GB GPU (A100 80 GB for
> the paper's ~35 min). On an 8 GB GPU the default `DEVICE=cuda` OOMs, and the
> CPU sanity line is **very slow** (~10–15 min per 1024-token step) — use it
> only to confirm the code path, not to produce the adapter. In practice the
> trained adapter (needed by Steps 6–7) is only produced on a ≥16 GB GPU.

**Expected output** (paper §"Training Dynamics") — printed before/at end:

- **210** replaced layers
- **372,029,440 / 2,456,074,240 trainable (15.15%)**
- **4,143** optimizer steps (3 epochs × 1,381 @ batch 8)
- final train loss ≈ **0.007**, validation loss ≈ **0.040**, token accuracy > **99%**
- packed model ≈ **1.22 GB** VRAM

Adapter → `outputs/gaqa-v1-lora/lora_weights.pt`.

---

## Step 6 — Evaluate a single model

```bash
# zero-shot BitNet (bf16) → ~34.2%
python eval.py --split heldout

# fine-tuned BitNet v1 — packed model (paper's exact 69.3% / 84.0%)
python eval.py --split heldout --packed --packed-weights packed_weights.pt \
    --adapter /path/to/adapter_dir
python eval.py --split knowledge --packed --packed-weights packed_weights.pt \
    --adapter /path/to/adapter_dir

# baselines
python eval.py --split heldout --model Qwen/Qwen2.5-7B-Instruct   # → ~54.4%
```

- **Packed vs bf16:** the paper's 69.3% is measured on the **packed (offline-ternary)
  model** (~1.22 GB). Use `--packed` with `--packed-weights` (the packed ternary base)
  and `--adapter` (the adapter *directory* containing `lora_weights.pt`) to reproduce it
  exactly. Loading the bf16 model with `--adapter` alone (no `--packed`) gives a higher
  number (~81.6%) because online quantization is more accurate than the packed deployment.
- The `--adapter` lines require the adapter from [Step 5](#step-5--fine-tune-bitnet).
- The zero-shot `--split heldout` line runs on an 8 GB GPU (≈5.5 GB peak); on
  CPU add `--device cpu`.
- The Qwen/Mistral lines download a ~15 GB model and need a **≥16 GB GPU**;
  Mistral additionally requires `HF_TOKEN` (gated model).
- `--per-family` adds per-task accuracy + qc_diagnosis AUC (tool_sel **100%**,
  qc_diagnosis AUC **0.959**, remediation **39.5%**).
- `--max-samples N` caps the subset for fast sanity runs (full splits are
  114 / 181).

## Step 7 — Full multi-model benchmark

```bash
python benchmark.py --adapter outputs/gaqa-v1-lora/lora_weights.pt
python benchmark.py --adapter ... --models bitnet,zeroshot,qwen   # skip Mistral
```

Writes `outputs/multi_model_comparison.csv` — the paper's headline table.

> The default (`--models all`) also loads Mistral + Qwen, which need a ≥16 GB
> GPU and `HF_TOKEN`. On smaller hardware use `--models bitnet,zeroshot`. If one
> model fails, the others are still written to the CSV (the failed model is
> skipped with a message).

---

## Expected numbers (paper tables)

**Headline results** (`multi_model_comparison.csv`):

| model | v3-heldout | domain-knowledge |
|---|---|---|
| BitNet-2B-zeroshot | 34.2 | 68.5 |
| **BitNet-2B-v1** | **69.3** | **84.0** |
| Mistral-7B-Instruct-v0.3 | 39.5 | 84.5 |
| Qwen2.5-7B-Instruct | 54.4 | 94.5 |

**Per-family heldout (BitNet-2B-v1)**: tool_selection **100%**, qc_diagnosis
AUC **0.959**, remediation **39.5%**. Deltas: +35.1 pp over zero-shot heldout,
+15.5 pp over zero-shot knowledge; Qwen beat by 14.9 pp, Mistral by 29.8 pp.

**Training**: 4,143 steps, loss 0.007 / val 0.040, token acc >99%, 372M trainable
(15.15%), packed model 1.22 GB VRAM.

**Dataset (GAQA v1)**: 11,043 train (4,740 qc_diagnosis + 4,740 remediation +
159 tool_selection + 1,404 knowledge), 124 val, 114 heldout, 181 knowledge test.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ImportError: ... requires accelerate` | `accelerate` missing (transformers ≥5) → `pip install -r requirements.txt` |
| `pip install` `Read timed out` | Flaky network on the torch wheel → re-run, or `pip install --default-timeout=200 -r requirements.txt` |
| `DATA_SOURCE=local` but no `GAQA_DATA_DIR` | Set `GAQA_DATA_DIR` in `.env` to the folder with the released `gaqa_v1_*.jsonl` |
| Rebuilt data not found (`DATA_SOURCE=rebuild`) | Run `bash scripts/rebuild_data.sh` first, or point the `REBUILD_*_DIR` vars at the builder output |
| CUDA OOM during `train.py` (8 GB GPU) | Full config needs ~80 GB. Use `--max-steps N` sanity runs, or lower `LORA_R` in `.env`, or run on CPU |
| Mistral 401 / access error | Gated model → set `HF_TOKEN` in `.env` or `huggingface-cli login` |
| Very slow on CPU | Normal (2B bf16 + STE). Use small `--max-samples`/`--max-steps` for checks |
| `qc_auc: NaN` in eval | Too few qc_diagnosis samples (AUC undefined) — run the full 114-split on GPU |
| First `load_jsonl` slow | Zenodo download (~13 MB) on first use → cached in `data_cache/` |
| 7B baselines OOM on GPU | Need ≥16 GB; run fewer models (`--models ...`) or on CPU |
| `micromamba create` hangs with no error | `checkm-genome` post-link downloads the CheckM DB from Zenodo (can 504). Pre-download it and set `CHECKM_DATA_PATH` — see `data_pipeline/README.md` |
| `realval_v3_conditions.py` → `FileNotFoundError: manifest.csv` | Point `--data-dir` at `data_pipeline/data/realval_v3` (the dir `build_v3_manifest.py` writes to) |

---

## Repo layout

```
.env, .env.example   all configurable parameters
config.py            reads .env / env vars (paper defaults)
data_utils.py        Zenodo download + JSONL loading + gold/option parsing
bitnet_lora.py       STE quantization, LoRAAutoBitLinear, adapter apply
packed_model.py      packed (offline-ternary) BitNet + LoRA loading
train.py             LoRA + STE fine-tuning (completion-only loss)
eval.py              logit-based MCQ evaluation (+ per-family, qc AUC)
benchmark.py         multi-model comparison → CSV
smoke_test.py        tiny-model end-to-end pipeline check
data_pipeline/       original dataset-construction scripts (realval_pipeline,
                     conditions, builder, manifest, knowledge, env, smoke)
scripts/smoke_test.sh
scripts/full_test.sh
scripts/test.sh            # TEST_MODE=smoke|full
scripts/rebuild_data.sh    # rebuild dataset from raw reads (DATA_SOURCE=rebuild)
scripts/build_local_dataset.py
```

## License

Code: MIT. Dataset: CC BY 4.0 (Zenodo 10.5281/zenodo.21762025).
