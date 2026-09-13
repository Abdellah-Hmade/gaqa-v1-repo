# GAQA Real-Data External Validation — Server Run Manual

## Overview

This package creates a **real-data external validation set** for the GAQA article.
Real sequencing reads + known references → real assemblies with known injected failure modes
→ real QUAST/BUSCO/CheckM metrics → GAQA-format questions with labels exact by construction.

The pipeline is designed to run on the A100 server (CPU-only until final eval stage),
takes ≤5 hours wall time, and is safe to run concurrently with v2 GPU training.

**Paper framing:** "Trained on synthetic GAQA, validated on real assemblies with known failure modes."

## Files to Commit/Push (from laptop)

After generation on the laptop, commit and push these files:

```
data/realval/manifest.csv     # Experiment plan with verified URLs
env/realval.yml               # Conda environment specification
scripts/realval_pipeline.sh   # Main pipeline script (executable)
src/build_realval_dataset.py  # Question renderer (reuses v2 template style)
docs/gaqa_realval_run.md      # This manual
```

Do NOT commit `data/realval/` downloaded data, `results/realval/`, or any generated files.
These are produced on the server.

## On the Server

### 1. Clone / Pull

```bash
# Navigate to the repo (adjust path as needed)
cd /path/to/repo

# Ensure correct branch
git checkout experiment/gaqa-bitnet-assembly

# Pull latest changes
git pull origin experiment/gaqa-bitnet-assembly
```

### 2. Start a tmux Session (recommended for long-running jobs)

```bash
tmux new -s realval
```

### 3. Run the Full Pipeline

```bash
# Make the pipeline script executable
chmod +x scripts/realval_pipeline.sh

# Run all stages (CPU-only, ~5h total)
bash scripts/realval_pipeline.sh
```

This will:
1. Bootstrap micromamba into `~/micromamba` or `tools/micromamba` if conda is not available
2. Create the realval conda environment
3. Download sequencing reads and references from ENA/NCBI
4. Subsample reads to 10×, 30×, 100× coverage and create contamination spikes
5. Run SPAdes assemblies in parallel (configurable via `REALVAL_JOBS`, default 6)
6. Run QUAST, BUSCO, and CheckM quality assessment
7. Render GAQA-format questions
8. Generate a summary report

### 4. Detach from tmux (optional)

```
Ctrl+B, then D
```

To reattach later:
```bash
tmux attach -t realval
```

### 5. Run Evaluation (GPU, optional — do after training completes)

Once you have a trained adapter model, run the eval stage:

```bash
ADAPTER_DIR=models/bitnet-ste-adapter-v2-r8-seed42 \
BITNET_LORA_R=8 \
BITNET_LORA_ALPHA=16 \
bash scripts/realval_pipeline.sh --stage eval --stage report
```

For the r32 adapter:
```bash
ADAPTER_DIR=models/bitnet-ste-adapter-v2-r32-seed42 \
BITNET_LORA_R=32 \
BITNET_LORA_ALPHA=64 \
bash scripts/realval_pipeline.sh --stage eval --stage report
```

The eval uses the fixed logit scorer: `python -m src.eval --model-type bitnet --packed ...`
with `DATA_PROCESSED_DIR` automatically pointed at `data/realval/`.

### 6. Re-running a Single Stage

If a stage fails or you need to re-run it, delete its `.done` marker first:

```bash
# Delete the marker for the stage you want to re-run
rm data/realval/.state/download.done

# Re-run just that stage
bash scripts/realval_pipeline.sh --stage download
```

To re-run from a specific point forward:
```bash
# Wipe all state
rm -rf data/realval/.state/

# Run everything
bash scripts/realval_pipeline.sh
```

## Retrieving Results (from laptop)

After the pipeline completes, rsync results back to your laptop:

```bash
# From laptop (replace user@a100-host with actual credentials)
rsync -avz user@a100-host:/path/to/repo/data/realval/ ./data/realval/
rsync -avz user@a100-host:/path/to/repo/results/realval/ ./results/realval/
```

Or selectively:
```bash
# Just the test questions and report
rsync -avz user@a100-host:/path/to/repo/data/realval/test.jsonl ./data/realval/
rsync -avz user@a100-host:/path/to/repo/results/realval/REPORT.md ./results/realval/
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REALVAL_JOBS` | `6` | Max parallel assembly jobs (each uses -t 4 threads) |
| `REALVAL_ENV_PREFIX` | `realval` | Conda/micromamba environment name |
| `ADAPTER_DIR` | (unset) | Path to trained LoRA adapter for eval stage |
| `BITNET_LORA_R` | `8` | LoRA rank (must match adapter) |
| `BITNET_LORA_ALPHA` | `16` | LoRA alpha (must match adapter) |

## Notes

- The pipeline is **CPU-only** until the eval stage. Safe to run alongside GPU training.
- **Disk budget:** ~20 GB. The script downloads reads (~3-5 GB per organism), assemblies (~1 GB),
  and QC databases (BUSCO lineage ~500 MB, CheckM DB ~1.5 GB).
- **Idempotent:** Each stage creates a `.done` marker. Re-running skips completed stages.
- **Resilient:** Single assembly failures are logged but do not kill the whole pipeline.
  Low-coverage (10×) assemblies may fail — these are retried once with broader k-mer settings.
- **CheckM is optional:** If the CheckM database setup fails, the pipeline continues with
  a warning and CheckM metrics are omitted from questions.

## Troubleshooting

**"micromamba: command not found"**
The bootstrapper should handle this. If not, install manually:
```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
export PATH="$PWD/bin:$PATH"
```

**"quast.py: command not found"**
The conda environment may not be fully activated. Try:
```bash
micromamba activate realval
which quast.py
```

**Assembly failures at 10× coverage**
Low-coverage assemblies may fail with SPAdes. This is expected behavior.
The pipeline logs the failure and continues. If you want to retry manually:
```bash
conda_run spades.py -t 4 -o /tmp/retry -k 21,33,55 \
  --pe1-1 <reads_1.fastq.gz> --pe1-2 <reads_2.fastq.gz>
```

**Disk space issues**
If you run out of disk (~20 GB budget), you can:
1. Remove intermediate files: `rm -rf data/realval/reads/*.fastq.gz`
2. Compress assembly dirs: `find data/realval/assemblies -name "*.fasta" -exec gzip {} \;`

## File Layout on Server

```
repo/
├── data/realval/
│   ├── manifest.csv            # Experiment plan (URL-verified)
│   ├── conditions.csv          # Generated conditions table
│   ├── test.jsonl              # GAQA-format test questions (output)
│   ├── .state/                 # Stage done markers + job logs
│   ├── reads/                  # Downloaded raw reads
│   ├── subsampled/             # Downsampled reads + contamination spikes
│   ├── assemblies/             # SPAdes assembly outputs
│   ├── qc_results/             # QUAST/BUSCO/CheckM outputs
│   ├── references/             # Downloaded reference genomes
│   ├── checkm_data/            # CheckM database
│   └── busco_downloads/        # BUSCO lineage data
├── results/realval/
│   └── REPORT.md               # Aggregated report
├── env/realval.yml             # Conda environment
├── scripts/realval_pipeline.sh # Pipeline script
├── src/build_realval_dataset.py# Question renderer
└── docs/gaqa_realval_run.md    # This manual
```
