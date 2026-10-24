# GAQA v1 — Original Pipeline

This is a reconstructed version of the original work before ai usage, the ai usage was for automating the process, structuring the project and make it more of a cli customizable program for ease of use in servers.  

## Files

| File | Role |
|---|---|
| `constants.py` | bacterial list (12 organisms), model ids, LoRA/training/eval settings, env runner |
| `download.py` | resolve ENA reads + RefSeq references, write `manifest.csv`, download them |
| `qc.py` | subsample (C01–C11 conditions), assemble (SPAdes/MEGAHIT), run QUAST/BUSCO/CheckM |
| `data.py` | question answering: parse QC output, rule labels, render MCQs, build/load splits |
| `knowledge.py` | domain-knowledge questions: chunk sources, generate MCQs with Qwen, verify, split |
| `model.py` | STE quantization + LoRA layer + adapter apply |
| `train.py` | completion-only LoRA fine-tuning → `lora_weights.pt` |
| `eval.py` | next-token A/B/C/D accuracy (+ per-family); baselines via `MODEL` |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The assembly/QC tools (SPAdes, MEGAHIT, QUAST, BUSCO, CheckM, seqtk, seqkit)
come from a conda/micromamba environment; set its name in
`constants.ENV_RUNNER` (default `micromamba run -n realval`).


## License

Code: MIT. Dataset: CC BY 4.0.
