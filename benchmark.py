"""Multi-model benchmark: reproduce the paper's comparison table.

Runs the logit-based MCQ evaluation of four models on both external test sets
(organism-heldout and domain-knowledge) and prints/writes the table:

    model, test_set, accuracy, n_samples, qc_auc
    BitNet-2B-zeroshot,  v3-heldout,        34.2, 114
    BitNet-2B-zeroshot,  domain-knowledge,  68.5, 181
    BitNet-2B-v1,        v3-heldout,        69.3, 114
    BitNet-2B-v1,        domain-knowledge,  84.0, 181
    Mistral-7B-Instruct-v0.3, v3-heldout,   39.5, 114
    Mistral-7B-Instruct-v0.3, domain-knowledge, 84.5, 181
    Qwen2.5-7B-Instruct, v3-heldout,        54.4, 114
    Qwen2.5-7B-Instruct, domain-knowledge,  94.5, 181

Usage:
    python benchmark.py --adapter outputs/gaqa-v1-lora/lora_weights.pt
    python benchmark.py --adapter ... --models bitnet,zeroshot,qwen   # subset
"""
import argparse
import csv
import json

import config
from eval import load_model, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="trained LoRA adapter for BitNet-v1")
    ap.add_argument("--data-dir", default=None,
                    help="local data dir (else DATA_SOURCE in .env)")
    ap.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    ap.add_argument("--output", default=config.BENCHMARK_OUTPUT)
    ap.add_argument("--device", default=config.DEVICE)
    ap.add_argument("--models", default="all",
                    help="comma list of: bitnet,zeroshot,mistral,qwen (default all)")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = whole splits")
    args = ap.parse_args()

    wanted = set(args.models.split(",") if args.models != "all"
                 else ["bitnet", "zeroshot", "mistral", "qwen"])

    def run(name, model_id, adapter=None):
        print(f"\n=== {name} ===")
        model, tok = load_model(model_id, adapter, args.device)
        for split, tag_s in [("heldout", "v3-heldout"),
                             ("knowledge", "domain-knowledge")]:
            res = evaluate(model, tok, split, args.device, batch_size=args.batch_size,
                           max_samples=args.max_samples, data_dir=args.data_dir)
            row = {"model": name, "test_set": tag_s, "accuracy": res["accuracy"],
                   "n_samples": res["n"], "qc_auc": res.get("qc_auc", "")}
            rows.append(row)
            print(row)

    def safe_run(name, model_id, adapter=None):
        try:
            run(name, model_id, adapter)
        except Exception as e:  # noqa: BLE001 — one bad model must not lose the rest
            print(f"[ERROR] {name} failed (omitted from CSV): {e}")

    rows = []
    if "zeroshot" in wanted:
        safe_run("BitNet-2B-zeroshot", config.BITNET_MODEL)
    if "bitnet" in wanted:
        if not args.adapter:
            raise SystemExit("--adapter required for BitNet-2B-v1")
        safe_run("BitNet-2B-v1", config.BITNET_MODEL, args.adapter)
    if "mistral" in wanted:
        safe_run("Mistral-7B-Instruct-v0.3", config.MISTRAL_MODEL)
    if "qwen" in wanted:
        safe_run("Qwen2.5-7B-Instruct", config.QWEN_MODEL)

    import pathlib
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "test_set", "accuracy",
                                          "n_samples", "qc_auc"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()