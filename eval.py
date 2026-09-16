"""Logit-based forced-choice MCQ evaluation for GAQA v1.

Protocol (paper, "Evaluation Protocol"): the model is prompt-prefilled with the
question and ``The correct answer is''; the next-token logits for the four
answer letters A/B/C/D are extracted; the highest-probability letter is
compared with the gold answer. For qc_diagnosis we also compute the AUC of
acceptable-vs-defective classification using the probability assigned to the
acceptable-quality option as the diagnostic score.

Usage:
    python eval.py --split heldout                          # zero-shot BitNet
    python eval.py --split heldout --adapter outputs/.../lora_weights.pt
    python eval.py --split knowledge --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitNetForCausalLM

import config
from bitnet_lora import replace_with_lora, apply_adapter
from data_utils import load_jsonl, gold_letter, acceptable_letter
from packed_model import load_packed_model

LETTERS = ["A", "B", "C", "D"]


def load_model(model_id, adapter=None, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=config.HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = config.CHAT_TEMPLATE
    is_bitnet = "bitnet" in model_id.lower()
    cls = BitNetForCausalLM if is_bitnet else AutoModelForCausalLM
    model = cls.from_pretrained(model_id, torch_dtype=torch.bfloat16, token=config.HF_TOKEN)
    if is_bitnet:
        replaced = replace_with_lora(model, config.LORA_R, config.LORA_ALPHA, config.LORA_DROPOUT)
        if adapter:
            apply_adapter(model, adapter)
    model.to(device)
    model.eval()
    print(f"  loaded {model_id}" + (f" + adapter {adapter} ({replaced} layers)" if is_bitnet else ""))
    return model, tokenizer


def _letter_token_ids(tokenizer):
    """Candidate token IDs per letter, covering tokenizer variants."""
    ids = {}
    for L in LETTERS:
        cand = []
        for variant in [f" {L}", L, f" {L.lower()}", L.lower()]:
            cand.append(tokenizer.encode(variant, add_special_tokens=False)[0])
        ids[L] = cand
    return ids


def predict_batch(model, tokenizer, rows, device, batch_size=16, max_length=1024):
    """Return (pred_letters, letter_probability dicts) via next-token logits.

    Protocol matches the paper: each question is wrapped in the chat template and
    prefilled with ``The correct answer is``; each letter's score is the max
    logit over its token variants, softmaxed across the four letters.
    """
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": r["input"]}],
            tokenize=False, add_generation_prompt=True,
        ) + "The correct answer is"
        for r in rows
    ]
    letter_ids = _letter_token_ids(tokenizer)
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length)
            ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            logits = model(input_ids=ids, attention_mask=attn).logits
            lens = attn.sum(dim=1) - 1
            for j in range(len(chunk)):
                v = logits[j, lens[j].item()].float()
                ll = {L: v[letter_ids[L]].max().item() for L in LETTERS}
                vals = torch.tensor([ll[L] for L in LETTERS], device=device)
                p = torch.softmax(vals, dim=0)
                pv = {L: p[k].item() for k, L in enumerate(LETTERS)}
                preds.append(max(LETTERS, key=lambda L: ll[L]))
                probs.append(pv)
    return preds, probs


def evaluate(model, tokenizer, split, device, batch_size=16, per_family=False,
             max_samples=0, data_dir=None):
    rows = load_jsonl(split, data_dir)
    if max_samples:
        rows = rows[:max_samples]
    preds, probs = predict_batch(model, tokenizer, rows, device, batch_size)
    golds = [gold_letter(r["output"]) for r in rows]
    correct = [p == g for p, g in zip(preds, golds)]
    acc = float(np.mean(correct))
    res = {"split": split, "n": len(rows), "accuracy": round(acc, 4)}

    if per_family:
        fam = {}
        for row, ok in zip(rows, correct):
            fam.setdefault(row["subject"], []).append(ok)
        res["per_family"] = {k: round(float(np.mean(v)), 4)
                             for k, v in sorted(fam.items())}

    qc_idx = [i for i, r in enumerate(rows) if r["subject"] == "qc_diagnosis"]
    if qc_idx:
        acc_letters = [acceptable_letter(rows[i]) for i in qc_idx]
        if all(a is not None for a in acc_letters):
            y = [golds[i] == acc_letters[j] for j, i in enumerate(qc_idx)]
            score = [probs[i][acc_letters[j]] for j, i in enumerate(qc_idx)]
            res["qc_auc"] = round(float(roc_auc_score(y, score)), 4)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.BITNET_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--packed", action="store_true",
                    help="load the packed (offline-ternary) model instead of bf16")
    ap.add_argument("--packed-weights", default="packed_weights.pt",
                    help="path to packed_weights.pt (used with --packed)")
    ap.add_argument("--split", choices=["heldout", "knowledge"], required=True)
    ap.add_argument("--data-dir", default=None,
                    help="local data dir (else chosen by DATA_SOURCE in .env)")
    ap.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    ap.add_argument("--per-family", action="store_true")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = whole split (sanity runs: small N)")
    ap.add_argument("--device", default=config.DEVICE)
    args = ap.parse_args()

    if args.packed:
        adapter_dir = args.adapter or "."
        if not Path(adapter_dir).is_dir():
            adapter_dir = str(Path(adapter_dir).parent)
        model, tokenizer = load_packed_model(args.packed_weights, adapter_dir,
                                             args.device)
    else:
        model, tokenizer = load_model(args.model, args.adapter, args.device)
    res = evaluate(model, tokenizer, args.split, args.device,
                   batch_size=args.batch_size, per_family=args.per_family,
                   max_samples=args.max_samples, data_dir=args.data_dir)
    print(json.dumps(res, indent=2))
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()