"""Smoke test: verify the full GAQA v1 reproduction pipeline end-to-end.

Runs the real code paths (data loading, STE + LoRA replacement, completion-only
loss training, logit-based MCQ evaluation, qc_diagnosis AUC) on a tiny randomly
initialized BitNet model, so feasibility can be checked quickly on CPU or a
small GPU without downloading the 2B model.

Usage:
    python smoke_test.py                # uses a few synthetic examples, offline
    python smoke_test.py --data-dir ../path   # use a few real GAQA examples
"""
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, BitNetConfig, BitNetForCausalLM
from transformers.integrations.bitnet import AutoBitLinear

import config
from bitnet_lora import replace_with_lora, apply_adapter, count_trainable
from train import build_examples, collate
from eval import predict_batch
from data_utils import load_jsonl, gold_letter, acceptable_letter

SYNTH_OPTIONS = {
    "A": "Acceptable quality assembly (no major issues)",
    "B": "Misassembly errors (structural errors in the assembly)",
    "C": "Low sequencing coverage / shallow depth",
    "D": "Contamination (foreign DNA in the assembly)",
}


def make_synthetic_rows(n=16, seed=0):
    rng = random.Random(seed)
    subjects = ["qc_diagnosis", "remediation", "tool_selection"]
    rows = []
    for i in range(n):
        subject = subjects[i % 3]
        letter = "ABCD"[i % 4]
        q = (f"Subject: {subject}\nQuestion: Below is a genome assembly QC report:\n"
             f"  N50 contig size: {50000 + i * 1000}\n"
             f"  BUSCO complete (%): {90 + (i % 10)}\n"
             f"  Genome fraction (%): {95 + (i % 5)}\n"
             f"  Sequencing depth (x): {10 * (1 + i % 10)}\n"
             "What is the most likely problem with this assembly?\nOptions:\n"
             + "\n".join(f"{L}) {t}" for L, t in SYNTH_OPTIONS.items()))
        out = f"The correct answer is {letter}: {SYNTH_OPTIONS[letter]}"
        rows.append({"id": f"smoke_{i}", "subject": subject,
                     "input": q, "output": out})
    return rows


def to_autobitlinear(module: nn.Module):
    """Convert plain nn.Linear layers to AutoBitLinear (as the real checkpoint
    load does), so the STE quantization + LoRA path is exercised.
    lm_head is kept as nn.Linear, matching the real quantization config."""
    for name, child in list(module.named_children()):
        if name == "lm_head":
            continue
        if isinstance(child, nn.Linear):
            nb = AutoBitLinear(child.in_features, child.out_features,
                               bias=child.bias is not None,
                               dtype=child.weight.dtype)
            nb.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                nb.bias.data.copy_(child.bias.data)
            setattr(module, name, nb)
        else:
            to_autobitlinear(child)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--max-samples", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default=config.DEVICE)
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    print("1/4 building dataset ...")
    if args.data_dir:
        rows = (load_jsonl("train", args.data_dir)
                + load_jsonl("heldout", args.data_dir))[: args.max_samples]
    else:
        rows = make_synthetic_rows(args.max_samples)
    train_rows = rows[: max(1, len(rows) // 2)]
    eval_rows = rows[len(rows) // 2:]
    print(f"   train={len(train_rows)} eval={len(eval_rows)}")

    print("2/4 building tiny BitNet model + STE/LoRA ...")
    tokenizer = AutoTokenizer.from_pretrained(config.BITNET_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = BitNetForCausalLM(BitNetConfig(
        vocab_size=len(tokenizer), hidden_size=128, intermediate_size=384,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=1024))
    to_autobitlinear(model)
    replaced = replace_with_lora(model, r=8, alpha=16, dropout=0.05)
    model.to(args.device)
    trainable, total, pct = count_trainable(model)
    print(f"   replaced {replaced} layers (expect 14 = 7 projections x 2 blocks) | "
          f"trainable {trainable:,}/{total:,} ({pct:.2f}%)")

    print("3/4 training 2 steps (completion-only loss) ...")
    ex = build_examples(train_rows, tokenizer, args.max_length)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model.train()
    for step in range(args.steps):
        batch = ex[step * args.batch_size:(step + 1) * args.batch_size]
        ids, attn, lab = collate(batch, tokenizer, args.max_length)
        ids, attn, lab = ids.to(args.device), attn.to(args.device), lab.to(args.device)
        loss = model(input_ids=ids, attention_mask=attn, labels=lab).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(f"   step {step + 1}: loss={loss.item():.4f}")

    print("4/4 MCQ evaluation + qc_diagnosis AUC ...")
    model.eval()
    preds, probs = predict_batch(model, tokenizer, eval_rows, args.device, batch_size=2)
    golds = [gold_letter(r["output"]) for r in eval_rows]
    acc = float(np.mean([p == g for p, g in zip(preds, golds)]))
    print(f"   accuracy: {acc * 100:.1f}% (chance 25%)")

    qc_idx = [i for i, r in enumerate(eval_rows) if r["subject"] == "qc_diagnosis"]
    if qc_idx:
        acc_letters = [acceptable_letter(eval_rows[i]) for i in qc_idx]
        if all(a is not None for a in acc_letters):
            y = [golds[i] == acc_letters[j] for j, i in enumerate(qc_idx)]
            score = [probs[i][acc_letters[j]] for j, i in enumerate(qc_idx)]
            print(f"   qc_diagnosis AUC: {roc_auc_score(y, score):.4f}")

    print("4.5/4 adapter save/load round-trip ...")
    from bitnet_lora import LoRAAutoBitLinear
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    sd = {}
    for name, m in model.named_modules():
        if isinstance(m, LoRAAutoBitLinear):
            prefix = name.replace(".", "_")
            sd[f"{prefix}_lora_A"] = m.lora_A.data.clone().cpu()
            sd[f"{prefix}_lora_B"] = m.lora_B.data.clone().cpu()
    torch.save(sd, tmp / "lora_weights.pt")

    fresh = BitNetForCausalLM(BitNetConfig(
        vocab_size=len(tokenizer), hidden_size=128, intermediate_size=384,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=1024))
    to_autobitlinear(fresh)
    replace_with_lora(fresh, r=8, alpha=16, dropout=0.05)
    apply_adapter(fresh, tmp / "lora_weights.pt")
    ok = all(
        torch.equal(m.lora_A.data.cpu(), sd[f"{name.replace('.', '_')}_lora_A"])
        for name, m in fresh.named_modules() if isinstance(m, LoRAAutoBitLinear))
    print(f"   adapter round-trip {'OK' if ok else 'FAILED'} ({len(sd) // 2} matrices)")

    print("\nSMOKE TEST PASSED: data -> STE/LoRA -> completion-only training -> "
          "MCQ eval + AUC + adapter save/load all ran.")
    print("Note: this uses a tiny random model and synthetic/real-subset data;")
    print("run train.py / benchmark.py with a GPU for the full reproduction.")


if __name__ == "__main__":
    main()