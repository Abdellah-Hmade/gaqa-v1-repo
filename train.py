"""Fine-tune BitNet b1.58 2B with LoRA + STE on GAQA v1 (completion-only loss).

Reproduces the paper's training numbers: ~4,143 optimizer steps (3 epochs x
~1,381 steps at batch size 8), final train loss ~0.007, validation loss ~0.040,
token accuracy > 99%, and 372M trainable parameters (15.15% of 2.46B).

No report/figure generation: outputs only the LoRA adapter + console summary.

Usage:
    python train.py                      # full run (3 epochs, needs ~80 GB A100)
    python train.py --max-steps 20       # quick check
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BitNetForCausalLM
from transformers.optimization import get_cosine_schedule_with_warmup

import config
from bitnet_lora import replace_with_lora, count_trainable
from data_utils import load_jsonl


def build_examples(rows, tokenizer, max_length):
    """Completion-only examples: full text = input + output; input tokens masked.

    Returns list of (input_ids, labels) where labels are -100 on the input
    segment and the real token ids on the answer segment.
    """
    out = []
    for row in rows:
        text_in = row["input"]
        text_out = row["output"]
        full = text_in + "\n" + text_out
        ids_in = tokenizer(text_in, add_special_tokens=False)["input_ids"]
        ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        ids = ids[: max_length - 1]
        ids_in = ids_in[: len(ids)]
        labels = [-100] * len(ids_in) + ids[len(ids_in):]
        out.append((ids, labels))
    return out


def collate(batch, tokenizer, max_length):
    ids_list, lab_list = zip(*batch)
    batch_ids, batch_lab, batch_attn = [], [], []
    for ids, lab in zip(ids_list, lab_list):
        n = len(ids)
        pad = max_length - n
        batch_ids.append(ids + [tokenizer.pad_token_id] * pad)
        batch_lab.append(lab + [-100] * pad)
        batch_attn.append([1] * n + [0] * pad)
    return (torch.tensor(batch_ids), torch.tensor(batch_attn, dtype=torch.long),
            torch.tensor(batch_lab))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="local dir with gaqa_v1_*.jsonl (else DATA_SOURCE in .env)")
    ap.add_argument("--output-dir", default=config.OUTPUT_DIR)
    ap.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--grad-accum", type=int, default=config.GRAD_ACCUM)
    ap.add_argument("--max-length", type=int, default=config.MAX_LENGTH)
    ap.add_argument("--max-train-samples", type=int, default=config.MAX_TRAIN_SAMPLES,
                    help="0 = use all")
    ap.add_argument("--max-steps", type=int, default=config.MAX_STEPS, help="0 = run full epochs")
    ap.add_argument("--eval-samples", type=int, default=config.EVAL_SAMPLES,
                    help="val samples for mid/final eval")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--device", default=config.DEVICE)
    args = ap.parse_args()
    if args.data_dir is None:
        args.data_dir = config.effective_data_dir("train")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_rows = load_jsonl("train", args.data_dir)
    val_rows = load_jsonl("val", args.data_dir)[: args.eval_samples]
    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]
    print(f"Train: {len(train_rows)} | Val (eval subset): {len(val_rows)}")

    print(f"Loading tokenizer + {config.BITNET_MODEL} (bfloat16) ...")
    tokenizer = AutoTokenizer.from_pretrained(config.BITNET_MODEL, token=config.HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    model = BitNetForCausalLM.from_pretrained(config.BITNET_MODEL, torch_dtype=torch.bfloat16,
                                              token=config.HF_TOKEN)
    model.to(args.device)
    model.config.use_cache = False

    replaced = replace_with_lora(model, config.LORA_R, config.LORA_ALPHA, config.LORA_DROPOUT)
    trainable, total, pct = count_trainable(model)
    print(f"Replaced {replaced} layers (expect 210) | "
          f"Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")

    train_ex = build_examples(train_rows, tokenizer, args.max_length)
    # AdamW 8-bit when available (matches the paper's adamw_8bit optimizer),
    # otherwise plain AdamW.
    try:
        from bitsandbytes.optim import AdamW8bit as _OptCls
        print("  optimizer: AdamW 8-bit (bitsandbytes)")
    except Exception:
        from torch.optim import AdamW as _OptCls
        print("  optimizer: AdamW (fp32)")
    opt = _OptCls([p for p in model.parameters() if p.requires_grad],
                  lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    steps_per_epoch = max(1, math.ceil(len(train_ex) / args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=int(total_steps * config.WARMUP_RATIO),
        num_training_steps=total_steps)

    val_ex = build_examples(val_rows, tokenizer, args.max_length)

    def run_eval():
        model.eval()
        tot_loss, n_ex, correct, toks = 0.0, 0, 0, 0
        with torch.no_grad():
            for i in range(0, len(val_ex), args.batch_size):
                ids, attn, lab = collate(val_ex[i:i + args.batch_size], tokenizer, args.max_length)
                ids, attn, lab = ids.to(args.device), attn.to(args.device), lab.to(args.device)
                out = model(input_ids=ids, attention_mask=attn, labels=lab)
                tot_loss += out.loss.item() * len(ids)
                n_ex += len(ids)
                pred = out.logits[:, :-1].argmax(-1)
                gold = lab[:, 1:]
                m = gold != -100
                correct += (pred[m] == gold[m]).sum().item()
                toks += m.sum().item()
        model.train()
        return tot_loss / max(n_ex, 1), correct / max(toks, 1)

    g = torch.Generator().manual_seed(args.seed)
    model.train()
    step, last_loss = 0, float("nan")
    for ep in range(args.epochs):
        idxs = torch.randperm(len(train_ex), generator=g).tolist()
        pbar = tqdm(range(0, len(idxs), args.batch_size), desc=f"Epoch {ep + 1}/{args.epochs}")
        for i in pbar:
            batch = [train_ex[j] for j in idxs[i:i + args.batch_size]]
            ids, attn, lab = collate(batch, tokenizer, args.max_length)
            ids, attn, lab = ids.to(args.device), attn.to(args.device), lab.to(args.device)
            loss = model(input_ids=ids, attention_mask=attn, labels=lab).loss
            last_loss = loss.item()
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], config.GRAD_CLIP)
                opt.step()
                opt.zero_grad()
                sched.step()
                step += 1
                pbar.set_postfix({"loss": f"{last_loss:.4f}", "step": step})
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    vl, vacc = run_eval()
    print(f"\nFinal: steps={step} (expected {config.EXPECTED_STEPS}) | "
          f"last train loss={last_loss:.4f} (paper ~0.007) | "
          f"val loss={vl:.4f} (paper ~0.040) | val token acc={vacc * 100:.2f}% (paper >99%)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = {}
    from bitnet_lora import LoRAAutoBitLinear
    for name, m in model.named_modules():
        if isinstance(m, LoRAAutoBitLinear):
            prefix = name.replace(".", "_")
            weights[f"{prefix}_lora_A"] = m.lora_A.data.clone().cpu()
            weights[f"{prefix}_lora_B"] = m.lora_B.data.clone().cpu()
    torch.save(weights, out_dir / "lora_weights.pt")
    (out_dir / "lora_meta.json").write_text(json.dumps({
        "r": config.LORA_R, "alpha": config.LORA_ALPHA, "dropout": config.LORA_DROPOUT,
        "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
        "steps": step, "final_train_loss": round(last_loss, 4),
        "val_loss": round(vl, 4), "val_token_acc": round(vacc, 4),
    }, indent=2))
    print(f"LoRA adapter saved to {out_dir}/lora_weights.pt "
          f"({len(weights)} matrices)")


if __name__ == "__main__":
    main()