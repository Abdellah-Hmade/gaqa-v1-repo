import json
import math
import random
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BitNetForCausalLM
from transformers.optimization import get_cosine_schedule_with_warmup

import constants
import data
from model import LoRAAutoBitLinear, count_trainable, replace_with_lora

DATA_DIR = "data/processed"
OUTPUT_DIR = "outputs/lora"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_STEPS = 0
MAX_TRAIN_SAMPLES = 0
EVAL_SAMPLES = 256


def make_examples(rows, tokenizer, max_length):
    out = []
    for row in rows:
        prompt_ids = tokenizer(row["input"], add_special_tokens=False)["input_ids"]
        ids = tokenizer(row["input"] + "\n" + row["output"], add_special_tokens=False)["input_ids"][: max_length - 1]
        prompt_ids = prompt_ids[: len(ids)]
        out.append((ids, [-100] * len(prompt_ids) + ids[len(prompt_ids):]))
    return out


def collate(batch, tokenizer, max_length):
    ids_list, labels_list = zip(*batch)
    batch_ids, batch_labels, batch_attn = [], [], []
    for ids, labels in zip(ids_list, labels_list):
        pad = max_length - len(ids)
        batch_ids.append(ids + [tokenizer.pad_token_id] * pad)
        batch_labels.append(labels + [-100] * pad)
        batch_attn.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(batch_ids), torch.tensor(batch_attn, dtype=torch.long), torch.tensor(batch_labels))


def main():
    random.seed(constants.SEED)
    torch.manual_seed(constants.SEED)

    train_rows = data.load("train", DATA_DIR)
    val_rows = data.load("val", DATA_DIR)[:EVAL_SAMPLES]
    if MAX_TRAIN_SAMPLES:
        train_rows = train_rows[:MAX_TRAIN_SAMPLES]
    print(f"train={len(train_rows)} val={len(val_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(constants.BITNET_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = BitNetForCausalLM.from_pretrained(constants.BITNET_MODEL, torch_dtype=torch.bfloat16)
    model.to(DEVICE)
    model.config.use_cache = False

    replaced = replace_with_lora(model, constants.LORA_R, constants.LORA_ALPHA, constants.LORA_DROPOUT)
    trainable, total, pct = count_trainable(model)
    print(f"layers={replaced} trainable={trainable} total={total} pct={pct:.2f}")

    train_ex = make_examples(train_rows, tokenizer, constants.MAX_LENGTH)
    val_ex = make_examples(val_rows, tokenizer, constants.MAX_LENGTH)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=constants.LEARNING_RATE, weight_decay=constants.WEIGHT_DECAY)
    steps_per_epoch = max(1, math.ceil(len(train_ex) / constants.BATCH_SIZE))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(steps_per_epoch * constants.NUM_EPOCHS * constants.WARMUP_RATIO),
        num_training_steps=steps_per_epoch * constants.NUM_EPOCHS)

    def evaluate():
        model.eval()
        loss_sum, n, correct, toks = 0.0, 0, 0, 0
        with torch.no_grad():
            for i in range(0, len(val_ex), constants.BATCH_SIZE):
                ids, attn, labels = collate(val_ex[i:i + constants.BATCH_SIZE], tokenizer, constants.MAX_LENGTH)
                ids, attn, labels = ids.to(DEVICE), attn.to(DEVICE), labels.to(DEVICE)
                out = model(input_ids=ids, attention_mask=attn, labels=labels)
                loss_sum += out.loss.item() * len(ids)
                n += len(ids)
                pred = out.logits[:, :-1].argmax(-1)
                gold = labels[:, 1:]
                mask = gold != -100
                correct += (pred[mask] == gold[mask]).sum().item()
                toks += mask.sum().item()
        model.train()
        return loss_sum / max(n, 1), correct / max(toks, 1)

    generator = torch.Generator().manual_seed(constants.SEED)
    model.train()
    step, last_loss = 0, float("nan")
    for epoch in range(constants.NUM_EPOCHS):
        order = torch.randperm(len(train_ex), generator=generator).tolist()
        bar = tqdm(range(0, len(order), constants.BATCH_SIZE), desc=f"epoch {epoch + 1}/{constants.NUM_EPOCHS}")
        for i in bar:
            batch = [train_ex[j] for j in order[i:i + constants.BATCH_SIZE]]
            ids, attn, labels = collate(batch, tokenizer, constants.MAX_LENGTH)
            ids, attn, labels = ids.to(DEVICE), attn.to(DEVICE), labels.to(DEVICE)
            loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
            last_loss = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], constants.GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1
            bar.set_postfix({"loss": f"{last_loss:.4f}", "step": step})
            if MAX_STEPS and step >= MAX_STEPS:
                break
        if MAX_STEPS and step >= MAX_STEPS:
            break

    val_loss, val_acc = evaluate()
    print(f"steps={step} train_loss={last_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc * 100:.2f}")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRAAutoBitLinear):
            prefix = name.replace(".", "_")
            weights[f"{prefix}_lora_A"] = module.lora_A.data.clone().cpu()
            weights[f"{prefix}_lora_B"] = module.lora_B.data.clone().cpu()
    torch.save(weights, out_dir / "lora_weights.pt")
    (out_dir / "meta.json").write_text(json.dumps({
        "r": constants.LORA_R, "alpha": constants.LORA_ALPHA, "steps": step,
        "val_loss": round(val_loss, 4), "val_token_acc": round(val_acc, 4)}, indent=2))
    print(f"saved {out_dir / 'lora_weights.pt'} ({len(weights)} matrices)")


if __name__ == "__main__":
    main()
