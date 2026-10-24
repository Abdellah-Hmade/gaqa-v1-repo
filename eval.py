import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitNetForCausalLM

import constants
import data
from model import apply_adapter, replace_with_lora

LETTERS = ["A", "B", "C", "D"]

MODEL = constants.BITNET_MODEL
ADAPTER = None
SPLIT = "heldout"
DATA_DIR = "data/processed"
PER_FAMILY = False
MAX_SAMPLES = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_id, adapter, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = constants.CHAT_TEMPLATE
    is_bitnet = "bitnet" in model_id.lower()
    model = (BitNetForCausalLM if is_bitnet else AutoModelForCausalLM).from_pretrained(
        model_id, torch_dtype=torch.bfloat16)
    if is_bitnet:
        replace_with_lora(model, constants.LORA_R, constants.LORA_ALPHA, constants.LORA_DROPOUT)
        if adapter:
            apply_adapter(model, adapter)
    model.to(device)
    model.eval()
    return model, tokenizer


def letter_token_ids(tokenizer):
    return {L: [tokenizer.encode(v, add_special_tokens=False)[0]
                for v in (f" {L}", L, f" {L.lower()}", L.lower())] for L in LETTERS}


def predict(model, tokenizer, rows, device, batch_size, max_length=1024):
    prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": row["input"]}],
                                      tokenize=False, add_generation_prompt=True) + "The correct answer is"
        for row in rows
    ]
    ids_by_letter = letter_token_ids(tokenizer)
    preds = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            enc = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length)
            logits = model(input_ids=enc["input_ids"].to(device),
                           attention_mask=enc["attention_mask"].to(device)).logits
            positions = enc["attention_mask"].sum(dim=1) - 1
            for j in range(len(enc["input_ids"])):
                values = logits[j, positions[j].item()].float()
                scores = {L: values[ids_by_letter[L]].max().item() for L in LETTERS}
                preds.append(max(LETTERS, key=lambda L: scores[L]))
    return preds


def evaluate(model, tokenizer, split, device, batch_size, per_family=False, max_samples=0, data_dir="data"):
    rows = data.load(split, data_dir)
    if max_samples:
        rows = rows[:max_samples]
    preds = predict(model, tokenizer, rows, device, batch_size)
    golds = [data.gold_letter(row["output"]) for row in rows]
    correct = [p == g for p, g in zip(preds, golds)]
    result = {"split": split, "n": len(rows), "accuracy": round(sum(correct) / len(correct), 4)}
    if per_family:
        families = {}
        for row, ok in zip(rows, correct):
            families.setdefault(row["subject"], []).append(ok)
        result["per_family"] = {k: round(sum(v) / len(v), 4) for k, v in sorted(families.items())}
    return result


def main():
    model, tokenizer = load_model(MODEL, ADAPTER, DEVICE)
    result = evaluate(model, tokenizer, SPLIT, DEVICE, constants.EVAL_BATCH_SIZE,
                      PER_FAMILY, MAX_SAMPLES, DATA_DIR)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
