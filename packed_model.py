"""Packed (offline-ternary) BitNet + LoRA loading for GAQA v1.

Reproduces the paper's evaluation setup: the bf16 base model is converted to
offline-ternary BitLinear layers, the pre-packed ternary weights are loaded, and
the LoRA adapter is applied on top (the ~1.22 GB deployment).

Usage:
    from packed_model import load_packed_model
    model, tokenizer = load_packed_model(packed_path, adapter_dir, device)
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BitNetForCausalLM
from transformers.integrations.bitnet import BitLinear

import config


def apply_bitnet_quantization(module, verbose=True):
    """Replace nn.Linear layers with offline-ternary BitLinear layers."""
    target_names = {"q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"}
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, BitLinear):
            if name in target_names:
                new = BitLinear(child.in_features, child.out_features,
                                bias=child.bias is not None,
                                dtype=child.weight.dtype)
                new.to(device=child.weight.device)
                setattr(module, name, new)
                count += 1
        else:
            count += apply_bitnet_quantization(child, verbose=False)
    if verbose:
        print(f"  Replaced {count} layers with BitLinear (packed)")
    return count


def load_packed_weights(model, packed_path):
    """Load the packed ternary weights into the BitLinear-converted model."""
    state = torch.load(packed_path, map_location="cpu")
    result = model.load_state_dict(state, strict=False)
    print(f"  Loaded packed weights ({len(result.missing_keys)} missing, "
          f"{len(result.unexpected_keys)} unexpected)")
    return True


class LoraBitLinearWrapper(nn.Module):
    """BitLinear + LoRA adapters (offline ternary base + bf16 LoRA delta)."""

    def __init__(self, bitlinear_layer, adapter_name, r=16, lora_alpha=16,
                 lora_dropout=0.05):
        super().__init__()
        self.base_layer = bitlinear_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        in_features = bitlinear_layer.in_features
        out_features = bitlinear_layer.out_features
        device = bitlinear_layer.weight.device
        self.lora_A = nn.Parameter(torch.zeros(r, in_features, device=device,
                                               dtype=torch.bfloat16))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device,
                                               dtype=torch.bfloat16))

    def forward(self, x):
        base = self.base_layer(x)
        delta = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return base + delta


def apply_lora_to_packed(model, adapter_dir):
    """Apply the custom-format LoRA adapter (lora_weights.pt) onto BitLinear."""
    adapter_dir = Path(adapter_dir)
    lora_pt = adapter_dir / "lora_weights.pt"
    if not lora_pt.exists():
        raise FileNotFoundError(f"{lora_pt} not found")

    r, alpha = None, None
    for meta_name in ("adapter_config.json", "lora_meta.json"):
        mp = adapter_dir / meta_name
        if mp.exists():
            meta = json.loads(mp.read_text())
            r = meta.get("r")
            alpha = meta.get("lora_alpha") or meta.get("alpha")
            break

    state = torch.load(lora_pt, map_location="cpu")
    if r is None:
        for k in state:
            if k.endswith("_lora_A"):
                r = state[k].shape[0]
                break
    r = r or 8
    alpha = alpha or (2 * r)
    print(f"  Using LoRA r={r}, alpha={alpha}")

    loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            wrapper = LoraBitLinearWrapper(module, "default", r=r,
                                           lora_alpha=alpha, lora_dropout=0.0)
            prefix = name.replace(".", "_")
            if f"{prefix}_lora_A" in state:
                device = module.weight.device
                wrapper.lora_A.data = state[f"{prefix}_lora_A"].to(device)
                wrapper.lora_B.data = state[f"{prefix}_lora_B"].to(device)
                parent = model
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], wrapper)
                loaded += 1
    print(f"  LoRA applied to {loaded} BitLinear layers")
    return model


def load_packed_model(packed_path, adapter_dir, device="cuda",
                      model_id="microsoft/bitnet-b1.58-2B-4T-bf16"):
    """Load the packed GAQA v1 model (packed ternary base + LoRA adapter)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = config.CHAT_TEMPLATE

    model = BitNetForCausalLM.from_pretrained(model_id, device_map="cpu",
                                              dtype=torch.bfloat16)
    apply_bitnet_quantization(model)
    load_packed_weights(model, packed_path)
    model = model.to(device)
    model = apply_lora_to_packed(model, adapter_dir)
    model.eval()
    return model, tokenizer
