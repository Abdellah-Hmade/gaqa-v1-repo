"""LoRA + straight-through-estimator (STE) support for BitNet.

Mirrors the fine-tuning protocol in the paper: the ternary weights are frozen
and trained only through the STE, while small bfloat16 LoRA matrices absorb the
domain-specific updates. Gradient flow through the ternary layers is preserved
by STE (straight-through estimator) during the backward pass.
"""
import torch
import torch.nn as nn
from transformers.integrations import bitnet as bitnet_module
from transformers.integrations.bitnet import AutoBitLinear

import config


class WeightQuantNoCompile(torch.autograd.Function):
    """WeightQuant without @torch.compile (compatible with smaller GPUs)."""
    @staticmethod
    def forward(ctx, weight):
        dtype = weight.dtype
        w = weight.float()
        scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        w = (w * scale).round().clamp(-1, 1) / scale
        return w.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class ActQuantNoCompile(torch.autograd.Function):
    """ActQuant without @torch.compile."""
    @staticmethod
    def forward(ctx, activation):
        dtype = activation.dtype
        a = activation.float()
        scale = 127 / a.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        a = (a * scale).round().clamp(-128, 127) / scale
        return a.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


bitnet_module.WeightQuant = WeightQuantNoCompile
bitnet_module.ActQuant = ActQuantNoCompile


class LoRAAutoBitLinear(nn.Module):
    """AutoBitLinear (ternary) + LoRA adapters. Base weights frozen."""

    def __init__(self, base: AutoBitLinear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base
        self.base.online_quant = True
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        in_f, out_f = base.in_features, base.out_features
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.randn(r, in_f, dtype=torch.bfloat16, device=dev) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, dtype=torch.bfloat16, device=dev))

    def forward(self, x):
        base_out = self.base(x)
        lora_A = self.lora_A.to(x.dtype)
        lora_B = self.lora_B.to(x.dtype)
        lora_out = self.scaling * (self.dropout(x) @ lora_A.T) @ lora_B.T
        return base_out + lora_out


def replace_with_lora(module: nn.Module, r=8, alpha=16, dropout=0.05) -> int:
    """Replace every AutoBitLinear layer with LoRAAutoBitLinear. Returns count."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, AutoBitLinear):
            setattr(module, name, LoRAAutoBitLinear(child, r=r, alpha=alpha, dropout=dropout))
            n += 1
        else:
            n += replace_with_lora(child, r=r, alpha=alpha, dropout=dropout)
    return n


def apply_adapter(model: nn.Module, adapter_path: str) -> None:
    """Load a saved lora_weights*.pt dict into the model's LoRA matrices."""
    sd = torch.load(adapter_path, map_location="cpu")
    found = 0
    for name, m in model.named_modules():
        if isinstance(m, LoRAAutoBitLinear):
            prefix = name.replace(".", "_")
            m.lora_A.data = sd[f"{prefix}_lora_A"].to(m.lora_A.device).to(torch.bfloat16)
            m.lora_B.data = sd[f"{prefix}_lora_B"].to(m.lora_B.device).to(torch.bfloat16)
            found += 1
    if found == 0:
        raise RuntimeError("no LoRAAutoBitLinear layers found to apply adapter to")


def count_trainable(model: nn.Module) -> tuple[int, int, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, 100.0 * trainable / total