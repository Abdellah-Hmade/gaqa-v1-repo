import torch
import torch.nn as nn
from transformers.integrations import bitnet as bitnet_module
from transformers.integrations.bitnet import AutoBitLinear


class WeightQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        dtype = weight.dtype
        w = weight.float()
        scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        w = (w * scale).round().clamp(-1, 1) / scale
        return w.to(dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad.clone()


class ActQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, act):
        dtype = act.dtype
        a = act.float()
        scale = 127 / a.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        a = (a * scale).round().clamp(-128, 127) / scale
        return a.to(dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad.clone()


bitnet_module.WeightQuant = WeightQuant
bitnet_module.ActQuant = ActQuant


class LoRAAutoBitLinear(nn.Module):
    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base = base
        self.base.online_quant = True
        for p in self.base.parameters():
            p.requires_grad = False
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.randn(r, base.in_features, dtype=torch.bfloat16, device=device) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.bfloat16, device=device))

    def forward(self, x):
        a = self.lora_A.to(x.dtype)
        b = self.lora_B.to(x.dtype)
        return self.base(x) + self.scaling * (self.dropout(x) @ a.T) @ b.T


def replace_with_lora(module, r, alpha, dropout):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, AutoBitLinear):
            setattr(module, name, LoRAAutoBitLinear(child, r, alpha, dropout))
            count += 1
        else:
            count += replace_with_lora(child, r, alpha, dropout)
    return count


def apply_adapter(model, path):
    state = torch.load(path, map_location="cpu")
    found = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRAAutoBitLinear):
            prefix = name.replace(".", "_")
            module.lora_A.data = state[f"{prefix}_lora_A"].to(module.lora_A.device).to(torch.bfloat16)
            module.lora_B.data = state[f"{prefix}_lora_B"].to(module.lora_B.device).to(torch.bfloat16)
            found += 1
    if not found:
        raise RuntimeError("no LoRAAutoBitLinear layers to apply adapter to")
    return found


def count_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, 100.0 * trainable / total
