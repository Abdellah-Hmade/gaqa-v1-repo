import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-05)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-05)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u

class BitLinearTrain(nn.Linear):

    def forward(self, x):
        w_q = self.weight + (weight_quant(self.weight) - self.weight).detach()
        x_q = x + (activation_quant(x) - x).detach()
        return F.linear(x_q.to(self.weight.dtype), w_q)

class LoRABitLinearTrain(nn.Module):

    def __init__(self, in_features, out_features, r=16, alpha=16, dropout=0.05, bias=False):
        super().__init__()
        self.base = BitLinearTrain(in_features, out_features, bias=bias)
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.randn(r, in_features, dtype=torch.bfloat16) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=torch.bfloat16))

    def forward(self, x):
        return self.base(x) + self.scaling * (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T

def replace_bitlinear_for_training(module):
    from transformers.integrations.bitnet import BitLinear
    replacements = 0
    for (name, child) in list(module.named_children()):
        sub_prefix = f'{prefix}.{name}' if prefix else name
        if isinstance(child, BitLinear):
            new = BitLinearTrain(child.in_features, child.out_features, bias=child.bias is not None)
            new.weight.data = child.weight.data.clone()
            if child.bias is not None:
                new.bias.data = child.bias.data.clone()
            new.to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(module, name, new)
            replacements += 1
        else:
            replace_bitlinear_for_training(child)
    return replacements

def replace_bitlinear_with_lora_train(module, r=16, alpha=16, dropout=0.05):
    from transformers.integrations.bitnet import BitLinear
    replacements = 0
    for (name, child) in list(module.named_children()):
        if isinstance(child, BitLinear):
            new = LoRABitLinearTrain(child.in_features, child.out_features, r=r, alpha=alpha, dropout=dropout, bias=child.bias is not None)
            new.base.weight.data = child.weight.data.clone()
            if child.bias is not None:
                new.base.bias.data = child.bias.data.clone()
            new.to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(module, name, new)
            replacements += 1
        else:
            replace_bitlinear_with_lora_train(child, r=r, alpha=alpha, dropout=dropout)
    return replacements

def extract_answer(text):
    for line in text.strip().split('\n'):
        lower = line.strip().lower()
        if 'correct answer is' in lower:
            rest = line.split('correct answer is')[-1].strip()
            if rest and rest[0].upper() in 'ABCD':
                return rest[0].upper()
        if lower.startswith('answer:'):
            rest = lower.split('answer:')[-1].strip()
            if rest and rest[0].upper() in 'ABCD':
                return rest[0].upper()
        if lower.startswith('answer'):
            rest = lower[4:].lstrip(':').strip()
            if rest and rest[0].upper() in 'ABCD':
                return rest[0].upper()
    return None

def get_chat_template():
    return "{% for message in messages %}{% if message['role'] == 'user' %}<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{% elif message['role'] == 'assistant' %}{{ message['content'] }}<|eot_id|>\n{% endif %}{% endfor %}"

def setup_hf_cache():
    import os
    from src.config import cfg
    os.environ['HF_HOME'] = str(cfg.hf_cache_dir.resolve())
    cfg.hf_cache_dir.mkdir(parents=True, exist_ok=True)

def apply_bitnet_quantization(module, verbose=True):
    from transformers.integrations.bitnet import BitLinear
    target_names = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    count = 0
    for (name, child) in list(module.named_children()):
        if isinstance(child, nn.Linear) and (not isinstance(child, BitLinear)):
            if name in target_names:
                new = BitLinear(child.in_features, child.out_features, bias=child.bias is not None, dtype=child.weight.dtype)
                new.to(device=child.weight.device)
                setattr(module, name, new)
                count += 1
        else:
            count += apply_bitnet_quantization(child, verbose=False)
    if verbose:
        print(f'  Replaced {count} layers with BitLinear (packed)')
        if count == 0:
            print('  WARNING: No layers replaced! Packing failed.')
    return count

def load_packed_weights(model, packed_dir):
    packed_path = Path(packed_dir)
    sd_path = packed_path / 'packed_weights.pt'
    if not sd_path.exists():
        print(f'  WARNING: Packed weights not found at {sd_path}')
        return False
    state = torch.load(sd_path, map_location='cpu')
    result = model.load_state_dict(state, strict=False)
    missing = len(result.missing_keys)
    unexpected = len(result.unexpected_keys)
    print(f'  Loaded packed weights: {missing} missing, {unexpected} unexpected keys')
    if missing > 0 and missing < 10:
        print(f'    Missing: {result.missing_keys[:5]}...')
    if unexpected > 0 and unexpected < 10:
        print(f'    Unexpected: {result.unexpected_keys[:5]}...')
    return True

class LoraBitLinearWrapper(nn.Module):

    def __init__(self, bitlinear_layer, adapter_name, r=16, lora_alpha=16, lora_dropout=0.05, **kwargs):
        super().__init__()
        self.base_layer = bitlinear_layer
        self._active_adapter = adapter_name
        self._merged = False
        self._disable_adapters = False
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        out_features = bitlinear_layer.out_features
        in_features = bitlinear_layer.in_features
        device = bitlinear_layer.weight.device
        self.lora_A = nn.Parameter(torch.zeros(r, in_features, device=device, dtype=torch.bfloat16))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=torch.bfloat16))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    @property
    def active_adapter(self):
        return self._active_adapter

    @property
    def merged(self):
        return self._merged

    def set_adapter(self, name):
        self._active_adapter = name

    def enable_adapters(self):
        self._disable_adapters = False

    def disable_adapters(self):
        self._disable_adapters = True

    def forward(self, x):
        base = self.base_layer(x)
        if self.merged or self._disable_adapters:
            return base
        delta = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return base + delta

def apply_lora_to_packed(model, adapter_path):
    adapter_dir = Path(adapter_path)
    lora_r = int(os.environ.get('BITNET_LORA_R', '8'))
    lora_alpha = int(os.environ.get('BITNET_LORA_ALPHA', '16'))
    print(f'  apply_lora_to_packed: r={lora_r}, alpha={lora_alpha}')
    lora_pt = adapter_dir / 'lora_weights.pt'
    if lora_pt.exists():
        print(f'  Loading custom LoRA weights from {lora_pt}')
        from transformers.integrations.bitnet import BitLinear
        state = torch.load(lora_pt, map_location='cpu')
        loaded = 0
        for (name, module) in model.named_modules():
            if isinstance(module, BitLinear):
                wrapper = LoraBitLinearWrapper(module, 'default', r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0)
                prefix = name.replace('.', '_')
                if f'{prefix}_lora_A' in state:
                    device = module.weight.device
                    expected_a_shape = (lora_r, module.in_features)
                    expected_b_shape = (module.out_features, lora_r)
                    actual_a_shape = state[f'{prefix}_lora_A'].shape
                    actual_b_shape = state[f'{prefix}_lora_B'].shape
                    if actual_a_shape != expected_a_shape:
                        raise ValueError(f'Shape mismatch for {prefix}_lora_A: expected {expected_a_shape}, got {actual_a_shape}. Check BITNET_LORA_R setting.')
                    if actual_b_shape != expected_b_shape:
                        raise ValueError(f'Shape mismatch for {prefix}_lora_B: expected {expected_b_shape}, got {actual_b_shape}. Check BITNET_LORA_R setting.')
                    wrapper.lora_A.data = state[f'{prefix}_lora_A'].to(device)
                    wrapper.lora_B.data = state[f'{prefix}_lora_B'].to(device)
                    parent = model
                    parts = name.split('.')
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    setattr(parent, parts[-1], wrapper)
                    loaded += 1
        print(f'  LoRA applied to {loaded} BitLinear layers')
        return model
    config_path = adapter_dir / 'adapter_config.json'
    if not config_path.exists():
        print(f'  No adapter_config.json found at {adapter_path}')
        return None
    cfg = json.loads(config_path.read_text())
    r = cfg.get('r', 16)
    lora_alpha = cfg.get('lora_alpha', 16)
    lora_dropout = cfg.get('lora_dropout', 0.05)
    target_modules = cfg.get('target_modules', [])
    print(f'  Adapter config: r={r}, alpha={lora_alpha}, targets={target_modules}')
    lora_config = LoraConfig(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=target_modules, task_type='CAUSAL_LM')
    lora_config._register_custom_module({BitLinear: LoraBitLinearWrapper})
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter_dir), config=lora_config)
    device = next(iter(model.parameters())).device
    for (name, module) in model.named_modules():
        if isinstance(module, LoraBitLinearWrapper):
            module.lora_A.data = module.lora_A.data.to(torch.bfloat16)
            module.lora_B.data = module.lora_B.data.to(torch.bfloat16)
    model = model.to(device)
    lora_count = sum((1 for (n, m) in model.named_modules() if isinstance(m, LoraBitLinearWrapper)))
    print(f'  LoRA applied to {lora_count} BitLinear layers (bf16)')
    return model
