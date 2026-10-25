import torch, json
from pathlib import Path
from transformers import BitNetForCausalLM, AutoTokenizer
from src.config import cfg
from src.common import extract_answer, get_chat_template, setup_hf_cache

def pack_ternary_weights(w_ternary):
    (out, in_) = w_ternary.shape
    assert out % 4 == 0
    b = out // 4
    w = (w_ternary + 1).to(torch.uint8).contiguous()
    return w[0:b] | w[b:2 * b] << 2 | w[2 * b:3 * b] << 4 | w[3 * b:out] << 6

def convert_model(model, output_dir):
    from transformers.integrations.bitnet import BitLinear
    target_names = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    converted = 0

    def _convert(module, prefix=''):
        nonlocal converted
        for (name, child) in list(module.named_children()):
            layer = child
            if hasattr(child, 'base_layer') and isinstance(child.base_layer, torch.nn.Linear):
                layer = child.base_layer
            if isinstance(layer, torch.nn.Linear) and (not isinstance(layer, BitLinear)):
                if name in target_names:
                    w = layer.weight.data
                    scale = 1.0 / w.abs().mean().clamp_(min=1e-05)
                    w_ternary = (w * scale).round().clamp_(-1, 1)
                    packed = pack_ternary_weights(w_ternary)
                    new = BitLinear(layer.in_features, layer.out_features, bias=layer.bias is not None, dtype=w.dtype, device=w.device)
                    new.weight.copy_(packed)
                    new.weight_scale.copy_(scale.to(w.dtype))
                    if layer.bias is not None:
                        new.bias.copy_(layer.bias.data)
                    setattr(module, name, new)
                    converted += 1
            else:
                _convert(child, name)
    _convert(model)
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(output_dir / 'packed_weights.pt'))
    print(f'  Converted: {converted} layers | Saved to {output_dir}/packed_weights.pt')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='models/bitnet-packed')
    args = parser.parse_args()
    setup_hf_cache()
    print('=' * 60)
    print('BitNet bf16 → packed ternary (1.58-bit edge)')
    print('=' * 60)
    print(f'[1/2] Loading base model: {cfg.bitnet_model}')
    model = BitNetForCausalLM.from_pretrained(cfg.bitnet_model, device_map='cpu', torch_dtype=torch.bfloat16)
    print(f'  {sum((p.numel() for p in model.parameters())) / 1000000.0:.0f}M params')
    print(f'\n[2/2] Converting to packed BitNet...')
    convert_model(model, args.output)
    print(f'\n  Packed base model (no LoRA merge) — use --packed --adapter for LoRA')
if __name__ == '__main__':
    main()
