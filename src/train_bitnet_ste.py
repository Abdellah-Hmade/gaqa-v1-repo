import os, torch, random, numpy as np
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from pathlib import Path
from transformers import AutoTokenizer, BitNetForCausalLM
from transformers.integrations.bitnet import AutoBitLinear
from transformers.integrations import bitnet as bitnet_module
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn as nn

class WeightQuantNoCompile(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight):
        dtype = weight.dtype
        weight = weight.float()
        scale = 1.0 / weight.abs().mean().clamp_(min=1e-05)
        weight = (weight * scale).round().clamp(-1, 1) / scale
        return weight.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class ActQuantNoCompile(torch.autograd.Function):

    @staticmethod
    def forward(ctx, activation):
        dtype = activation.dtype
        activation = activation.float()
        scale = 127 / activation.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-05)
        activation = (activation * scale).round().clamp(-128, 127) / scale
        return activation.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
bitnet_module.WeightQuant = WeightQuantNoCompile
bitnet_module.ActQuant = ActQuantNoCompile
from src.config import cfg
from src.common import get_chat_template, setup_hf_cache
from src.training_logger import TrainingLogger
from src.notify import notify

class LoRAAutoBitLinear(nn.Module):

    def __init__(self, autbitlinear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = autbitlinear
        self.base.online_quant = True
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        in_f = autbitlinear.in_features
        out_f = autbitlinear.out_features
        device = autbitlinear.weight.device
        self.lora_A = nn.Parameter(torch.randn(r, in_f, dtype=torch.bfloat16, device=device) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, dtype=torch.bfloat16, device=device))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.scaling * (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out

def replace_with_lora(module, r=8, alpha=16, dropout=0.05):
    count = 0
    for (name, child) in list(module.named_children()):
        if isinstance(child, AutoBitLinear):
            new = LoRAAutoBitLinear(child, r=r, alpha=alpha, dropout=dropout)
            setattr(module, name, new)
            count += 1
        else:
            count += replace_with_lora(child, r=r, alpha=alpha, dropout=dropout)
    return count

def save_lora_weights(model, output_dir, name):
    lora_weights = {}
    for (name_mod, module) in model.named_modules():
        if isinstance(module, LoRAAutoBitLinear):
            prefix = name_mod.replace('.', '_')
            lora_weights[f'{prefix}_lora_A'] = module.lora_A.data.clone().cpu()
            lora_weights[f'{prefix}_lora_B'] = module.lora_B.data.clone().cpu()
    path = Path(output_dir) / f'lora_weights_{name}.pt'
    torch.save(lora_weights, str(path))
    print(f'  Saved {len(lora_weights)} LoRA params to {path}')
    return lora_weights

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default='logs/training', help='Directory to save training logs')
    parser.add_argument('--dry-run', action='store_true', help='Run with small dataset for testing')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()
    setup_hf_cache()
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model_label = f'BitNet-2B-STE-seed{seed}'
    print('=' * 60)
    print('BitNet LoRA training with STE (differentiable quantization)')
    if args.dry_run:
        print('DRY RUN MODE: Using 100 samples, 1 epoch')
    print(f'Model: {cfg.bitnet_model}')
    print(f'Seed: {seed}')
    print(f'LoRA: r={cfg.bitnet_lora_r}, alpha={cfg.bitnet_lora_alpha}')
    print(f'LR: {cfg.bitnet_learning_rate}, epochs: {(1 if args.dry_run else cfg.bitnet_num_epochs)}')
    print(f'Batch: {cfg.batch_size}, Grad accum: {cfg.grad_accum}')
    print(f'Max length: {cfg.max_length}, Samples: {(100 if args.dry_run else cfg.max_train_samples)}')
    print('=' * 60)
    notify('BitNet STE Training', f'Starting seed {seed}...', '4')
    config_dict = {'lora_r': cfg.bitnet_lora_r, 'lora_alpha': cfg.bitnet_lora_alpha, 'lora_dropout': cfg.bitnet_lora_dropout, 'learning_rate': cfg.bitnet_learning_rate, 'num_epochs': 1 if args.dry_run else cfg.bitnet_num_epochs, 'batch_size': cfg.batch_size, 'grad_accum': cfg.grad_accum, 'max_length': cfg.max_length, 'warmup_ratio': cfg.warmup_ratio, 'scheduler': cfg.scheduler, 'seed': seed}
    logger = TrainingLogger(log_dir=args.log_dir, model_name=model_label, config=config_dict, resume=args.resume)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available. Please use a GPU with >= 16GB VRAM.')
    print(f'\nGPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1000000000.0:.1f} GB')
    print('\n[1/6] Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(cfg.bitnet_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenizer.chat_template = get_chat_template()
    print('[2/6] Loading dataset...')
    train_ds = load_dataset('json', data_files=str(cfg.data_processed_dir / 'train.jsonl'), split='train')
    val_ds = load_dataset('json', data_files=str(cfg.data_processed_dir / 'val.jsonl'), split='train')
    if args.dry_run:
        max_train = 100
        max_eval = 20
    else:
        max_train = cfg.max_train_samples
        max_eval = cfg.max_eval_samples
    if max_train and len(train_ds) > max_train:
        train_ds = train_ds.select(range(max_train))
    if max_eval and len(val_ds) > max_eval:
        val_ds = val_ds.select(range(max_eval))
    completion_only = os.environ.get('COMPLETION_ONLY_LOSS', 'false').lower() == 'true'
    print(f"  Loss mode: {('COMPLETION ONLY (prompt masked)' if completion_only else 'FULL SEQUENCE')}")

    def format_text(example):
        instr = example.get('instruction', '')
        user_content = f"{instr}\n\n{example['input']}" if instr else example['input']
        user_msg = {'role': 'user', 'content': user_content}
        assistant_msg = {'role': 'assistant', 'content': example['output']}
        full_text = tokenizer.apply_chat_template([user_msg, assistant_msg], tokenize=False)
        result = {'text': full_text}
        if completion_only:
            prompt_text = tokenizer.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)
            prompt_len = len(tokenizer(prompt_text, add_special_tokens=True).input_ids)
            result['prompt_len'] = prompt_len
        return result
    train_ds = train_ds.map(format_text)
    val_ds = val_ds.map(format_text)
    print(f'  Train: {len(train_ds)} | Val: {len(val_ds)}')
    print('[3/6] Loading BF16 model on GPU...')
    model = BitNetForCausalLM.from_pretrained(cfg.bitnet_model, device_map='auto', torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    print('[4/6] Replacing AutoBitLinear with LoRAAutoBitLinear (STE)...')
    replacements = replace_with_lora(model, r=cfg.bitnet_lora_r, alpha=cfg.bitnet_lora_alpha, dropout=cfg.bitnet_lora_dropout)
    print(f'  Replaced {replacements} layers')
    trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    total_params = sum((p.numel() for p in model.parameters()))
    print(f'  Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)')
    logger.log_model_info(total_params, trainable_params)
    logger.save()
    if cfg.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    print('[5/6] Preparing training...')
    device = next(model.parameters()).device

    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=cfg.max_length)
        inputs['labels'] = inputs['input_ids'].clone()
        if 'prompt_len' in batch[0]:
            for i in range(len(batch)):
                prompt_len = min(batch[i]['prompt_len'], inputs['input_ids'].shape[1])
                inputs['labels'][i, :prompt_len] = -100
            inputs['labels'][inputs['attention_mask'] == 0] = -100
        return inputs
    g = torch.Generator()
    g.manual_seed(seed)
    _worker_init = lambda worker_id: np.random.seed(seed + worker_id)
    dataloader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=cfg.dataloader_num_workers, pin_memory=True, generator=g, worker_init_fn=_worker_init, persistent_workers=True if cfg.dataloader_num_workers > 0 else False)
    val_dataloader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=cfg.dataloader_num_workers, pin_memory=True, worker_init_fn=_worker_init, persistent_workers=True if cfg.dataloader_num_workers > 0 else False)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_name = os.environ.get('BITNET_OPTIMIZER', 'adamw_torch').lower()
    if optimizer_name == 'adamw_torch':
        optimizer = AdamW(lora_params, lr=cfg.bitnet_learning_rate, weight_decay=0.01)
    elif optimizer_name == 'adamw_8bit':
        try:
            from bitsandbytes.optim import AdamW8bit
            optimizer = AdamW8bit(lora_params, lr=cfg.bitnet_learning_rate, weight_decay=0.01)
        except ImportError:
            raise ImportError('bitsandbytes is not installed. Install it with: pip install bitsandbytes. The adamw_8bit optimizer requires this package.')
    else:
        raise ValueError(f"Unknown BITNET_OPTIMIZER: '{optimizer_name}'. Expected 'adamw_torch' or 'adamw_8bit'.")
    print(f'  Optimizer: {optimizer_name}')
    adapter_dir = Path(os.environ.get('BITNET_STE_ADAPTER_DIR', f'models/bitnet-ste-adapter-seed{seed}'))
    adapter_dir.mkdir(parents=True, exist_ok=True)
    print(f'  Adapter output: {adapter_dir}')
    print('[6/6] Training LoRA with STE quantization...')
    model.train()
    global_step = 0
    total_loss = 0
    best_loss = float('inf')
    num_epochs = 1 if args.dry_run else cfg.bitnet_num_epochs
    masking_stats_logged = False
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for (batch_idx, batch) in enumerate(pbar):
            batch = {k: v.to(device) for (k, v) in batch.items()}
            if not masking_stats_logged and 'labels' in batch:
                masked_tokens = (batch['labels'] == -100).sum().item()
                total_tokens = batch['labels'].numel()
                mean_masked_pct = 100 * masked_tokens / total_tokens if total_tokens > 0 else 0
                print(f'  Completion-only loss: mean {mean_masked_pct:.1f}% of tokens masked')
                if args.dry_run:
                    print('\n  [DRY-RUN] Decode check for completion-only loss:')
                    batch_size = batch['input_ids'].shape[0]
                    for i in range(min(2, batch_size)):
                        input_ids = batch['input_ids'][i]
                        labels = batch['labels'][i]
                        prompt_mask = labels == -100
                        prompt_end = prompt_mask.sum().item()
                        print(f'  Sample {i}: prompt_len={prompt_end}, total_len={input_ids.shape[0]}')
                        print(f'    Prompt (decoded): {tokenizer.decode(input_ids[:prompt_end])[:200]}...')
                        print(f'    Completion (decoded): {tokenizer.decode(input_ids[prompt_end:])[:200]}...')
                    print()
                masking_stats_logged = True
            outputs = model(**batch)
            loss = outputs.loss / cfg.grad_accum
            loss.backward()
            total_loss += outputs.loss.item()
            if (batch_idx + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % 10 == 0:
                    avg_loss = total_loss / 10
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'step': global_step})
                    current_lr = optimizer.param_groups[0]['lr']
                    current_epoch = epoch + (batch_idx + 1) / len(dataloader)
                    logger.log_training_step(global_step, current_epoch, avg_loss, current_lr)
                    logger.update_vram_tracking()
                    logger.save()
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        save_lora_weights(model, adapter_dir, 'best')
                    total_loss = 0
                if global_step % 500 == 0 and (not args.dry_run):
                    save_lora_weights(model, adapter_dir, f'checkpoint-{global_step}')
                    torch.cuda.empty_cache()
                    print(f'\n  Running eval at step {global_step}...')
                    model.eval()
                    eval_losses = []
                    eval_correct = 0
                    eval_total = 0
                    with torch.no_grad():
                        for eval_batch in tqdm(val_dataloader, desc='  Eval', leave=False):
                            eval_batch = {k: v.to(device) for (k, v) in eval_batch.items()}
                            eval_outputs = model(**eval_batch)
                            eval_losses.append(eval_outputs.loss.item())
                            logits = eval_outputs.logits[:, :-1, :]
                            labels = eval_batch['labels'][:, 1:]
                            predictions = logits.argmax(dim=-1)
                            mask = labels != -100
                            eval_correct += (predictions[mask] == labels[mask]).sum().item()
                            eval_total += mask.sum().item()
                    eval_loss = sum(eval_losses) / len(eval_losses)
                    token_acc = eval_correct / eval_total if eval_total > 0 else 0
                    logger.log_eval_step(global_step, current_epoch, eval_loss, token_acc)
                    logger.save()
                    print(f'  Eval loss: {eval_loss:.4f}, Token acc: {token_acc:.4f}')
                    model.train()
    lora_weights = save_lora_weights(model, adapter_dir, 'final')
    torch.save(lora_weights, str(adapter_dir / 'lora_weights.pt'))
    model.config.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.finalize()
    logger.save()
    logger.print_summary()
    print(f'\nTraining complete! Saved {len(lora_weights)} LoRA params to: {adapter_dir}')
    print(f'Best loss: {best_loss:.4f}')
    print(f'\nNext steps:')
    print(f'  1. Pack base weights: python -m src.pack_ste_base --seed {seed}')
    print(f'  2. Evaluate: python -m src.eval --model-type bitnet --packed --adapter {adapter_dir}')
    notify('BitNet STE Training DONE', f'Seed {seed} done - best loss {best_loss:.4f}', '3')
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        notify('BitNet STE Training FAILED', str(e)[:200], '5')
        raise
