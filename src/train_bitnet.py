import os, torch
from pathlib import Path
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers.integrations.bitnet import BitLinear
from src.config import cfg
from src.common import get_chat_template, setup_hf_cache, apply_bitnet_quantization, load_packed_weights, LoraBitLinearWrapper

def main():
    setup_hf_cache()
    print('=' * 60)
    print(f'BitNet packed + LoRA (QVAC-style)')
    print(f'Model: {cfg.bitnet_model}')
    print(f'LoRA: r={cfg.bitnet_lora_r}, alpha={cfg.bitnet_lora_alpha}')
    print(f'LR: {cfg.bitnet_learning_rate}, epochs: {cfg.bitnet_num_epochs}')
    print('=' * 60)
    print('\n[1/6] Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(cfg.bitnet_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenizer.chat_template = get_chat_template()
    print('[2/6] Loading dataset...')
    train_ds = load_dataset('json', data_files=str(cfg.data_processed_dir / 'train.jsonl'), split='train')
    val_ds = load_dataset('json', data_files=str(cfg.data_processed_dir / 'val.jsonl'), split='train')
    if cfg.max_train_samples and len(train_ds) > cfg.max_train_samples:
        train_ds = train_ds.select(range(cfg.max_train_samples))
    if cfg.max_eval_samples and len(val_ds) > cfg.max_eval_samples:
        val_ds = val_ds.select(range(cfg.max_eval_samples))

    def format_text(example):
        return {'text': tokenizer.apply_chat_template([{'role': 'user', 'content': f"{example['instruction']}\n\n{example['input']}"}, {'role': 'assistant', 'content': example['output']}], tokenize=False)}
    train_ds = train_ds.map(format_text)
    val_ds = val_ds.map(format_text)
    print(f'  Train: {len(train_ds)} | Val: {len(val_ds)}')
    print('[3/6] Loading bf16 model and converting to packed...')
    from transformers import BitNetForCausalLM
    model = BitNetForCausalLM.from_pretrained(cfg.bitnet_model, device_map='auto', torch_dtype=torch.bfloat16)
    apply_bitnet_quantization(model)
    packed_ok = load_packed_weights(model, 'models/bitnet-packed')
    print(f'  Packed weights loaded: {packed_ok}')
    model.config.use_cache = False
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    if cfg.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    print('[4/6] Applying LoRA on BitLinear (QVAC-style)...')
    lora_config = LoraConfig(r=cfg.bitnet_lora_r, lora_alpha=cfg.bitnet_lora_alpha, target_modules=cfg.lora_target_modules, lora_dropout=cfg.bitnet_lora_dropout, bias='none', task_type='CAUSAL_LM')
    lora_config._register_custom_module({BitLinear: LoraBitLinearWrapper})
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    adapter_dir = cfg.bitnet_packed_adapter_dir
    adapter_dir.mkdir(parents=True, exist_ok=True)
    print('[5/6] Starting training...')
    training_args = SFTConfig(output_dir=str(adapter_dir), per_device_train_batch_size=cfg.batch_size, per_device_eval_batch_size=cfg.batch_size, gradient_accumulation_steps=cfg.grad_accum, learning_rate=cfg.bitnet_learning_rate, warmup_ratio=cfg.warmup_ratio, num_train_epochs=cfg.bitnet_num_epochs, logging_steps=10, eval_strategy='steps', eval_steps=500, save_strategy='steps', save_steps=1000, save_total_limit=2, load_best_model_at_end=True, metric_for_best_model='eval_loss', lr_scheduler_type=cfg.scheduler, optim=cfg.bitnet_optimizer, bf16=True, report_to='none', gradient_checkpointing=cfg.gradient_checkpointing, dataloader_num_workers=2, max_grad_norm=0.3, dataset_text_field='text', max_length=cfg.max_length)
    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds, processing_class=tokenizer)
    trainer.train()
    lora_weights = {}
    for (name, module) in model.named_modules():
        if isinstance(module, LoraBitLinearWrapper):
            prefix = name.replace('.', '_')
            lora_weights[f'{prefix}_lora_A'] = module.lora_A.data.clone()
            lora_weights[f'{prefix}_lora_B'] = module.lora_B.data.clone()
    torch.save(lora_weights, str(adapter_dir / 'lora_weights.pt'))
    model.config.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f'\nTraining complete! Saved {len(lora_weights)} LoRA params to: {adapter_dir}')
if __name__ == '__main__':
    main()
