import os, json, shutil, torch, time, numpy as np, random
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from src.config import cfg
from src.common import extract_answer, get_chat_template, setup_hf_cache, apply_bitnet_quantization, load_packed_weights, apply_lora_to_packed
from src.evaluation_logger import EvaluationLogger
from src.notify import notify

def _user_content(example):
    instr = example.get('instruction', '')
    return f"{instr}\n\n{example['input']}" if instr else example['input']

def _few_shot_line(example):
    q = example.get('instruction', '')
    return f"Question: {q}\n{example['input']}\n" if q else f"Question: {example['input']}\n"

def load_few_shot_examples(num_examples=3, seed=None):
    if seed is None:
        seed = cfg.seed
    train_ds = load_dataset('json', data_files=str(cfg.data_processed_dir / 'train.jsonl'), split='train')
    random.seed(seed)
    indices = random.sample(range(len(train_ds)), min(num_examples, len(train_ds)))
    examples = [train_ds[i] for i in indices]
    return examples

def format_few_shot_prompt(examples, tokenizer):
    few_shot_text = ''
    for (i, example) in enumerate(examples, 1):
        few_shot_text += f'Example {i}:\n'
        few_shot_text += _few_shot_line(example)
        few_shot_text += f"Answer: {example['output']}\n\n"
    return few_shot_text

def compute_perplexity(model, tokenizer, texts, batch_size=4, max_length=1024):
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(next(model.parameters()).device) for (k, v) in inputs.items()}
            labels = inputs['input_ids'].clone()
            labels[inputs['attention_mask'] == 0] = -100
            outputs = model(**inputs, labels=labels)
            losses.append(outputs.loss.item())
    return np.exp(np.mean(losses)) if losses else 0.0

def measure_latency(model, tokenizer, texts, max_new_tokens=128):
    model.eval()
    times = []
    device = next(model.parameters()).device
    eos_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')]
    use_cuda = device.type == 'cuda'
    for text in texts[:10]:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        inputs = {k: v.to(device) for (k, v) in inputs.items()}
        if use_cuda:
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_ids)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.time() - start
        generated = outputs.shape[1] - inputs['input_ids'].shape[1]
        if generated > 0:
            times.append(elapsed / generated)
    return np.median(times) if times else 0.0


def evaluate_gaqa(model, tokenizer, max_samples=cfg.eval_num_samples, max_new_tokens=cfg.eval_max_new_tokens, max_length=cfg.max_length, few_shot=0, batch_size=cfg.eval_batch_size):
    if few_shot > 0:
        print('  NOTE: few_shot > 0 is ignored for logit-based GAQA scoring')
    from collections import defaultdict
    try:
        from sklearn.metrics import roc_auc_score
        _has_sklearn = True
    except ImportError:
        _has_sklearn = False
        print('  WARNING: sklearn not available, AUC will not be computed')
    test_ds = load_dataset('json', data_files=str(cfg.data_processed_dir / 'test.jsonl'), split='train')
    test_ds = test_ds.select(range(min(max_samples, len(test_ds))))
    device = next(model.parameters()).device
    letter_token_ids = {}
    for letter in ['A', 'B', 'C', 'D']:
        ids = []
        for variant in [f' {letter}', letter, f' {letter.lower()}', letter.lower()]:
            tid = tokenizer.encode(variant, add_special_tokens=False)[0]
            ids.append(tid)
        letter_token_ids[letter] = ids
    prompts = []
    gold_letters = []
    subjects = []
    sample_ids = []
    for example in test_ds:
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': _user_content(example)}], tokenize=False, add_generation_prompt=True)
        prompt += 'The correct answer is'
        prompts.append(prompt)
        gold_letters.append(extract_answer(example['output']))
        subjects.append(example.get('subject', 'unknown'))
        sample_ids.append(example.get('id', str(len(sample_ids))))
    correct = []
    predicted_letters = []
    all_probs = []
    n_total = len(prompts)
    for i in tqdm(range(0, n_total, batch_size), desc='GAQA (logit)'):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for (k, v) in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        for j in range(len(batch)):
            idx = i + j
            letter_logits = {}
            for letter in ['A', 'B', 'C', 'D']:
                ids = letter_token_ids[letter]
                letter_logits[letter] = logits[j, ids].max().item()
            logit_vals = torch.tensor([letter_logits[l] for l in ['A', 'B', 'C', 'D']], device=device)
            probs = torch.softmax(logit_vals, dim=0)
            prob_dict = {l: probs[k].item() for (k, l) in enumerate(['A', 'B', 'C', 'D'])}
            pred = max(letter_logits, key=letter_logits.get)
            gold = gold_letters[idx]
            predicted_letters.append(pred)
            correct.append(pred == gold if gold else False)
            all_probs.append(prob_dict)
    overall = round(np.mean(correct), 4)
    family_correct = defaultdict(list)
    for (c, subj) in zip(correct, subjects):
        family_correct[subj].append(c)
    per_family = {}
    for (subj, vals) in sorted(family_correct.items()):
        per_family[subj] = round(np.mean(vals), 4)
    print(f'\n  GAQA overall accuracy: {overall:.4f}')
    for (subj, acc) in per_family.items():
        print(f'    {subj}: {acc:.4f}  ({len(family_correct[subj])} samples)')
    auc_result = None
    if _has_sklearn:
        qc_indices = [k for (k, s) in enumerate(subjects) if s == 'qc_diagnosis']
        if qc_indices:
            acceptable_letters = []
            gold_qc = []
            scores_qc = []
            for k in qc_indices:
                sample = test_ds[k]
                input_text = sample['input']
                acceptable_letter = None
                for line in input_text.split('\n'):
                    line = line.strip()
                    if line.startswith(('A)', 'B)', 'C)', 'D)')) and 'Acceptable quality' in line:
                        acceptable_letter = line[0]
                        break
                if acceptable_letter is None:
                    continue
                acceptable_letters.append(acceptable_letter)
                gold_qc.append(1 if gold_letters[k] == acceptable_letter else 0)
                scores_qc.append(all_probs[k][acceptable_letter])
            if len(set(gold_qc)) > 1:
                auc_result = roc_auc_score(gold_qc, scores_qc)
                print(f'    qc_diagnosis AUC (acceptable quality): {auc_result:.4f}  ({len(gold_qc)} samples)')
            elif len(gold_qc) > 0:
                print(f'    qc_diagnosis AUC: N/A (all one class: {set(gold_qc).pop()})  ({len(gold_qc)} samples)')
            else:
                print(f'    qc_diagnosis AUC: N/A (no samples with parseable acceptable-quality option)')
    slug = f'bitnet-seed{cfg.seed}'
    samples_out = []
    for (sid, subj, gold, pred, is_correct, probs) in zip(sample_ids, subjects, gold_letters, predicted_letters, correct, all_probs):
        samples_out.append({'id': sid, 'subject': subj, 'gold_answer': gold, 'predicted_letter': pred, 'correct': bool(is_correct), 'prob_A': probs.get('A', 0.0), 'prob_B': probs.get('B', 0.0), 'prob_C': probs.get('C', 0.0), 'prob_D': probs.get('D', 0.0)})
    samples_path = Path(cfg.output_dir) / f'gaqa_scored_predictions_{slug}.json'
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with open(samples_path, 'w') as f:
        json.dump(samples_out, f, indent=2)
    print(f'  Per-sample predictions saved to {samples_path}')
    summary = {'overall': overall, 'per_family': per_family, 'auc_acceptable_quality': auc_result, 'n_samples': n_total, 'samples_file': str(samples_path)}
    summary_path = Path(cfg.output_dir) / f'gaqa_summary_{slug}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  GAQA summary saved to {summary_path}')
    return overall


def load_model(model_type, adapter_path=None, packed=False):
    if model_type != 'bitnet':
        raise ValueError(f'Unknown model_type: {model_type}')
    model_name = cfg.bitnet_model
    tokenizer_name = cfg.bitnet_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.chat_template = get_chat_template()
    if model_type == 'bitnet':
        from transformers import BitNetForCausalLM
        if packed:
            print('  Loading model on CPU for packing...')
            model = BitNetForCausalLM.from_pretrained(model_name, device_map='cpu', dtype=torch.bfloat16)
            print('  Applying BitLinear quantization...')
            count = apply_bitnet_quantization(model)
            if count == 0:
                raise RuntimeError('Packing failed: no layers replaced')
            seed = cfg.seed if hasattr(cfg, 'seed') else 42
            if Path(f'models/bitnet-packed-ste-seed{seed}/packed_weights.pt').exists():
                packed_dir = f'models/bitnet-packed-ste-seed{seed}'
            elif Path('models/bitnet-packed-ste/packed_weights.pt').exists():
                packed_dir = 'models/bitnet-packed-ste'
            else:
                packed_dir = 'models/bitnet-packed'
            print(f'  Loading packed weights from {packed_dir}...')
            if not load_packed_weights(model, packed_dir):
                raise RuntimeError('Packing failed: could not load packed weights')
            print(f"  Moving packed model to {('GPU' if torch.cuda.is_available() else 'CPU')}...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
        else:
            model = BitNetForCausalLM.from_pretrained(model_name, device_map='auto', dtype=torch.bfloat16)
    if adapter_path and Path(adapter_path).exists():
        if packed:
            model = apply_lora_to_packed(model, adapter_path)
        else:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    if hasattr(model, 'config'):
        model.config.use_cache = True
    return (model, tokenizer)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, required=True, choices=['bitnet'])
    parser.add_argument('--adapter', type=str, default=None)
    parser.add_argument('--packed', action='store_true', help='Use packed BitNet model (from models/bitnet-packed/)')
    parser.add_argument('--benchmarks', type=str, default='gaqa')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default='logs/evaluation', help='Directory to save evaluation logs')
    parser.add_argument('--few-shot', type=int, default=0, help='Number of few-shot examples to use (0 for zero-shot)')
    parser.add_argument('--eval-batch-size', type=int, default=8, help='Batch size for batched evaluation (default: 8)')
    parser.add_argument('--samples', type=int, default=None, help='Override eval_num_samples (for quick testing)')
    args = parser.parse_args()
    setup_hf_cache()
    benchmarks = [b.strip() for b in args.benchmarks.split(',')]
    label_map = {'bitnet': 'BitNet-2B'}
    model_label = label_map[args.model_type]
    if args.packed:
        model_label = model_label.replace('2B', '2B Packed')
    seed_suffix = f'-seed{cfg.seed}'
    model_label = f'{model_label}{seed_suffix}'
    if args.few_shot > 0 and args.adapter:
        model_type_label = f'{model_label} ({args.few_shot}-shot, fine-tuned)'
    elif args.few_shot > 0:
        model_type_label = f'{model_label} ({args.few_shot}-shot)'
    elif args.adapter:
        model_type_label = f'{model_label} (fine-tuned)'
    else:
        model_type_label = f'{model_label} (zero-shot)'
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = time.strftime('%Y%m%d_%H%M%S')
        output_dir = Path(cfg.output_dir) / f'eval_{args.model_type}_{ts}'
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path_map = {'bitnet': cfg.bitnet_model}
    notify('Eval start', f'Starting {model_type_label}...', '3')
    eval_logger = EvaluationLogger(log_dir=args.log_dir, model_name=model_label, model_path=model_path_map[args.model_type], adapter_path=args.adapter, packed=args.packed, few_shot=args.few_shot)
    print(f'Loading {model_type_label}...')
    model_load_start = time.time()
    (model, tokenizer) = load_model(args.model_type, args.adapter, args.packed)
    model_load_time = time.time() - model_load_start
    print(f'  Model loaded in {model_load_time:.2f} seconds')
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = next(model.parameters()).device
    use_cuda = device.type == 'cuda'
    if use_cuda:
        try:
            vram = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f'  VRAM: {vram:.2f} GB')
            if args.packed and vram > 2.5:
                print(f'  WARNING: VRAM too high ({vram:.2f} GB), packing may have failed!')
                print(f'  Expected ~1.7 GB for packed model.')
        except Exception:
            pass
    total_params = sum((p.numel() for p in model.parameters()))
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
        torch.save(model.state_dict(), tmp.name)
        model_size_mb = Path(tmp.name).stat().st_size / (1024 * 1024)
        Path(tmp.name).unlink()
    eval_logger.log_model_info(total_params, model_size_mb)
    test_ds = load_dataset('json', data_files=str(cfg.data_processed_dir / 'test.jsonl'), split='train')
    texts = []
    for e in test_ds.select(range(min(100, len(test_ds)))):
        texts.append(tokenizer.apply_chat_template([{'role': 'user', 'content': _user_content(e)}, {'role': 'assistant', 'content': e['output']}], tokenize=False))
    ppl = compute_perplexity(model, tokenizer, texts, batch_size=cfg.eval_batch_size)
    lat = measure_latency(model, tokenizer, texts, cfg.eval_max_new_tokens)
    vram = 0.0
    if use_cuda:
        try:
            torch.cuda.reset_peak_memory_stats()
            _ = model.generate(**{k: v.to(device) for (k, v) in tokenizer('test', return_tensors='pt').items()}, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id)
            vram = torch.cuda.max_memory_allocated() / 1024 ** 3
        except Exception:
            vram = 0.0
    print(f'  Perplexity: {ppl:.2f} | Latency: {lat * 1000:.1f} ms | VRAM: {vram:.2f} GB')
    eval_logger.log_efficiency(vram, lat * 1000, ppl)
    results = {'model': model_label, 'type': model_type_label, 'perplexity': round(ppl, 2), 'latency_ms': round(lat * 1000, 1), 'peak_vram_gb': round(vram, 2), 'benchmarks': {}}
    num_samples = args.samples if args.samples is not None else cfg.eval_num_samples
    for bench in benchmarks:
        eval_logger.start_benchmark(bench)
        if bench != 'gaqa':
            print(f'  Unknown benchmark: {bench}')
            continue
        acc = evaluate_gaqa(model, tokenizer, num_samples, cfg.eval_max_new_tokens, cfg.max_length, args.few_shot, args.eval_batch_size)
        results['benchmarks'][bench] = acc
        eval_logger.end_benchmark(bench, acc, num_samples)
    seed_tag = f'-seed{cfg.seed}'
    if args.few_shot > 0 and args.adapter:
        slug = f'{args.model_type}{seed_tag}_ft_fewshot{args.few_shot}'
    elif args.few_shot > 0:
        slug = f'{args.model_type}{seed_tag}_fewshot{args.few_shot}'
    elif args.adapter:
        slug = f'{args.model_type}{seed_tag}_ft'
    else:
        slug = f'{args.model_type}{seed_tag}_zero'
    (output_dir / f'{slug}_results.json').write_text(json.dumps(results, indent=2))
    print(f'\nResults saved to {output_dir}/{slug}_results.json')
    comparison_dir = Path(cfg.output_dir) / 'comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_dir / f'{slug}_results.json', comparison_dir / f'{slug}_results.json')
    print(f'Synced to {comparison_dir}/{slug}_results.json')
    notify('Eval done', f'{model_type_label} done', '3')
    eval_logger.finalize()
    eval_logger.save()
    eval_logger.print_summary()
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
if __name__ == '__main__':
    main()
