import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
import torch
import torch.distributed.tensor as _dt
if not hasattr(_dt, 'DTensor'):
    _dt.DTensor = object
sys.path.insert(0, str(Path(__file__).parent))
from eval import evaluate_gaqa, load_model
BITNET_MODEL = os.getenv('BITNET_MODEL', 'microsoft/bitnet-b1.58-2B-4T-bf16')
MODELS = [{'name': 'BitNet-2B-zeroshot', 'type': 'bitnet', 'model': BITNET_MODEL, 'adapter': None, 'packed': True, 'quant4bit': False, 'tags': '2B base'}, {'name': 'BitNet-2B', 'type': 'bitnet', 'model': BITNET_MODEL, 'adapter': 'models/bitnet-ste-adapter-r32-seed42', 'packed': True, 'quant4bit': False, 'tags': 'ours'}, {'name': 'Mistral-7B-Instruct-v0.3', 'type': 'other', 'model': 'mistralai/Mistral-7B-Instruct-v0.3', 'adapter': None, 'packed': False, 'quant4bit': False, 'tags': '7B instruct'}, {'name': 'Qwen2.5-7B-Instruct', 'type': 'other', 'model': 'Qwen/Qwen2.5-7B-Instruct', 'adapter': None, 'packed': False, 'quant4bit': False, 'tags': '7B instruct'}]
TEST_SETS = [('heldout', 'data/heldout/test.jsonl'), ('domain-knowledge', 'data/gaqa_knowledge/test.jsonl')]
TEST_SETS = [(n, p) for (n, p) in TEST_SETS if os.path.exists(p)]
if not TEST_SETS:
    print('ERROR: No test files found in data/. Check paths.')
    sys.exit(1)

def _load_generic_model(model_name, quant4bit=False):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import BitsAndBytesConfig
    from common import get_chat_template
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.chat_template = get_chat_template()
    kw = dict(device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True)
    if quant4bit:
        kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')
        kw['device_map'] = None
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    return (model, tokenizer)

def eval_one(model_type, model_name, adapter, packed, test_jsonl, max_samples, few_shot, batch_size, quant4bit):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cfg_mod = sys.modules.get('src.config') or sys.modules.get('config')
    cfg_obj = getattr(cfg_mod, 'cfg', None) if cfg_mod else None
    old_dir = str(cfg_obj.data_processed_dir) if cfg_obj else None
    if cfg_obj:
        cfg_obj.data_processed_dir = Path(test_jsonl).parent
    try:
        if model_type == 'bitnet':
            if adapter:
                print(f'    Loading BitNet with adapter: {adapter} ...', flush=True)
            else:
                print(f'    Loading BitNet (no adapter) ...', flush=True)
            if adapter and (not Path(adapter).exists()):
                print(f'    WARNING: adapter not found at {adapter}')
            os.environ['BITNET_LORA_R'] = '32'
            os.environ['BITNET_LORA_ALPHA'] = '64'
            (model, tokenizer) = load_model('bitnet', adapter_path=adapter, packed=packed)
        else:
            (model, tokenizer) = _load_generic_model(model_name, quant4bit=quant4bit)
    except Exception as e:
        if cfg_obj and old_dir:
            cfg_obj.data_processed_dir = Path(old_dir)
        raise RuntimeError(f'Failed to load {model_name}: {e}')
    try:
        results = evaluate_gaqa(model, tokenizer, max_samples=max_samples, max_new_tokens=128, max_length=1024, few_shot=few_shot, batch_size=batch_size)
    finally:
        if cfg_obj and old_dir:
            cfg_obj.data_processed_dir = Path(old_dir)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-samples', type=int, default=200)
    ap.add_argument('--few-shot', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--output', default='results/multi_model_comparison.csv')
    ap.add_argument('--models', nargs='*', default=None, help='Filter model names (e.g. BitNet-2B)')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    rows = []
    for (mi, mspec) in enumerate(MODELS):
        if args.models and mspec['name'] not in args.models:
            continue
        t0 = time.time()
        print(f"\n{'=' * 60}")
        print(f"[{mi + 1}/{len(MODELS)}] {mspec['name']} ({mspec['tags']})")
        print(f"{'=' * 60}")
        for (ts_name, ts_path) in TEST_SETS:
            if not os.path.exists(ts_path):
                print(f'  {ts_name}: SKIP — {ts_path} not found')
                continue
            n_total = sum((1 for _ in open(ts_path)))
            n_eval = min(args.max_samples, n_total)
            print(f'  {ts_name}: {n_eval} samples ...', end='', flush=True)
            try:
                results = eval_one(mspec['type'], mspec['model'], mspec['adapter'], mspec['packed'], ts_path, max_samples=args.max_samples, few_shot=args.few_shot, batch_size=args.batch_size, quant4bit=mspec['quant4bit'])
            except Exception as e:
                import traceback
                print(f' ERROR: {e}')
                traceback.print_exc()
                rows.append({'model': mspec['name'], 'test_set': ts_name, 'accuracy': None, 'auc': None, 'n_samples': n_eval, 'error': str(e)[:200]})
                continue
            if isinstance(results, dict):
                acc = results.get('accuracy')
                auc = results.get('auc')
                per_fam = results.get('per_family_accuracy', {})
            else:
                acc = float(results) if results is not None else None
                auc = None
                per_fam = {}
            print(f' acc={acc:.1%} auc={auc:.3f}' if acc is not None and auc is not None else f' acc={acc:.1%}' if acc is not None else ' (no results)')
            for (fam, fam_acc) in per_fam.items():
                print(f'    {fam}: {fam_acc:.1%}')
            print()
            rows.append({'model': mspec['name'], 'test_set': ts_name, 'accuracy': round(acc * 100, 1) if acc is not None else None, 'auc': round(auc, 3) if auc is not None else None, 'n_samples': n_eval, **{f'acc_{k}': round(v * 100, 1) for (k, v) in per_fam.items()}})
        dt = time.time() - t0
        print(f'  [done in {dt / 60:.1f} min]')
    print(f"\n{'=' * 60}")
    print(f"{'Model':<28} {'Test set':<20} {'Acc':<7} {'AUC':<7}")
    print(f"{'-' * 62}")
    for r in rows:
        acc_s = f"{r['accuracy']:.1f}" if r.get('accuracy') is not None else 'ERR'
        auc_s = f"{r['auc']:.3f}" if r.get('auc') is not None else '-'
        print(f"{r['model']:<28} {r['test_set']:<20} {acc_s:<7} {auc_s:<7}")
    print(f"{'=' * 60}")
    import csv
    if rows:
        keys = ['model', 'test_set', 'accuracy', 'auc', 'n_samples', 'error']
        family_keys = sorted(set((k for r in rows for k in r if k.startswith('acc_'))))
        keys += family_keys
        with open(args.output, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        print(f'Results saved to {args.output}')
    print('\nPer-model averages:')
    by_model = {}
    for r in rows:
        by_model.setdefault(r['model'], []).append(r)
    for (model, rs) in by_model.items():
        accs = [r['accuracy'] for r in rs if r.get('accuracy') is not None]
        aucs = [r.get('auc') for r in rs if r.get('auc') is not None]
        avg_acc = sum(accs) / len(accs) if accs else None
        avg_auc = sum(aucs) / len(aucs) if aucs else None
        if avg_acc is None:
            print(f'  {model}: all evaluations failed (adapter missing?)')
        else:
            auc_str = f' auc={avg_auc:.3f}' if avg_auc is not None else ''
            print(f'  {model}: acc={avg_acc:.1f}%{auc_str} ({len(rs)} test sets)')
if __name__ == '__main__':
    main()
