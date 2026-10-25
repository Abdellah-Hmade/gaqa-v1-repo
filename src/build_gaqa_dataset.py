import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from src.build_common import QC_DIAGNOSIS_TEMPLATES, REMEDIATION_TEMPLATES, TOOL_SELECTION_TEMPLATES, FAILURE_MODES, make_question_output, make_input_text, assign_letters, generate_tool_scenarios
from src.build_realval_dataset import gather_qc_metrics, render_metrics, add_metric_noise
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'realval'
OUT = ROOT / 'data' / 'processed'
HELDOUT = ROOT / 'data' / 'heldout'
KNOWLEDGE = ROOT / 'data' / 'gaqa_knowledge'
MODES = ['low_coverage', 'contamination', 'misassembly', 'acceptable']

def rule_label(m: dict, coverage: float):
    checkm_contam = m.get('checkm_contamination_pct')
    busco_dup = m.get('busco_duplicated_pct')
    busco_comp = m.get('busco_complete_pct')
    misassemblies = m.get('num_misassemblies')
    genome_frac = m.get('genome_fraction_pct')
    if checkm_contam is not None and checkm_contam >= 5.0 or (busco_dup is not None and busco_dup >= 8.0):
        return 'contamination'
    if coverage is not None and coverage <= 20 and (busco_comp is not None and busco_comp < 80.0):
        return 'low_coverage'
    if misassemblies is not None and misassemblies >= 20 or (genome_frac is not None and genome_frac < 80.0 and (coverage is not None) and (coverage > 20)):
        return 'misassembly'
    if busco_comp is not None and busco_comp >= 90.0 and (genome_frac is None or genome_frac >= 85.0) and (checkm_contam is None or checkm_contam < 5.0):
        return 'acceptable'
    return None
CONTINUOUS_KEYS = {'n50', 'largest_contig', 'total_length', 'num_contigs', 'gc_content_pct', 'busco_complete_pct', 'busco_duplicated_pct', 'busco_fragmented_pct', 'busco_missing_pct', 'checkm_completeness_pct', 'checkm_contamination_pct', 'genome_fraction_pct', 'num_misassemblies', 'mismatches_per_100kbp', 'sequencing_depth'}

def jitter_vector(m: dict, rng, mag=0.1):
    out = dict(m)
    for (k, v) in list(out.items()):
        if k not in CONTINUOUS_KEYS or not isinstance(v, (int, float)):
            continue
        f = 1.0 + rng.uniform(-mag, mag)
        nv = v * f
        if isinstance(v, int):
            nv = max(0, round(nv))
        else:
            nv = max(0.0, round(nv, 2))
        if k.endswith('_pct'):
            nv = min(nv, 100.0)
        out[k] = nv
    if 'n50' in out and 'largest_contig' in out and (out['n50'] > out['largest_contig']):
        out['n50'] = max(1, int(out['largest_contig'] * rng.uniform(0.1, 0.8)))
    if 'largest_contig' in out and 'total_length' in out and (out['largest_contig'] > out['total_length']):
        out['total_length'] = int(out['largest_contig'] * rng.uniform(1.0, 1.3))
    return out

def render_qc_question(metrics, label, rng, uid):
    template = rng.choice(QC_DIAGNOSIS_TEMPLATES)
    metrics_text = render_metrics(metrics, rng, use_alternative_labels=rng.random() < 0.5)
    correct = FAILURE_MODES[label]['description']
    distractors = [FAILURE_MODES[m]['description'] for m in MODES if m != label]
    options = [correct] + distractors
    rng.shuffle(options)
    letter = assign_letters(options, correct)
    question = template.replace('{metrics}', metrics_text)
    return {'id': f'qc_{uid}', 'subject': 'qc_diagnosis', 'input': make_input_text('qc_diagnosis', question, options), 'output': make_question_output(letter, correct)}

def render_remediation_question(metrics, label, rng, uid):
    template = rng.choice(REMEDIATION_TEMPLATES)
    metrics_text = render_metrics(metrics, rng, use_alternative_labels=rng.random() < 0.5)
    correct = FAILURE_MODES[label]['remediation']
    distractors = [FAILURE_MODES[m]['remediation'] for m in MODES if m != label]
    options = [correct] + distractors
    rng.shuffle(options)
    letter = assign_letters(options, correct)
    question = template.replace('{metrics}', metrics_text)
    return {'id': f'rem_{uid}', 'subject': 'remediation', 'input': make_input_text('remediation', question, options), 'output': make_question_output(letter, correct)}

def render_corpus_tool_questions(cond, manifest_row, rng, uid):
    qs = []
    size = float(manifest_row['genome_size_mb'])
    cov = cond['coverage']
    contam = int(cond['contamination_pct'])
    sample = 'pure isolate' if contam == 0 else f'isolate with ~{contam}% foreign DNA'
    stats = f"Organism: {cond['organism']} (genome ~{size:.1f} Mb)\nSequencing technology: Illumina (paired-end)\nSequencing depth: {cov}x\nSample type: {sample}"
    template = rng.choice(TOOL_SELECTION_TEMPLATES)
    cfg = cond['assembler_config']
    if cfg == 'megahit':
        correct = 'SPAdes (isolate assembler)'
        distractors = ['MEGAHIT (metagenome assembler)', 'Flye (long-read assembler)', 'Hifiasm (HiFi assembler)']
    elif cfg == 'wrong':
        correct = 'SPAdes with default k-mer range (e.g., 21,33,55,77)'
        distractors = ['SPAdes with a single short k-mer (k=21)', 'MEGAHIT (metagenome assembler)', 'Flye (long-read assembler)']
    elif cov <= 20:
        correct = 'SPAdes with smaller k-mers (e.g., 21,33,55) for low coverage'
        distractors = ['SPAdes with large k-mers only (e.g., 77,99,127)', 'MEGAHIT (metagenome assembler)', 'Flye (long-read assembler)']
    else:
        correct = 'SPAdes with default k-mer range (e.g., 21,33,55,77)'
        distractors = ['MEGAHIT (metagenome assembler)', 'Flye (long-read assembler)', 'Hifiasm (HiFi assembler)']
    options = [correct] + distractors
    rng.shuffle(options)
    letter = assign_letters(options, correct)
    question = template.replace('{stats}', stats)
    qs.append({'id': f'tool_{uid}', 'subject': 'tool_selection', 'input': make_input_text('tool_selection', question, options), 'output': make_question_output(letter, correct)})
    return qs

def render_scenario_tool(stats_text, correct, distractors, rng, uid):
    template = rng.choice(TOOL_SELECTION_TEMPLATES)
    options = [correct] + list(distractors)
    rng.shuffle(options)
    letter = assign_letters(options, correct)
    question = template.replace('{stats}', stats_text)
    return {'id': f'tool_s_{uid}', 'subject': 'tool_selection', 'input': make_input_text('tool_selection', question, options), 'output': make_question_output(letter, correct)}

def load_corpus(data_dir: Path):
    conds = list(csv.DictReader((data_dir / 'conditions.csv').open()))
    manifest = {r['organism']: r for r in csv.DictReader((data_dir / 'manifest_full.csv').open())}
    corpus = []
    for cond in conds:
        cid = cond['condition_id']
        metrics = gather_qc_metrics(cid, str(data_dir / 'qc_results'))
        if not metrics:
            continue
        metrics = dict(metrics)
        metrics['sequencing_depth'] = int(cond['coverage'])
        corpus.append({'condition_id': cid, 'organism': cond['organism'], 'coverage': int(cond['coverage']), 'contamination_pct': int(cond['contamination_pct']), 'assembler_config': cond['assembler_config'], 'label_construction': cond['label_qc'], 'split': manifest.get(cond['organism'], {}).get('split', 'train'), 'manifest_row': manifest.get(cond['organism'], {}), 'metrics': metrics})
    return corpus

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=str(DATA), help='corpus dir (conditions.csv + qc_results/)')
    ap.add_argument('--jitter', type=int, default=60, help='jittered copies per train condition vector')
    ap.add_argument('--jitter-mag', type=float, default=0.1)
    ap.add_argument('--tool-scenarios', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-knowledge', action='store_true', help='Skip domain knowledge merge')
    args = ap.parse_args()
    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir)
    corpus = load_corpus(data_dir)
    if not corpus:
        print('FATAL: no QC metrics found — run the pipeline qc stage first.')
        return 1
    print(f"Corpus: {len(corpus)} conditions with QC metrics ({sum((1 for c in corpus if c['split'] == 'train'))} train, {sum((1 for c in corpus if c['split'] == 'heldout'))} heldout)")
    expert_overrides = {}
    review_path = data_dir / 'expert_review_qc.csv'
    if review_path.exists():
        with review_path.open() as f:
            for row in csv.DictReader(f):
                if 'final_label' in row and row['final_label'].strip():
                    expert_overrides[row['condition_id']] = row['final_label'].strip()
    stats = Counter()
    review_rows = []
    for c in corpus:
        rl = rule_label(c['metrics'], c['coverage'])
        cl = c['label_construction']
        cid = c['condition_id']
        if rl is None:
            c['label'] = cl
            c['hard_case'] = True
            stats['ambiguous->construction'] += 1
        elif rl == cl:
            c['label'] = rl
            c['hard_case'] = False
            stats['agree'] += 1
        elif cid in expert_overrides:
            c['label'] = expert_overrides[cid]
            c['hard_case'] = True
            stats['conflict->expert'] += 1
            review_rows.append({'condition_id': cid, 'organism': c['organism'], 'coverage': c['coverage'], 'contamination_pct': c['contamination_pct'], 'assembler_config': c['assembler_config'], 'rule_label': rl, 'construction_label': cl, 'final_label': expert_overrides[cid], 'busco_complete': c['metrics'].get('busco_complete_pct'), 'busco_dup': c['metrics'].get('busco_duplicated_pct'), 'checkm_contam': c['metrics'].get('checkm_contamination_pct'), 'misassemblies': c['metrics'].get('num_misassemblies'), 'genome_fraction': c['metrics'].get('genome_fraction_pct')})
        else:
            c['label'] = rl
            c['hard_case'] = True
            stats['conflict->rule'] += 1
            review_rows.append({'condition_id': cid, 'organism': c['organism'], 'coverage': c['coverage'], 'contamination_pct': c['contamination_pct'], 'assembler_config': c['assembler_config'], 'rule_label': rl, 'construction_label': cl, 'busco_complete': c['metrics'].get('busco_complete_pct'), 'busco_dup': c['metrics'].get('busco_duplicated_pct'), 'checkm_contam': c['metrics'].get('checkm_contamination_pct'), 'misassemblies': c['metrics'].get('num_misassemblies'), 'genome_fraction': c['metrics'].get('genome_fraction_pct')})
    labeled = [c for c in corpus if c['label'] in MODES]
    n_conf = stats['conflict->rule'] + stats.get('conflict->expert', 0)
    agree_rate = stats['agree'] / max(1, len(labeled))
    print(f"Labeling: agree={stats['agree']} ({agree_rate:.0%}), ambiguous={stats['ambiguous->construction']}, conflicts={stats.get('conflict->rule', 0)} (+{stats.get('conflict->expert', 0)} expert overrides)")
    with (data_dir / 'expert_review_qc.csv').open('w', newline='') as f:
        if review_rows:
            w = csv.DictWriter(f, fieldnames=list(review_rows[0].keys()))
            w.writeheader()
            w.writerows(review_rows)
    HELDOUT.mkdir(parents=True, exist_ok=True)
    heldout_qs = []
    for c in labeled:
        if c['split'] != 'heldout':
            continue
        uid = c['condition_id']
        heldout_qs.append(render_qc_question(c['metrics'], c['label'], rng, uid + '_h'))
        heldout_qs.append(render_remediation_question(c['metrics'], c['label'], rng, uid + '_h'))
        heldout_qs += render_corpus_tool_questions(c, c['manifest_row'], rng, uid + '_h')
    with (HELDOUT / 'test.jsonl').open('w') as f:
        for q in heldout_qs:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')
    print(f'Heldout test: {len(heldout_qs)} questions -> {HELDOUT}/test.jsonl')
    train_conds = [c for c in labeled if c['split'] == 'train']
    rng.shuffle(train_conds)
    n_val_conds = max(5, len(train_conds) // 10)
    val_conds = set((c['condition_id'] for c in train_conds[:n_val_conds]))
    (train_qs, val_qs) = ([], [])
    uid = 0
    for c in train_conds:
        is_val = c['condition_id'] in val_conds
        n_copies = 1 if is_val else args.jitter
        for j in range(n_copies):
            vec = c['metrics'] if j == 0 else jitter_vector(c['metrics'], rng, args.jitter_mag)
            if rng.random() < 0.25:
                vec = add_metric_noise(dict(vec), rng, noise_prob=1.0)
            uid += 1
            qc = render_qc_question(vec, c['label'], rng, uid)
            rem = render_remediation_question(vec, c['label'], rng, uid)
            qc['_condition_id'] = rem['_condition_id'] = c['condition_id']
            qc['_organism'] = rem['_organism'] = c['organism']
            qc['_jitter'] = rem['_jitter'] = j
            (val_qs if is_val else train_qs).extend([qc, rem])
    for c in train_conds:
        uid += 1
        for q in render_corpus_tool_questions(c, c['manifest_row'], rng, uid):
            q['_condition_id'] = c['condition_id']
            q['_organism'] = c['organism']
            (val_qs if c['condition_id'] in val_conds else train_qs).append(q)
    scen_rng = random.Random(args.seed + 1)
    for (i, (stats_text, correct, distractors)) in enumerate(generate_tool_scenarios(scen_rng)[:args.tool_scenarios]):
        uid += 1
        train_qs.append(render_scenario_tool(stats_text, correct, distractors, rng, uid))
    n_know = 0
    n_know_open = 0
    if not args.skip_knowledge:
        if (KNOWLEDGE / 'train.jsonl').exists():
            for line in (KNOWLEDGE / 'train.jsonl').open():
                train_qs.append(json.loads(line))
                n_know += 1
        if (KNOWLEDGE / 'val.jsonl').exists():
            for line in (KNOWLEDGE / 'val.jsonl').open():
                val_qs.append(json.loads(line))
        if (KNOWLEDGE / 'open_qa.jsonl').exists():
            for line in (KNOWLEDGE / 'open_qa.jsonl').open():
                r = json.loads(line)
                if r.get('type') != 'open':
                    continue
                uid += 1
                train_qs.append({'id': f'know_open_{uid}', 'subject': 'domain_knowledge', 'input': f"Subject: domain_knowledge\nQuestion: {r['question']}", 'output': r['answer'], '_source_url': r.get('url', ''), '_doc_id': r.get('doc_id', '')})
                n_know_open += 1
    rng.shuffle(train_qs)
    rng.shuffle(val_qs)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'train.jsonl').open('w') as f:
        for q in train_qs:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')
    with (OUT / 'val.jsonl').open('w') as f:
        for q in val_qs:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')
    report = {'corpus_conditions': len(corpus), 'label_agreement': agree_rate, 'label_stats': dict(stats), 'train': len(train_qs), 'val': len(val_qs), 'heldout_test': len(heldout_qs), 'knowledge_mcq': n_know, 'knowledge_open': n_know_open, 'train_subjects': dict(Counter((q['subject'] for q in train_qs))), 'val_subjects': dict(Counter((q['subject'] for q in val_qs))), 'heldout_subjects': dict(Counter((q['subject'] for q in heldout_qs))), 'jitter': args.jitter, 'seed': args.seed}
    (data_dir / 'build_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
