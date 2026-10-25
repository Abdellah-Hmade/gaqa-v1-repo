import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
KD = ROOT / 'data' / 'gaqa_knowledge'
CHUNKS = KD / 'chunks.jsonl'
RAW = KD / 'generated_raw.jsonl'
VERIFIED = KD / 'verified_mcq.jsonl'
OPEN_QA = KD / 'open_qa.jsonl'
REPORT = KD / 'generation_report.json'
MODEL = os.environ.get('KNOWLEDGE_MODEL', 'Qwen/Qwen2.5-32B-Instruct')
SEED = 42
GEN_SYSTEM = 'You are an expert bioinformatics educator writing exam questions about genome assembly and genome annotation. You write precise, factual questions grounded ONLY in the provided source text. You output strict JSON.'
GEN_PROMPT = 'Below is an excerpt from an authoritative bioinformatics training document ({title}).\n\n--- SOURCE TEXT ---\n{text}\n--- END SOURCE TEXT ---\n\nWrite questions whose answers are DIRECTLY supported by the source text above.\n\nRequirements:\n- {n_mcq} multiple-choice questions (4 options A-D, exactly ONE correct).\n  * The correct answer must be stated or clearly implied in the source text.\n  * Distractors must be plausible but clearly wrong given domain knowledge.\n  * Vary difficulty: recall, comprehension, and application.\n  * Do NOT ask about Galaxy UI mechanics (which button to click), dataset\n    upload steps, or tutorial-specific file names. Ask about concepts, tools,\n    metrics, parameters, and best practices.\n- {n_open} open-ended question-answer pairs (2-5 sentence answers).\n  * Answers must be fully supported by the source text.\n- Do not reference "the text", "the excerpt", or "the tutorial" in questions.\n- Do not include time-sensitive facts (versions, dates).\n\nOutput STRICT JSON (no markdown fences):\n{{"mcq": [{{"question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "answer": "A", "explanation": "..."}}],\n  "open": [{{"question": "...", "answer": "..."}}]}}'
ANSWER_PROMPT = 'Answer this multiple-choice question about genome assembly or annotation.\nReply with ONLY the letter (A, B, C, or D).\n\n{question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}'
DEFENSIBLE_PROMPT = 'Consider this multiple-choice question about genome assembly or annotation.\nList EVERY option that could be defended as correct, based on your domain knowledge.\nReply with ONLY the letters, comma-separated (e.g., "A" or "A,C").\n\n{question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}'

def load_llm(max_len=32768):
    from vllm import LLM
    return LLM(model=MODEL, dtype='bfloat16', gpu_memory_utilization=0.9, max_model_len=max_len, seed=SEED, enforce_eager=False)

def chat(llm, prompts, temperature=0.7, max_tokens=4096, system=GEN_SYSTEM, n=1):
    from vllm import SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    texts = [tok.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    actual_temp = temperature if temperature > 0 else 0.01
    sp = SamplingParams(temperature=actual_temp, top_p=1.0, max_tokens=max_tokens, n=n, seed=SEED)
    outs = llm.generate(texts, sp)
    return [[o.text for o in out.outputs] for out in outs]

def extract_json(text: str):
    m = re.search('\\{', text)
    if not m:
        return None
    depth = 0
    for i in range(m.start(), len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[m.start():i + 1])
                except json.JSONDecodeError:
                    return None
    return None

def stage_generate(limit=None, n_mcq=4, n_open=1):
    from transformers import AutoTokenizer
    chunks = [json.loads(l) for l in CHUNKS.open()]
    if limit:
        chunks = chunks[:limit]
    print(f'Generating from {len(chunks)} chunks with {MODEL} ...')
    llm = load_llm()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    max_input_tokens = 32768 - 3000
    dummy = {'title': 'X', 'text': 'Y', 'n_mcq': n_mcq, 'n_open': n_open}
    template_str = GEN_PROMPT.format(**dummy)
    full_dummy = tok.apply_chat_template([{'role': 'system', 'content': GEN_SYSTEM}, {'role': 'user', 'content': template_str}], tokenize=True, add_generation_prompt=True)
    overhead = len(full_dummy)
    print(f'  prompt overhead: ~{overhead} tokens')
    prompts = []
    for c in chunks:
        text = c['text']
        text_tokens = tok.encode(text)
        budget = max_input_tokens - overhead - 20
        if len(text_tokens) > budget:
            text_tokens = text_tokens[:budget]
            text = tok.decode(text_tokens, skip_special_tokens=True) + '\n...[truncated]...'
        prompts.append(GEN_PROMPT.format(title=c['title'], text=text, n_mcq=n_mcq, n_open=n_open))
    outputs = chat(llm, prompts, temperature=0.7, max_tokens=3000)
    (n_mcq_ok, n_open_ok, n_fail) = (0, 0, 0)
    with RAW.open('w') as out:
        for (c, texts) in zip(chunks, outputs):
            obj = extract_json(texts[0]) if texts else None
            if not obj:
                n_fail += 1
                continue
            for q in obj.get('mcq', []):
                opts = q.get('options', {})
                ans = str(q.get('answer', '')).strip().upper()
                if isinstance(opts, dict) and set(opts) == {'A', 'B', 'C', 'D'} and (ans in 'ABCD') and (len(ans) == 1) and q.get('question'):
                    rec = {'type': 'mcq', 'chunk_id': c['chunk_id'], 'doc_id': c['doc_id'], 'title': c['title'], 'url': c['url'], 'license': c['license'], 'question': q['question'].strip(), 'options': {k: str(v).strip() for (k, v) in opts.items()}, 'answer': ans, 'explanation': str(q.get('explanation', '')).strip()}
                    out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    n_mcq_ok += 1
            for q in obj.get('open', []):
                if q.get('question') and q.get('answer'):
                    rec = {'type': 'open', 'chunk_id': c['chunk_id'], 'doc_id': c['doc_id'], 'title': c['title'], 'url': c['url'], 'license': c['license'], 'question': q['question'].strip(), 'answer': q['answer'].strip()}
                    out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    n_open_ok += 1
    print(f'Raw: {n_mcq_ok} MCQ, {n_open_ok} open QA, {n_fail} chunk failures')

def _parse_letters(text: str):
    return sorted(set(re.findall('\\b([A-D])\\b', text.upper())))

def stage_verify():
    recs = [json.loads(l) for l in RAW.open() if json.loads(l)['type'] == 'mcq']
    print(f'Verifying {len(recs)} MCQs with {MODEL} ...')
    llm = load_llm(max_len=4096)
    ans_prompts = [ANSWER_PROMPT.format(question=r['question'], **r['options']) for r in recs]
    answers = chat(llm, ans_prompts, temperature=0.0, max_tokens=4, n=3, system='You answer multiple-choice questions with a single letter.')
    def_prompts = [DEFENSIBLE_PROMPT.format(question=r['question'], **r['options']) for r in recs]
    defensible = chat(llm, def_prompts, temperature=0.0, max_tokens=10, system='You evaluate multiple-choice options precisely.')
    stats = Counter()
    kept = []
    for (r, ans3, def1) in zip(recs, answers, defensible):
        votes = [_parse_letters(a) for a in ans3]
        consistent = len(votes) == 3 and all((v == [r['answer']] for v in votes))
        if not consistent:
            stats['fail_selfconsistency'] += 1
            continue
        def_letters = _parse_letters(def1[0]) if def1 else []
        if def_letters != [r['answer']]:
            stats['fail_one_correct'] += 1
            continue
        stats['pass'] += 1
        r['verified'] = True
        kept.append(r)

    def toks(q):
        return set(re.findall('[a-z0-9]+', q.lower()))
    (seen_exact, kept_tokens, final) = (set(), [], [])
    for r in kept:
        key = re.sub('\\s+', ' ', r['question'].lower().strip())
        if key in seen_exact:
            stats['dup_exact'] += 1
            continue
        t = toks(r['question'])
        if any((len(t & u) / max(1, len(t | u)) >= 0.8 for u in kept_tokens)):
            stats['dup_near'] += 1
            continue
        seen_exact.add(key)
        kept_tokens.append(t)
        final.append(r)
    stats['final'] = len(final)
    with VERIFIED.open('w') as out:
        for r in final:
            out.write(json.dumps(r, ensure_ascii=False) + '\n')
    with OPEN_QA.open('w') as out:
        for l in RAW.open():
            r = json.loads(l)
            if r['type'] == 'open':
                out.write(json.dumps(r, ensure_ascii=False) + '\n')
    REPORT.write_text(json.dumps(dict(stats), indent=2))
    print('Verification:', dict(stats))

def stage_split():
    rng = random.Random(SEED)
    recs = [json.loads(l) for l in VERIFIED.open()]
    docs = sorted({r['doc_id'] for r in recs})
    rng.shuffle(docs)
    n_test_docs = max(1, round(len(docs) * 0.15))
    test_docs = set(docs[:n_test_docs])
    test_pool = [r for r in recs if r['doc_id'] in test_docs]
    trainval = [r for r in recs if r['doc_id'] not in test_docs]
    rng.shuffle(test_pool)
    test = test_pool[:500]
    rng.shuffle(trainval)
    n_val = min(500, max(100, len(trainval) // 12))
    (val, train) = (trainval[:n_val], trainval[n_val:])

    def to_gaqa(r, idx, split):
        opts = r['options']
        lines = ['Subject: domain_knowledge', f"Question: {r['question']}", 'Options:']
        lines += [f'{k}) {opts[k]}' for k in 'ABCD']
        return {'id': f'knowledge_{split}_{idx}', 'subject': 'domain_knowledge', 'input': '\n'.join(lines), 'output': f"The correct answer is {r['answer']}: {opts[r['answer']]}", '_source_url': r['url'], '_doc_id': r['doc_id'], '_explanation': r.get('explanation', '')}
    for (name, split_recs) in (('train', train), ('val', val), ('test', test)):
        with (KD / f'{name}.jsonl').open('w') as out:
            for (i, r) in enumerate(split_recs):
                out.write(json.dumps(to_gaqa(r, i, name), ensure_ascii=False) + '\n')
    print(f'Splits: train={len(train)} val={len(val)} test={len(test)} (test from {len(test_docs)} held-out docs)')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['generate', 'verify', 'split'], required=True)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--n-mcq', type=int, default=4)
    ap.add_argument('--n-open', type=int, default=1)
    args = ap.parse_args()
    if args.stage == 'generate':
        stage_generate(args.limit, args.n_mcq, args.n_open)
    elif args.stage == 'verify':
        stage_verify()
    else:
        stage_split()
if __name__ == '__main__':
    main()
