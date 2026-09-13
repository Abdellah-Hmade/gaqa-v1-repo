"""Generate + verify GAQA domain_knowledge questions from source chunks.

Teacher: Qwen2.5-32B-Instruct via vLLM (run on A100). Two stages:

  Stage 1 (generate): per chunk, produce 3-5 MCQs + 1-2 open QA pairs,
    grounded in the chunk (answer must be supported by the text).
  Stage 2 (verify):
    a) self-consistency: teacher answers each MCQ 3x at T=0 without seeing
       the intended answer; keep only 3/3 agreement with the gold letter.
    b) one-correct-option: teacher lists ALL defensible options; keep only
       questions where exactly the gold option is defensible.
    c) exact + near-dup removal (Jaccard >= 0.8 on question tokens).

Split: by doc_id (85/15 train+val/test docs), so no near-duplicates across splits.
Test = 500 MCQs sampled from held-out docs.

Outputs (data/gaqa_knowledge/):
  generated_raw.jsonl      - all generated MCQs + open QA (pre-verification)
  verified_mcq.jsonl       - MCQs passing verification, with split labels
  open_qa.jsonl            - open-ended QA pairs (train only)
  {train,val,test}.jsonl   - GAQA-schema MCQ splits (subject=domain_knowledge)
  generation_report.json   - verification pass rates, per-doc stats

Usage (A100):
  python -m src.generate_knowledge_qa --stage generate [--limit 20]
  python -m src.generate_knowledge_qa --stage verify
  python -m src.generate_knowledge_qa --stage split
Env: KNOWLEDGE_MODEL (default Qwen/Qwen2.5-32B-Instruct), HF_CACHE_DIR
"""

import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KD = ROOT / "data" / "gaqa_knowledge"
CHUNKS = KD / "chunks.jsonl"
RAW = KD / "generated_raw.jsonl"
VERIFIED = KD / "verified_mcq.jsonl"
OPEN_QA = KD / "open_qa.jsonl"
REPORT = KD / "generation_report.json"

MODEL = os.environ.get("KNOWLEDGE_MODEL", "Qwen/Qwen2.5-32B-Instruct")
SEED = 42

GEN_SYSTEM = (
    "You are an expert bioinformatics educator writing exam questions about "
    "genome assembly and genome annotation. You write precise, factual questions "
    "grounded ONLY in the provided source text. You output strict JSON."
)

GEN_PROMPT = """Below is an excerpt from an authoritative bioinformatics training document ({title}).

--- SOURCE TEXT ---
{text}
--- END SOURCE TEXT ---

Write questions whose answers are DIRECTLY supported by the source text above.

Requirements:
- {n_mcq} multiple-choice questions (4 options A-D, exactly ONE correct).
  * The correct answer must be stated or clearly implied in the source text.
  * Distractors must be plausible but clearly wrong given domain knowledge.
  * Vary difficulty: recall, comprehension, and application.
  * Do NOT ask about Galaxy UI mechanics (which button to click), dataset
    upload steps, or tutorial-specific file names. Ask about concepts, tools,
    metrics, parameters, and best practices.
- {n_open} open-ended question-answer pairs (2-5 sentence answers).
  * Answers must be fully supported by the source text.
- Do not reference "the text", "the excerpt", or "the tutorial" in questions.
- Do not include time-sensitive facts (versions, dates).

Output STRICT JSON (no markdown fences):
{{"mcq": [{{"question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "answer": "A", "explanation": "..."}}],
  "open": [{{"question": "...", "answer": "..."}}]}}"""

ANSWER_PROMPT = """Answer this multiple-choice question about genome assembly or annotation.
Reply with ONLY the letter (A, B, C, or D).

{question}

A) {A}
B) {B}
C) {C}
D) {D}"""

DEFENSIBLE_PROMPT = """Consider this multiple-choice question about genome assembly or annotation.
List EVERY option that could be defended as correct, based on your domain knowledge.
Reply with ONLY the letters, comma-separated (e.g., "A" or "A,C").

{question}

A) {A}
B) {B}
C) {C}
D) {D}"""


def load_llm(max_len=8192):
    from vllm import LLM
    return LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.9,
               max_model_len=max_len, seed=SEED, enforce_eager=False)


def chat(llm, prompts, temperature=0.7, max_tokens=4096, system=GEN_SYSTEM, n=1):
    from vllm import SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    texts = [
        tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    sp = SamplingParams(temperature=temperature, top_p=0.9 if temperature > 0 else 1.0,
                        max_tokens=max_tokens, n=n, seed=SEED)
    outs = llm.generate(texts, sp)
    return [[o.text for o in out.outputs] for out in outs]


def extract_json(text: str):
    """Extract the first balanced {...} JSON object from model output."""
    m = re.search(r"\{", text)
    if not m:
        return None
    depth = 0
    for i in range(m.start(), len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[m.start():i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def stage_generate(limit=None, n_mcq=4, n_open=1):
    chunks = [json.loads(l) for l in CHUNKS.open()]
    if limit:
        chunks = chunks[:limit]
    print(f"Generating from {len(chunks)} chunks with {MODEL} ...")
    llm = load_llm()
    prompts = [GEN_PROMPT.format(title=c["title"], text=c["text"],
                                 n_mcq=n_mcq, n_open=n_open) for c in chunks]
    # vLLM batch; do in two passes to limit memory pressure on long outputs
    outputs = chat(llm, prompts, temperature=0.7, max_tokens=3000)
    n_mcq_ok, n_open_ok, n_fail = 0, 0, 0
    with RAW.open("w") as out:
        for c, texts in zip(chunks, outputs):
            obj = extract_json(texts[0]) if texts else None
            if not obj:
                n_fail += 1
                continue
            for q in obj.get("mcq", []):
                opts = q.get("options", {})
                ans = str(q.get("answer", "")).strip().upper()
                if (isinstance(opts, dict) and set(opts) == {"A", "B", "C", "D"}
                        and ans in "ABCD" and len(ans) == 1
                        and q.get("question")):
                    rec = {"type": "mcq", "chunk_id": c["chunk_id"],
                           "doc_id": c["doc_id"], "title": c["title"],
                           "url": c["url"], "license": c["license"],
                           "question": q["question"].strip(),
                           "options": {k: str(v).strip() for k, v in opts.items()},
                           "answer": ans,
                           "explanation": str(q.get("explanation", "")).strip()}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_mcq_ok += 1
            for q in obj.get("open", []):
                if q.get("question") and q.get("answer"):
                    rec = {"type": "open", "chunk_id": c["chunk_id"],
                           "doc_id": c["doc_id"], "title": c["title"],
                           "url": c["url"], "license": c["license"],
                           "question": q["question"].strip(),
                           "answer": q["answer"].strip()}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_open_ok += 1
    print(f"Raw: {n_mcq_ok} MCQ, {n_open_ok} open QA, {n_fail} chunk failures")


def _parse_letters(text: str):
    return sorted(set(re.findall(r"\b([A-D])\b", text.upper())))


def stage_verify():
    recs = [json.loads(l) for l in RAW.open() if json.loads(l)["type"] == "mcq"]
    print(f"Verifying {len(recs)} MCQs with {MODEL} ...")
    llm = load_llm(max_len=4096)

    # (a) self-consistency: answer 3x at T=0 (n=3 in one call)
    ans_prompts = [ANSWER_PROMPT.format(question=r["question"], **r["options"])
                   for r in recs]
    answers = chat(llm, ans_prompts, temperature=0.0, max_tokens=4, n=3,
                   system="You answer multiple-choice questions with a single letter.")
    # (b) one-correct-option check
    def_prompts = [DEFENSIBLE_PROMPT.format(question=r["question"], **r["options"])
                   for r in recs]
    defensible = chat(llm, def_prompts, temperature=0.0, max_tokens=10,
                      system="You evaluate multiple-choice options precisely.")

    stats = Counter()
    kept = []
    for r, ans3, def1 in zip(recs, answers, defensible):
        votes = [_parse_letters(a) for a in ans3]
        consistent = (len(votes) == 3 and all(v == [r["answer"]] for v in votes))
        if not consistent:
            stats["fail_selfconsistency"] += 1
            continue
        def_letters = _parse_letters(def1[0]) if def1 else []
        if def_letters != [r["answer"]]:
            stats["fail_one_correct"] += 1
            continue
        stats["pass"] += 1
        r["verified"] = True
        kept.append(r)

    # (c) dedup: exact + Jaccard near-dup on question tokens
    def toks(q):
        return set(re.findall(r"[a-z0-9]+", q.lower()))
    seen_exact, kept_tokens, final = set(), [], []
    for r in kept:
        key = re.sub(r"\s+", " ", r["question"].lower().strip())
        if key in seen_exact:
            stats["dup_exact"] += 1
            continue
        t = toks(r["question"])
        if any(len(t & u) / max(1, len(t | u)) >= 0.8 for u in kept_tokens):
            stats["dup_near"] += 1
            continue
        seen_exact.add(key)
        kept_tokens.append(t)
        final.append(r)

    stats["final"] = len(final)
    with VERIFIED.open("w") as out:
        for r in final:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    # open QA: keep all (verified via grounding in generation prompt)
    with OPEN_QA.open("w") as out:
        for l in RAW.open():
            r = json.loads(l)
            if r["type"] == "open":
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
    REPORT.write_text(json.dumps(dict(stats), indent=2))
    print("Verification:", dict(stats))


def stage_split():
    rng = random.Random(SEED)
    recs = [json.loads(l) for l in VERIFIED.open()]
    docs = sorted({r["doc_id"] for r in recs})
    rng.shuffle(docs)
    n_test_docs = max(1, round(len(docs) * 0.15))
    test_docs = set(docs[:n_test_docs])
    test_pool = [r for r in recs if r["doc_id"] in test_docs]
    trainval = [r for r in recs if r["doc_id"] not in test_docs]
    rng.shuffle(test_pool)
    test = test_pool[:500]
    rng.shuffle(trainval)
    n_val = min(500, max(100, len(trainval) // 12))
    val, train = trainval[:n_val], trainval[n_val:]

    def to_gaqa(r, idx, split):
        opts = r["options"]
        lines = ["Subject: domain_knowledge", f"Question: {r['question']}", "Options:"]
        lines += [f"{k}) {opts[k]}" for k in "ABCD"]
        return {"id": f"knowledge_{split}_{idx}", "subject": "domain_knowledge",
                "input": "\n".join(lines),
                "output": f"The correct answer is {r['answer']}: {opts[r['answer']]}",
                "_source_url": r["url"], "_doc_id": r["doc_id"],
                "_explanation": r.get("explanation", "")}

    for name, split_recs in (("train", train), ("val", val), ("test", test)):
        with (KD / f"{name}.jsonl").open("w") as out:
            for i, r in enumerate(split_recs):
                out.write(json.dumps(to_gaqa(r, i, name), ensure_ascii=False) + "\n")
    print(f"Splits: train={len(train)} val={len(val)} test={len(test)} "
          f"(test from {len(test_docs)} held-out docs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["generate", "verify", "split"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-mcq", type=int, default=4)
    ap.add_argument("--n-open", type=int, default=1)
    args = ap.parse_args()
    if args.stage == "generate":
        stage_generate(args.limit, args.n_mcq, args.n_open)
    elif args.stage == "verify":
        stage_verify()
    else:
        stage_split()


if __name__ == "__main__":
    main()
