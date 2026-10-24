import html
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import constants

KD = Path(constants.KNOWLEDGE_DIR)
CHUNKS = KD / "chunks.jsonl"
RAW = KD / "generated_raw.jsonl"
VERIFIED = KD / "verified_mcq.jsonl"
OPEN_QA = KD / "open_qa.jsonl"
REPORT = KD / "generation_report.json"

GEN_SYSTEM = (
    "You are an expert bioinformatics educator writing exam questions about "
    "genome assembly and genome annotation. You write precise, factual questions "
    "grounded ONLY in the provided source text. You output strict JSON."
)
ANSWER_SYSTEM = "You answer multiple-choice questions with a single letter."
DEFENSIBLE_SYSTEM = "You evaluate multiple-choice options precisely."

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


def clean_markdown(text):
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)
    text = re.sub(r"\{%.*?%\}", "", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\{:[^}]*\}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_text(text):
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"[ \t]+", " ", text).strip()


def chunk_text(text):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf, buf_len = [], [], 0
    for para in paragraphs:
        words = para.split()
        if buf_len + len(words) > constants.CHUNK_WORDS and buf:
            chunks.append(" ".join(buf))
            tail = " ".join(buf).split()[-constants.CHUNK_OVERLAP:]
            buf, buf_len = [" ".join(tail)], len(tail)
        buf.append(para)
        buf_len += len(words)
    if buf and buf_len >= constants.CHUNK_MIN_WORDS:
        chunks.append(" ".join(buf))
    if len(chunks) >= 2 and len(chunks[-1].split()) < constants.CHUNK_MIN_WORDS:
        chunks[-2] += "\n\n" + chunks[-1]
        chunks.pop()
    return [c for c in chunks if len(c.split()) >= constants.CHUNK_MIN_WORDS]


def chunk_sources(sources_dir):
    KD.mkdir(parents=True, exist_ok=True)
    n_docs, n_chunks = 0, 0
    with CHUNKS.open("w") as out:
        for path in sorted(Path(sources_dir).rglob("*")):
            if path.suffix.lower() not in (".md", ".txt", ".html", ".htm") or not path.is_file():
                continue
            raw = path.read_text(errors="replace")
            text = html_to_text(raw) if path.suffix.lower() in (".html", ".htm") else clean_markdown(raw)
            if len(text.split()) < constants.CHUNK_MIN_WORDS:
                continue
            doc_id = str(path.relative_to(sources_dir))
            n_docs += 1
            for i, c in enumerate(chunk_text(text)):
                out.write(json.dumps({"chunk_id": f"{doc_id}#{i}", "doc_id": doc_id,
                                      "title": path.stem, "url": "", "license": "",
                                      "text": c}, ensure_ascii=False) + "\n")
                n_chunks += 1
    print(f"chunks: {n_chunks} from {n_docs} docs -> {CHUNKS}")


def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(constants.KNOWLEDGE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(constants.KNOWLEDGE_MODEL,
                                                 torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return tokenizer, model


def generate_batch(tokenizer, model, prompts, system, n=1, temperature=0.0, max_new_tokens=3000):
    tokenizer.padding_side = "left"
    texts = [tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": p}],
        tokenize=False, add_generation_prompt=True) for p in prompts]
    enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    kwargs = {"max_new_tokens": max_new_tokens, "num_return_sequences": n,
              "pad_token_id": tokenizer.eos_token_id}
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        kwargs.update(do_sample=False)
    outputs = model.generate(**enc, **kwargs)
    n_in = enc["input_ids"].shape[1]
    texts_out = [tokenizer.decode(o[n_in:], skip_special_tokens=True) for o in outputs]
    return [texts_out[i * n:(i + 1) * n] for i in range(len(prompts))]


def extract_json(text):
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


def parse_letters(text):
    return sorted(set(re.findall(r"\b([A-D])\b", text.upper())))


def generate_all(n_mcq=4, n_open=1):
    chunks = [json.loads(l) for l in CHUNKS.open()]
    print(f"generating from {len(chunks)} chunks with {constants.KNOWLEDGE_MODEL}")
    tokenizer, model = load_llm()
    prompts = [GEN_PROMPT.format(title=c["title"], text=c["text"], n_mcq=n_mcq, n_open=n_open)
               for c in chunks]
    outputs = generate_batch(tokenizer, model, prompts, GEN_SYSTEM, temperature=0.7, max_new_tokens=3000)
    n_mcq_ok, n_open_ok = 0, 0
    with RAW.open("w") as out:
        for c, texts in zip(chunks, outputs):
            obj = extract_json(texts[0]) if texts else None
            if not obj:
                continue
            for q in obj.get("mcq", []):
                opts = q.get("options", {})
                ans = str(q.get("answer", "")).strip().upper()
                if (isinstance(opts, dict) and set(opts) == {"A", "B", "C", "D"}
                        and ans in "ABCD" and len(ans) == 1 and q.get("question")):
                    out.write(json.dumps({"type": "mcq", "chunk_id": c["chunk_id"],
                                          "doc_id": c["doc_id"], "title": c["title"],
                                          "url": c["url"], "license": c["license"],
                                          "question": q["question"].strip(),
                                          "options": {k: str(v).strip() for k, v in opts.items()},
                                          "answer": ans,
                                          "explanation": str(q.get("explanation", "")).strip()},
                                         ensure_ascii=False) + "\n")
                    n_mcq_ok += 1
            for q in obj.get("open", []):
                if q.get("question") and q.get("answer"):
                    out.write(json.dumps({"type": "open", "chunk_id": c["chunk_id"],
                                          "doc_id": c["doc_id"], "title": c["title"],
                                          "url": c["url"], "license": c["license"],
                                          "question": q["question"].strip(),
                                          "answer": q["answer"].strip()},
                                         ensure_ascii=False) + "\n")
                    n_open_ok += 1
    print(f"raw: {n_mcq_ok} mcq, {n_open_ok} open")


def verify():
    recs = [json.loads(l) for l in RAW.open()]
    recs = [r for r in recs if r["type"] == "mcq"]
    print(f"verifying {len(recs)} mcqs")
    tokenizer, model = load_llm()
    answers = generate_batch(tokenizer, model,
                             [ANSWER_PROMPT.format(question=r["question"], **r["options"]) for r in recs],
                             ANSWER_SYSTEM, n=3, temperature=0.0, max_new_tokens=4)
    defensible = generate_batch(tokenizer, model,
                                [DEFENSIBLE_PROMPT.format(question=r["question"], **r["options"]) for r in recs],
                                DEFENSIBLE_SYSTEM, n=1, temperature=0.0, max_new_tokens=10)
    stats = Counter()
    kept = []
    for r, ans3, def1 in zip(recs, answers, defensible):
        votes = [parse_letters(a) for a in ans3]
        if not (len(votes) == 3 and all(v == [r["answer"]] for v in votes)):
            stats["fail_selfconsistency"] += 1
            continue
        if parse_letters(def1[0] if def1 else "") != [r["answer"]]:
            stats["fail_one_correct"] += 1
            continue
        stats["pass"] += 1
        kept.append(r)

    def toks(q):
        return set(re.findall(r"[a-z0-9]+", q.lower()))

    seen, kept_tokens, final = set(), [], []
    for r in kept:
        key = re.sub(r"\s+", " ", r["question"].lower().strip())
        if key in seen:
            stats["dup_exact"] += 1
            continue
        t = toks(r["question"])
        if any(len(t & u) / max(1, len(t | u)) >= 0.8 for u in kept_tokens):
            stats["dup_near"] += 1
            continue
        seen.add(key)
        kept_tokens.append(t)
        final.append(r)
    stats["final"] = len(final)
    with VERIFIED.open("w") as out:
        for r in final:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    with OPEN_QA.open("w") as out:
        for l in RAW.open():
            r = json.loads(l)
            if r["type"] == "open":
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
    REPORT.write_text(json.dumps(dict(stats), indent=2))
    print("verification:", dict(stats))


def split():
    rng = random.Random(constants.SEED)
    recs = [json.loads(l) for l in VERIFIED.open()]
    docs = sorted({r["doc_id"] for r in recs})
    rng.shuffle(docs)
    test_docs = set(docs[: max(1, round(len(docs) * 0.15))])
    test_pool = [r for r in recs if r["doc_id"] in test_docs]
    trainval = [r for r in recs if r["doc_id"] not in test_docs]
    rng.shuffle(test_pool)
    rng.shuffle(trainval)
    test = test_pool[:500]
    n_val = min(500, max(100, len(trainval) // 12))
    val, train = trainval[:n_val], trainval[n_val:]

    def to_gaqa(r, idx, name):
        opts = r["options"]
        lines = ["Subject: domain_knowledge", f"Question: {r['question']}", "Options:"]
        lines += [f"{k}) {opts[k]}" for k in "ABCD"]
        return {"id": f"knowledge_{name}_{idx}", "subject": "domain_knowledge",
                "input": "\n".join(lines),
                "output": f"The correct answer is {r['answer']}: {opts[r['answer']]}"}

    for name, rows in (("train", train), ("val", val), ("test", test)):
        with (KD / f"{name}.jsonl").open("w") as out:
            for i, r in enumerate(rows):
                out.write(json.dumps(to_gaqa(r, i, name), ensure_ascii=False) + "\n")
    print(f"splits: train={len(train)} val={len(val)} test={len(test)}")


def main():
    chunk_sources(constants.KNOWLEDGE_SOURCES_DIR)
    generate_all()
    verify()
    split()


if __name__ == "__main__":
    main()
