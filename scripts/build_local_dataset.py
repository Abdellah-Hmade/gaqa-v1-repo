"""Build a small local GAQA dataset WITHOUT Zenodo, using the original
data_pipeline generators (build_gaqa_dataset.py). Writes
outputs/gaqa_v1_local/{gaqa_v1_train,gaqa_v1_val,gaqa_v1_heldout_test,
gaqa_v1_knowledge_test}.jsonl in the released schema, so train.py / eval.py can
consume it via --data-dir.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_pipeline"))

from src.build_gaqa_dataset import (  # noqa: E402
    generate_samples_qc_diagnosis,
    generate_samples_remediation,
    generate_samples_tool_selection,
)

OUT = Path("outputs/gaqa_v1_local")
FAMILY = {"qc_diagnosis": "qc", "remediation": "rem", "tool_selection": "tool"}

# Small synthetic domain-knowledge split (offline; mirrors the released
# gaqa_v1_knowledge_test.jsonl schema). The real 181-question split comes from
# Zenodo / data_pipeline knowledge generation (needs a vLLM endpoint).
_KNOWLEDGE = [
    ("What is the main purpose of the de Bruijn graph in short-read assembly?",
     "Representing k-mer overlaps between reads to resolve contiguous sequences",
     "Aligning reads to a reference genome",
     "Estimating genome size from read depth",
     "Detecting single-nucleotide variants"),
    ("What does an N50 of 50 kbp tell you about an assembly?",
     "Half the genome is in contigs of at least 50 kbp",
     "The assembly has exactly 50 contigs",
     "Average read length is 50 kbp",
     "Coverage is 50x"),
    ("High BUSCO duplication in a single isolate usually indicates:",
     "Contamination or inflated gene copy number",
     "Excellent assembly completeness",
     "Low sequencing depth",
     "Short contig sizes"),
    ("Why do paired-end reads improve scaffolding?",
     "Insert sizes span repeats, linking distant contigs",
     "They double the sequencing depth",
     "They are error-free",
     "They replace the need for assembly"),
    ("The CheckM 'contamination' metric estimates:",
     "The fraction of single-copy genes present in multiple copies",
     "The amount of sequencing error",
     "The GC bias of the library",
     "The proportion of gaps"),
]


def knowledge_rows():
    rng = random.Random(0)
    out = []
    for i, (q, correct, *distractors) in enumerate(_KNOWLEDGE):
        options = [correct] + list(distractors)
        rng.shuffle(options)
        letter = "ABCD"[options.index(correct)]
        opts = "\n".join(f"{c}) {o}" for c, o in zip("ABCD", options))
        out.append({
            "id": f"knowledge_test_{i}",
            "subject": "domain_knowledge",
            "input": f"Subject: domain_knowledge\nQuestion: {q}\nOptions:\n{opts}",
            "output": f"The correct answer is {letter}: {correct}.",
            "_source_url": "local-synthetic", "_doc_id": "local-synthetic",
        })
    return out


def rows(samples, subject, prefix, tag=""):
    return [{"id": f"{prefix}_{tag}{i}", "subject": subject,
             "input": s["input"], "output": s["output"],
             "_condition_id": "LC01", "_organism": "local", "_jitter": ""}
            for i, s in enumerate(samples)]


def main():
    n = 12
    rng = random.Random(0)
    train, val, heldout = [], [], []
    for subject in ("qc_diagnosis", "remediation", "tool_selection"):
        gen = {"qc_diagnosis": generate_samples_qc_diagnosis,
               "remediation": generate_samples_remediation,
               "tool_selection": generate_samples_tool_selection}[subject]
        samples = gen(2 * n, rng)
        train += rows(samples[:n], subject, FAMILY[subject])
        val += rows(samples[n:n + 4], subject, FAMILY[subject], "v")
        heldout += rows(samples[n + 4:], subject, FAMILY[subject], "h")

    OUT.mkdir(parents=True, exist_ok=True)
    for name, r in [("train", train), ("val", val), ("heldout_test", heldout),
                    ("knowledge_test", knowledge_rows())]:
        p = OUT / f"gaqa_v1_{name}.jsonl"
        p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in r),
                     encoding="utf-8")
        print(f"wrote {p}: {len(r)} rows")


if __name__ == "__main__":
    main()