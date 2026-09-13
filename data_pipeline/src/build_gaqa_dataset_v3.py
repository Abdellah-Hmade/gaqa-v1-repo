"""Build the GAQA v3 dataset — truth-anchored.

Every qc_diagnosis / remediation training example is derived from a REAL
assembly's QC output (QUAST + BUSCO + CheckM) in data/realval_v3, optionally
lightly jittered (5-15% multiplicative noise, invariants preserved).
Labels are HYBRID:
  1. calibrated metric rules (fitted doctrine thresholds), else
  2. construction label from conditions.csv (hard cases, metrics ambiguous).
Rule-vs-construction conflicts are exported for expert review.

qc_diagnosis and remediation questions are PAIRED from the same metric vector.
tool_selection reuses the v2 scenario generator (transfers well) plus
corpus-grounded scenarios from the real conditions.
domain_knowledge (from data/gaqa_knowledge) is merged if present.

Outputs:
  data/processed_gaqa_v3/{train,val}.jsonl
  data/realval_v3_heldout/test.jsonl      (heldout organisms, unjittered)
  data/realval_v3/expert_review_qc.csv    (label conflicts for user review)
  data/realval_v3/v3_build_report.json    (agreement stats, counts)

Usage:
  python -m src.build_gaqa_dataset_v3 [--jitter 60] [--seed 42]
"""

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

from src.build_gaqa_dataset import (
    QC_DIAGNOSIS_TEMPLATES, REMEDIATION_TEMPLATES, TOOL_SELECTION_TEMPLATES,
    FAILURE_MODES, make_question_output, make_input_text, assign_letters,
    generate_tool_selection_scenarios,
)
from src.build_realval_dataset import (
    gather_qc_metrics, render_metrics, add_metric_noise,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "realval_v3"
OUT = ROOT / "data" / "processed_gaqa_v3"
HELDOUT = ROOT / "data" / "realval_v3_heldout"
KNOWLEDGE = ROOT / "data" / "gaqa_knowledge"

MODES = ["low_coverage", "contamination", "misassembly", "acceptable"]


# ─── Calibrated label rules ─────────────────────────────────────────────────

def rule_label(m: dict, coverage: float):
    """Deterministic metric->label rules (QC doctrine thresholds).

    Returns a mode string, or None when the metrics are ambiguous
    (hard case -> construction label is used instead).
    """
    checkm_contam = m.get("checkm_contamination_pct")
    busco_dup = m.get("busco_duplicated_pct")
    busco_comp = m.get("busco_complete_pct")
    misassemblies = m.get("num_misassemblies")
    genome_frac = m.get("genome_fraction_pct")

    # 1. contamination: elevated CheckM contamination or BUSCO duplication
    if (checkm_contam is not None and checkm_contam >= 5.0) or \
       (busco_dup is not None and busco_dup >= 8.0):
        return "contamination"
    # 2. low coverage: shallow depth + incomplete BUSCO
    if coverage is not None and coverage <= 20 and \
       (busco_comp is not None and busco_comp < 80.0):
        return "low_coverage"
    # 3. misassembly: many misassemblies or poor genome fraction at depth
    if (misassemblies is not None and misassemblies >= 20) or \
       (genome_frac is not None and genome_frac < 80.0 and
            coverage is not None and coverage > 20):
        return "misassembly"
    # 4. acceptable: complete BUSCO + good genome fraction, clean CheckM
    if busco_comp is not None and busco_comp >= 90.0 and \
       (genome_frac is None or genome_frac >= 85.0) and \
       (checkm_contam is None or checkm_contam < 5.0):
        return "acceptable"
    return None


# ─── Jitter ──────────────────────────────────────────────────────────────────

CONTINUOUS_KEYS = {
    "n50", "largest_contig", "total_length", "num_contigs", "gc_content_pct",
    "busco_complete_pct", "busco_duplicated_pct", "busco_fragmented_pct",
    "busco_missing_pct", "checkm_completeness_pct", "checkm_contamination_pct",
    "genome_fraction_pct", "num_misassemblies", "mismatches_per_100kbp",
    "sequencing_depth",
}


def jitter_vector(m: dict, rng, mag=0.10):
    """Light multiplicative jitter preserving realism + invariants."""
    out = dict(m)
    for k, v in list(out.items()):
        if k not in CONTINUOUS_KEYS or not isinstance(v, (int, float)):
            continue
        f = 1.0 + rng.uniform(-mag, mag)
        nv = v * f
        if isinstance(v, int):
            nv = max(0, round(nv))
        else:
            nv = max(0.0, round(nv, 2))
        if k.endswith("_pct"):
            nv = min(nv, 100.0)
        out[k] = nv
    # invariants
    if "n50" in out and "largest_contig" in out and out["n50"] > out["largest_contig"]:
        out["n50"] = max(1, int(out["largest_contig"] * rng.uniform(0.1, 0.8)))
    if "largest_contig" in out and "total_length" in out and \
       out["largest_contig"] > out["total_length"]:
        out["total_length"] = int(out["largest_contig"] * rng.uniform(1.0, 1.3))
    return out


# ─── Question rendering ──────────────────────────────────────────────────────

def render_qc_question(metrics, label, rng, uid):
    """qc_diagnosis MCQ from a metric vector."""
    template = rng.choice(QC_DIAGNOSIS_TEMPLATES)
    metrics_text = render_metrics(metrics, rng, use_alternative_labels=rng.random() < 0.5)
    correct = FAILURE_MODES[label]["description"]
    distractors = [FAILURE_MODES[m]["description"] for m in MODES if m != label]
    options = [correct] + distractors
    rng.shuffle(options)
    letter = assign_letters(options, correct)
    question = template.replace("{metrics}", metrics_text)
    return {
        "id": f"qc_v3_{uid}", "subject": "qc_diagnosis",
        "input": make_input_text("qc_diagnosis", question, options),
        "output": make_question_output(letter, correct),
    }


def render_remediation_question(metrics, label, rng, uid):
    """remediation MCQ from the SAME metric vector as its qc pair."""
    template = rng.choice(REMEDIATION_TEMPLATES)
    metrics_text = render_metrics(metrics, rng, use_alternative_labels=rng.random() < 0.5)
    correct = FAILURE_MODES[label]["remediation"]
    distractors = [FAILURE_MODES[m]["remediation"] for m in MODES if m != label]
    options = [correct] + distractors
    rng.shuffle(options)
    letter = assign_letters(options, correct)
    question = template.replace("{metrics}", metrics_text)
    return {
        "id": f"rem_v3_{uid}", "subject": "remediation",
        "input": make_input_text("remediation", question, options),
        "output": make_question_output(letter, correct),
    }


def render_corpus_tool_questions(cond, manifest_row, rng, uid):
    """tool_selection questions grounded in a real condition."""
    qs = []
    size = float(manifest_row["genome_size_mb"])
    cov = cond["coverage"]
    contam = int(cond["contamination_pct"])
    sample = "pure isolate" if contam == 0 else f"isolate with ~{contam}% foreign DNA"
    stats = (f"Organism: {cond['organism']} (genome ~{size:.1f} Mb)\n"
             f"Sequencing technology: Illumina (paired-end)\n"
             f"Sequencing depth: {cov}x\n"
             f"Sample type: {sample}")
    template = rng.choice(TOOL_SELECTION_TEMPLATES)
    cfg = cond["assembler_config"]

    if cfg == "megahit":
        correct = "SPAdes (isolate assembler)"
        distractors = ["MEGAHIT (metagenome assembler)",
                       "Flye (long-read assembler)",
                       "Hifiasm (HiFi assembler)"]
    elif cfg == "wrong":
        correct = "SPAdes with default k-mer range (e.g., 21,33,55,77)"
        distractors = ["SPAdes with a single short k-mer (k=21)",
                       "MEGAHIT (metagenome assembler)",
                       "Flye (long-read assembler)"]
    elif cov <= 20:
        correct = "SPAdes with smaller k-mers (e.g., 21,33,55) for low coverage"
        distractors = ["SPAdes with large k-mers only (e.g., 77,99,127)",
                       "MEGAHIT (metagenome assembler)",
                       "Flye (long-read assembler)"]
    else:
        correct = "SPAdes with default k-mer range (e.g., 21,33,55,77)"
        distractors = ["MEGAHIT (metagenome assembler)",
                       "Flye (long-read assembler)",
                       "Hifiasm (HiFi assembler)"]
    options = [correct] + distractors
    rng.shuffle(options)
    letter = assign_letters(options, correct)
    question = template.replace("{stats}", stats)
    qs.append({
        "id": f"tool_v3_{uid}", "subject": "tool_selection",
        "input": make_input_text("tool_selection", question, options),
        "output": make_question_output(letter, correct),
    })
    return qs


def render_v2_tool_scenario(stats_text, correct, distractors, rng, uid):
    template = rng.choice(TOOL_SELECTION_TEMPLATES)
    options = [correct] + list(distractors)
    rng.shuffle(options)
    letter = assign_letters(options, correct)
    question = template.replace("{stats}", stats_text)
    return {
        "id": f"tool_v3s_{uid}", "subject": "tool_selection",
        "input": make_input_text("tool_selection", question, options),
        "output": make_question_output(letter, correct),
    }


# ─── Corpus loading ──────────────────────────────────────────────────────────

def load_corpus(data_dir: Path):
    """Yield dicts with condition info + real QC metrics."""
    conds = list(csv.DictReader((data_dir / "conditions.csv").open()))
    manifest = {r["organism"]: r
                for r in csv.DictReader((data_dir / "manifest.csv").open())}
    corpus = []
    for cond in conds:
        cid = cond["condition_id"]
        metrics = gather_qc_metrics(cid, str(data_dir / "qc_results"))
        if not metrics:
            continue
        metrics = dict(metrics)
        metrics["sequencing_depth"] = int(cond["coverage"])
        corpus.append({
            "condition_id": cid,
            "organism": cond["organism"],
            "coverage": int(cond["coverage"]),
            "contamination_pct": int(cond["contamination_pct"]),
            "assembler_config": cond["assembler_config"],
            "label_construction": cond["label_qc"],
            "split": manifest.get(cond["organism"], {}).get("split", "train"),
            "manifest_row": manifest.get(cond["organism"], {}),
            "metrics": metrics,
        })
    return corpus


# ─── Main build ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA),
                    help="v3 corpus dir (conditions.csv + qc_results/)")
    ap.add_argument("--jitter", type=int, default=60,
                    help="jittered copies per train condition vector")
    ap.add_argument("--jitter-mag", type=float, default=0.10)
    ap.add_argument("--v2-tool-scenarios", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-knowledge", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    data_dir = Path(args.data_dir)
    corpus = load_corpus(data_dir)
    if not corpus:
        print("FATAL: no QC metrics found — run the pipeline qc stage first.")
        return 1
    print(f"Corpus: {len(corpus)} conditions with QC metrics "
          f"({sum(1 for c in corpus if c['split']=='train')} train, "
          f"{sum(1 for c in corpus if c['split']=='heldout')} heldout)")

    # ── hybrid labeling + agreement stats ──
    stats = Counter()
    review_rows = []
    for c in corpus:
        rl = rule_label(c["metrics"], c["coverage"])
        cl = c["label_construction"]
        if rl is None:
            c["label"] = cl
            c["hard_case"] = True
            stats["ambiguous->construction"] += 1
        elif rl == cl:
            c["label"] = rl
            c["hard_case"] = False
            stats["agree"] += 1
        else:
            c["label"] = rl
            c["hard_case"] = True
            stats["conflict->rule"] += 1
            review_rows.append({
                "condition_id": c["condition_id"], "organism": c["organism"],
                "coverage": c["coverage"],
                "contamination_pct": c["contamination_pct"],
                "assembler_config": c["assembler_config"],
                "rule_label": rl, "construction_label": cl,
                "busco_complete": c["metrics"].get("busco_complete_pct"),
                "busco_dup": c["metrics"].get("busco_duplicated_pct"),
                "checkm_contam": c["metrics"].get("checkm_contamination_pct"),
                "misassemblies": c["metrics"].get("num_misassemblies"),
                "genome_fraction": c["metrics"].get("genome_fraction_pct"),
            })
    labeled = [c for c in corpus if c["label"] in MODES]
    n_conf = stats["conflict->rule"]
    agree_rate = stats["agree"] / max(1, len(labeled))
    print(f"Labeling: agree={stats['agree']} ({agree_rate:.0%}), "
          f"ambiguous={stats['ambiguous->construction']}, conflicts={n_conf}")

    with (data_dir / "expert_review_qc.csv").open("w", newline="") as f:
        if review_rows:
            w = csv.DictWriter(f, fieldnames=list(review_rows[0].keys()))
            w.writeheader()
            w.writerows(review_rows)

    # ── heldout test set: real vectors, unjittered ──
    HELDOUT.mkdir(parents=True, exist_ok=True)
    heldout_qs = []
    for c in labeled:
        if c["split"] != "heldout":
            continue
        uid = c["condition_id"]
        heldout_qs.append(render_qc_question(c["metrics"], c["label"], rng, uid + "_h"))
        heldout_qs.append(render_remediation_question(c["metrics"], c["label"], rng, uid + "_h"))
        heldout_qs += render_corpus_tool_questions(c, c["manifest_row"], rng, uid + "_h")
    with (HELDOUT / "test.jsonl").open("w") as f:
        for q in heldout_qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Heldout test: {len(heldout_qs)} questions -> {HELDOUT}/test.jsonl")

    # ── train/val: jittered vectors from train organisms ──
    train_conds = [c for c in labeled if c["split"] == "train"]
    rng.shuffle(train_conds)
    n_val_conds = max(5, len(train_conds) // 10)
    val_conds = set(c["condition_id"] for c in train_conds[:n_val_conds])

    train_qs, val_qs = [], []
    uid = 0
    for c in train_conds:
        is_val = c["condition_id"] in val_conds
        n_copies = 1 if is_val else args.jitter
        for j in range(n_copies):
            vec = c["metrics"] if j == 0 else jitter_vector(
                c["metrics"], rng, args.jitter_mag)
            # metric noise on a fraction of copies (irrelevant metrics/omissions)
            if rng.random() < 0.25:
                vec = add_metric_noise(dict(vec), rng, noise_prob=1.0)
            uid += 1
            qc = render_qc_question(vec, c["label"], rng, uid)
            rem = render_remediation_question(vec, c["label"], rng, uid)
            qc["_condition_id"] = rem["_condition_id"] = c["condition_id"]
            qc["_organism"] = rem["_organism"] = c["organism"]
            qc["_jitter"] = rem["_jitter"] = j
            (val_qs if is_val else train_qs).extend([qc, rem])

    # ── tool selection: corpus-grounded + v2 scenarios ──
    for c in train_conds:
        uid += 1
        for q in render_corpus_tool_questions(c, c["manifest_row"], rng, uid):
            q["_condition_id"] = c["condition_id"]
            q["_organism"] = c["organism"]
            (val_qs if c["condition_id"] in val_conds else train_qs).append(q)
    scen_rng = random.Random(args.seed + 1)
    for i, (stats_text, correct, distractors) in enumerate(
            generate_tool_selection_scenarios(scen_rng)[:args.v2_tool_scenarios]):
        uid += 1
        train_qs.append(render_v2_tool_scenario(
            stats_text, correct, distractors, rng, uid))

    # ── knowledge merge (optional) ──
    n_know = 0
    if args.include_knowledge and (KNOWLEDGE / "train.jsonl").exists():
        for line in (KNOWLEDGE / "train.jsonl").open():
            train_qs.append(json.loads(line))
            n_know += 1
        if (KNOWLEDGE / "val.jsonl").exists():
            for line in (KNOWLEDGE / "val.jsonl").open():
                val_qs.append(json.loads(line))

    rng.shuffle(train_qs)
    rng.shuffle(val_qs)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "train.jsonl").open("w") as f:
        for q in train_qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with (OUT / "val.jsonl").open("w") as f:
        for q in val_qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    report = {
        "corpus_conditions": len(corpus),
        "label_agreement": agree_rate,
        "label_stats": dict(stats),
        "train": len(train_qs),
        "val": len(val_qs),
        "heldout_test": len(heldout_qs),
        "knowledge_merged": n_know,
        "train_subjects": dict(Counter(q["subject"] for q in train_qs)),
        "val_subjects": dict(Counter(q["subject"] for q in val_qs)),
        "heldout_subjects": dict(Counter(q["subject"] for q in heldout_qs)),
        "jitter": args.jitter,
        "seed": args.seed,
    }
    (data_dir / "v3_build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
