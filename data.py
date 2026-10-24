import csv
import json
import os
import random
import re
from pathlib import Path

import constants


def organism_info(name):
    key = " ".join(name.split()[:2]).lower()
    for b in constants.BACTERIA:
        if " ".join(b["name"].split()[:2]).lower() == key:
            return b["size_mb"], b["split"]
    return 0.0, "train"


def parse_quast(path):
    metrics = {}
    mapping = {
        "# contigs": "num_contigs",
        "N50": "n50",
        "Largest contig": "largest_contig",
        "Total length": "total_length",
        "GC (%)": "gc_content_pct",
        "# misassemblies": "num_misassemblies",
        "Genome fraction (%)": "genome_fraction_pct",
        "Mismatches per 100 kbp": "mismatches_per_100kbp",
    }
    if not os.path.exists(path):
        return metrics
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2 or parts[0] not in mapping:
                continue
            key, val = mapping[parts[0]], parts[1].strip()
            try:
                metrics[key] = float(val) if "." in val else int(val)
            except ValueError:
                metrics[key] = val
    return metrics


def parse_busco(path):
    metrics = {}
    if not os.path.exists(path):
        return metrics
    m = re.search(
        r"C:\s*([\d.]+)%\s*\[\s*S:\s*([\d.]+)%\s*,\s*D:\s*([\d.]+)%\s*\]"
        r"\s*,\s*F:\s*([\d.]+)%\s*,\s*M:\s*([\d.]+)%", open(path).read())
    if m:
        metrics["busco_complete_pct"] = float(m.group(1))
        metrics["busco_single_copy_pct"] = float(m.group(2))
        metrics["busco_duplicated_pct"] = float(m.group(3))
        metrics["busco_fragmented_pct"] = float(m.group(4))
        metrics["busco_missing_pct"] = float(m.group(5))
    return metrics


def parse_checkm(path):
    metrics = {}
    if not os.path.exists(path):
        return metrics
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 13 or parts[0] == "Bin Id":
                continue
            try:
                metrics["checkm_completeness_pct"] = float(parts[11])
                metrics["checkm_contamination_pct"] = float(parts[12])
            except (ValueError, IndexError):
                pass
    return metrics


def gather_metrics(condition_id, qc_dir):
    d = os.path.join(qc_dir, condition_id)
    if not os.path.isdir(d):
        return {}
    metrics = {}
    metrics.update(parse_quast(os.path.join(d, "quast", "report.tsv")))
    busco_dir = os.path.join(d, "busco")
    if os.path.isdir(busco_dir):
        for root, _, files in os.walk(busco_dir):
            for fn in files:
                if "short_summary" in fn:
                    metrics.update(parse_busco(os.path.join(root, fn)))
                    break
    metrics.update(parse_checkm(os.path.join(d, "checkm", "qa.txt")))
    return metrics


def rule_label(m, coverage):
    contam = m.get("checkm_contamination_pct")
    dup = m.get("busco_duplicated_pct")
    comp = m.get("busco_complete_pct")
    mis = m.get("num_misassemblies")
    frac = m.get("genome_fraction_pct")
    if (contam is not None and contam >= 5.0) or (dup is not None and dup >= 8.0):
        return "contamination"
    if coverage is not None and coverage <= 20 and comp is not None and comp < 80.0:
        return "low_coverage"
    if (mis is not None and mis >= 20) or (frac is not None and frac < 80.0 and coverage is not None and coverage > 20):
        return "misassembly"
    if comp is not None and comp >= 90.0 and (frac is None or frac >= 85.0) and (contam is None or contam < 5.0):
        return "acceptable"
    return None


def render_metrics(m, rng, alternative=False):
    items = list(m.items())
    rng.shuffle(items)
    lines = []
    for key, value in items:
        if alternative and key in constants.ALTERNATIVE_LABELS and rng.random() < 0.4:
            label = rng.choice(constants.ALTERNATIVE_LABELS[key])
        else:
            label = constants.METRIC_LABELS.get(key, key)
        if isinstance(value, float):
            lines.append(f"  {label}: {value:.1f}" if abs(value) < 100 else f"  {label}: {value:.0f}")
        else:
            lines.append(f"  {label}: {value}")
    return "\n".join(lines)


def jitter(m, rng, mag=0.10):
    out = dict(m)
    for k, v in list(out.items()):
        if k not in constants.CONTINUOUS_KEYS or not isinstance(v, (int, float)):
            continue
        nv = v * (1.0 + rng.uniform(-mag, mag))
        nv = max(0, round(nv)) if isinstance(v, int) else max(0.0, round(nv, 2))
        if k.endswith("_pct"):
            nv = min(nv, 100.0)
        out[k] = nv
    if "n50" in out and "largest_contig" in out and out["n50"] > out["largest_contig"]:
        out["n50"] = max(1, int(out["largest_contig"] * rng.uniform(0.1, 0.8)))
    return out


def add_noise(m, rng, prob=0.25):
    if rng.random() >= prob:
        return m
    out = dict(m)
    r = rng.random()
    if r < 0.6:
        for label, gen in rng.sample(constants.NOISE_METRICS, rng.randint(1, 3)):
            out[label] = gen(rng)
    elif r < 0.85:
        drop = [k for k in out if k not in ("n50", "busco_complete_pct", "checkm_completeness_pct")]
        if drop:
            del out[rng.choice(drop)]
    return out


def make_input_text(subject, question, options):
    lines = [f"Subject: {subject}", f"Question: {question}", "Options:"]
    for letter, opt in zip(["A", "B", "C", "D"], options):
        lines.append(f"{letter}) {opt}")
    return "\n".join(lines)


def make_question_output(letter, text):
    return f"The correct answer is {letter}: {text}"


def assign_letters(options, correct):
    for letter, opt in zip(["A", "B", "C", "D"], options):
        if opt == correct:
            return letter
    return "A"


def render_qc(metrics, label, rng, uid):
    question = rng.choice(constants.QC_DIAGNOSIS_TEMPLATES).replace("{metrics}", render_metrics(metrics, rng, rng.random() < 0.5))
    correct = constants.FAILURE_MODES[label]["description"]
    options = [correct] + [constants.FAILURE_MODES[m]["description"] for m in constants.MODES if m != label]
    rng.shuffle(options)
    letter = assign_letters(options, correct)
    return {"id": f"qc_{uid}", "subject": "qc_diagnosis",
            "input": make_input_text("qc_diagnosis", question, options),
            "output": make_question_output(letter, correct)}


def render_rem(metrics, label, rng, uid):
    question = rng.choice(constants.REMEDIATION_TEMPLATES).replace("{metrics}", render_metrics(metrics, rng, rng.random() < 0.5))
    correct = constants.FAILURE_MODES[label]["remediation"]
    options = [correct] + [constants.FAILURE_MODES[m]["remediation"] for m in constants.MODES if m != label]
    rng.shuffle(options)
    letter = assign_letters(options, correct)
    return {"id": f"rem_{uid}", "subject": "remediation",
            "input": make_input_text("remediation", question, options),
            "output": make_question_output(letter, correct)}


def render_tool(organism, size_mb, coverage, contamination_pct, assembler_config, rng, uid):
    sample = "pure isolate" if contamination_pct == 0 else f"isolate with ~{contamination_pct}% foreign DNA"
    stats = (f"Organism: {organism} (genome ~{size_mb:.1f} Mb)\n"
             f"Sequencing technology: Illumina (paired-end)\n"
             f"Sequencing depth: {coverage}x\n"
             f"Sample type: {sample}")
    if assembler_config == "megahit":
        correct = "SPAdes (isolate assembler)"
        distractors = ["MEGAHIT (metagenome assembler)", "Flye (long-read assembler)", "Hifiasm (HiFi assembler)"]
    elif assembler_config == "wrong":
        correct = "SPAdes with default k-mer range (e.g., 21,33,55,77)"
        distractors = ["SPAdes with a single short k-mer (k=21)", "MEGAHIT (metagenome assembler)", "Flye (long-read assembler)"]
    elif coverage <= 20:
        correct = "SPAdes with smaller k-mers (e.g., 21,33,55) for low coverage"
        distractors = ["SPAdes with large k-mers only (e.g., 77,99,127)", "MEGAHIT (metagenome assembler)", "Flye (long-read assembler)"]
    else:
        correct = "SPAdes with default k-mer range (e.g., 21,33,55,77)"
        distractors = ["MEGAHIT (metagenome assembler)", "Flye (long-read assembler)", "Hifiasm (HiFi assembler)"]
    question = rng.choice(constants.TOOL_SELECTION_TEMPLATES).replace("{stats}", stats)
    options = [correct] + distractors
    rng.shuffle(options)
    letter = assign_letters(options, correct)
    return {"id": f"tool_{uid}", "subject": "tool_selection",
            "input": make_input_text("tool_selection", question, options),
            "output": make_question_output(letter, correct)}


def build(data_dir, out_dir, jitter_copies=60, jitter_mag=0.10, seed=42):
    rng = random.Random(seed)
    data_dir = Path(data_dir)
    conditions = list(csv.DictReader((data_dir / "conditions.csv").open()))

    corpus = []
    for cond in conditions:
        metrics = gather_metrics(cond["condition_id"], str(data_dir / "qc_results"))
        if not metrics:
            continue
        metrics = dict(metrics)
        metrics["sequencing_depth"] = int(cond["coverage"])
        label = rule_label(metrics, int(cond["coverage"])) or cond["label_qc"]
        size, split = organism_info(cond["organism"])
        corpus.append({
            "condition_id": cond["condition_id"],
            "organism": cond["organism"],
            "size_mb": size,
            "split": split,
            "coverage": int(cond["coverage"]),
            "contamination_pct": int(cond["contamination_pct"]),
            "assembler_config": cond["assembler_config"],
            "label": label,
            "metrics": metrics,
        })

    heldout = []
    for c in corpus:
        if c["split"] != "heldout":
            continue
        heldout.append(render_qc(c["metrics"], c["label"], rng, c["condition_id"]))
        heldout.append(render_rem(c["metrics"], c["label"], rng, c["condition_id"]))
        heldout.append(render_tool(c["organism"], c["size_mb"], c["coverage"], c["contamination_pct"],
                                   c["assembler_config"], rng, c["condition_id"]))

    train_conds = [c for c in corpus if c["split"] == "train"]
    rng.shuffle(train_conds)
    val_ids = {c["condition_id"] for c in train_conds[: max(5, len(train_conds) // 10)]}

    train_qs, val_qs, uid = [], [], 0
    for c in train_conds:
        is_val = c["condition_id"] in val_ids
        for j in range(1 if is_val else jitter_copies):
            vec = c["metrics"] if j == 0 else add_noise(jitter(c["metrics"], rng, jitter_mag), rng)
            uid += 1
            (val_qs if is_val else train_qs).extend([
                render_qc(vec, c["label"], rng, uid),
                render_rem(vec, c["label"], rng, uid),
            ])
        uid += 1
        (val_qs if is_val else train_qs).append(
            render_tool(c["organism"], c["size_mb"], c["coverage"], c["contamination_pct"], c["assembler_config"], rng, uid))

    rng.shuffle(train_qs)
    rng.shuffle(val_qs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train.jsonl", train_qs), ("val.jsonl", val_qs), ("test.jsonl", heldout)]:
        with (out_dir / name).open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"train": len(train_qs), "val": len(val_qs), "heldout": len(heldout)}


def gold_letter(output):
    m = re.search(r"The correct answer is\s+([A-D])", output)
    if not m:
        raise ValueError(f"no gold letter in {output[:80]!r}")
    return m.group(1)


def acceptable_letter(row):
    m = re.search(r"([A-D])\s*\)\s*Acceptable quality", row["input"])
    return m.group(1) if m else None


def load(split, data_dir="data"):
    path = Path(data_dir) / constants.SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
