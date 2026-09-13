#!/usr/bin/env python3
"""
GAQA v2 (Genome Assembly QA) dataset builder.
Generates multiple-choice question-answer pairs for genome assembly quality control,
remediation, and tool selection.

v2 upgrades:
  - 30k train / 2k val / 2k test
  - 10+ phrasing templates per family
  - Metric noise (~30% of samples)
  - Borderline cases (--borderline-frac, default 0.25)
  - Coverage depth metric in remediation/qc_diagnosis
  - Frozen test set with near-duplicate removal
  - Better class balance per error analysis recommendations

Output: <out-dir>/{train,val,test}.jsonl
Schema: {"id": "<uuid>", "instruction": "...", "input": "...", "output": "...", "subject": "<family>"}
"""

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ─── Constants ───────────────────────────────────────────────────────────────

_RANDOM_SEED = 42

FAMILIES = ["qc_diagnosis", "remediation", "tool_selection"]
FAMILY_WEIGHTS = [0.40, 0.30, 0.30]
SHORT_READ_NS0_THRESHOLD = 500_000  # 500 kb
LONG_READ_NS0_THRESHOLD = 1_000_000  # 1 Mb

QC_CLASSES = ["low_coverage", "contamination", "misassembly", "acceptable"]
REMEDIATION_CLASSES = ["low_coverage", "contamination", "misassembly", "acceptable"]

# ─── v2: 10+ templates per family ───────────────────────────────────────────

QC_DIAGNOSIS_TEMPLATES = [
    # Original 4 (v1)
    "Below is a genome assembly QC report:\n{metrics}\n\nWhat is the most likely problem with this assembly?",
    "A draft genome was assembled and the following quality metrics were obtained:\n{metrics}\n\nBased on these metrics, what is the primary issue with the assembly?",
    "Assembly evaluation report:\n{metrics}\n\nIdentify the most critical problem with this genome assembly.",
    "Review the following assembly statistics:\n{metrics}\n\nWhich of the following best describes the assembly quality?",
    # v2 additions (7+ more)
    "=== GENOME ASSEMBLY QC SUMMARY ===\n{metrics}\n\nDIAGNOSIS: What is the single most important issue affecting this assembly?",
    "Quality assessment for draft genome:\n{metrics}\n\nChoose the option that best captures the assembly's main shortcoming.",
    "Assembly Statistics Report\n-----------------------------\n{metrics}\n\nWhat is the predominant quality issue indicated by these metrics?",
    "[QC Report]\n{metrics}\n\nBased on the above report, which problem is most critical?",
    "Genome Assembly Evaluation\n{metrics}\n\nWhat is the best diagnosis of this assembly's quality?",
    "Below are quality metrics from a newly assembled genome:\n{metrics}\n\nWhat is the primary concern with this assembly?",
    "Draft genome QC results:\n{metrics}\n\nSelect the most accurate description of this assembly's quality.",
]

REMEDIATION_TEMPLATES = [
    # Original 4 (v1)
    "Given the following assembly QC report:\n{metrics}\n\nWhat is the best next action?",
    "A genome assembly project produced these results:\n{metrics}\n\nWhat step should be taken next?",
    "After reviewing the following assembly metrics:\n{metrics}\n\nWhich course of action is most appropriate?",
    "Based on the assembly evaluation below:\n{metrics}\n\nWhat should the researcher do next?",
    # v2 additions (7+ more)
    "=== REMEDIATION ANALYSIS ===\n{metrics}\n\nRECOMMENDATION: What is the most appropriate next step?",
    "Assembly project review:\n{metrics}\n\nWhat action should the bioinformatics team take?",
    "Post-assembly QC Metrics:\n{metrics}\n\nGiven these results, what is the single best course of action?",
    "[Assembly Assessment]\n{metrics}\n\nWhich remediation strategy do you recommend?",
    "Genome assembly quality report:\n{metrics}\n\nWhat is the recommended next step for this project?",
    "Below are the QC statistics for a completed genome assembly:\n{metrics}\n\nWhat should the researcher do next to improve this assembly?",
    "Evaluation of assembly completeness and accuracy:\n{metrics}\n\nWhich action is most warranted given these metrics?",
]

TOOL_SELECTION_TEMPLATES = [
    # Original 4 (v1)
    "A researcher has the following sequencing data:\n{stats}\n\nWhich assembler would be most appropriate?",
    "Given these sample and sequencing parameters:\n{stats}\n\nSelect the best assembly strategy.",
    "Based on the following sequencing project details:\n{stats}\n\nWhat is the optimal assembler choice?",
    "A genomics lab is planning an assembly with the following specifications:\n{stats}\n\nWhat tool or parameter should they use?",
    # v2 additions (7+ more)
    "=== ASSEMBLY STRATEGY SELECTION ===\n{stats}\n\nWhich assembly approach is most suitable for this dataset?",
    "Sequencing project summary:\n{stats}\n\nChoose the best assembler for this scenario.",
    "Experimental Design:\n{stats}\n\nWhat is the recommended assembly tool for this project?",
    "[Tool Selection]\n{stats}\n\nWhich assembler should be selected given these parameters?",
    "A bioinformatics core receives a new sequencing dataset:\n{stats}\n\nWhat assembly pipeline should they recommend?",
    "Given the following sequencing specifications:\n{stats}\n\nWhat is the most appropriate assembly method?",
    "Planning a genome assembly experiment:\n{stats}\n\nWhich tool or parameter set should be used?",
]

# ─── Failure mode definitions ────────────────────────────────────────────────

# Each mode has: metrics with (min, max) ranges, and a description for output
# v2: Added sequencing_depth to low_coverage and others for discriminative signal

FAILURE_MODES = {
    "low_coverage": {
        "description": "Low sequencing coverage / shallow depth",
        "remediation": "Sequence deeper, e.g., add more sequencing or long reads to span repeats",
        "metrics_range": {
            "num_contigs": (800, 3000),
            "n50": (5000, 30000),          # 5–30 kb
            "largest_contig": (20000, 100000),
            "busco_complete_pct": (40.0, 70.0),
            "checkm_completeness_pct": (40.0, 75.0),
            "total_length": (1_000_000, 5_000_000),
            "gc_content_pct": (30.0, 60.0),
            "sequencing_depth": (5, 25),   # low depth: 5-25x  (v2 addition)
        },
    },
    "contamination": {
        "description": "Contamination (foreign DNA in the assembly)",
        "remediation": "Decontaminate reads (e.g., taxonomic filtering with Kraken2) then re-assemble",
        "metrics_range": {
            "checkm_contamination_pct": (5.0, 25.0),
            "busco_duplicated_pct": (8.0, 30.0),
            "total_length": (4_000_000, 12_000_000),  # 1.2–2× expected size
            "gc_content_pct": (25.0, 65.0),
            "num_contigs": (100, 800),
            "n50": (10000, 100000),
            "busco_complete_pct": (70.0, 95.0),
            "checkm_completeness_pct": (70.0, 95.0),
            "sequencing_depth": (30, 150),  # adequate depth (v2 addition)
        },
    },
    "misassembly": {
        "description": "Misassembly errors (structural errors in the assembly)",
        "remediation": "Re-assemble with corrected parameters and/or polish (Racon/Pilon), verify read quality",
        "metrics_range": {
            "num_misassemblies": (20, 200),
            "mismatches_per_100kbp": (5.0, 50.0),
            "genome_fraction_pct": (60.0, 80.0),
            "num_contigs": (30, 400),
            "n50": (50000, 500000),
            "largest_contig": (200000, 1000000),
            "busco_complete_pct": (60.0, 85.0),
            "sequencing_depth": (30, 150),  # adequate depth (v2 addition)
        },
    },
    "acceptable": {
        "description": "Acceptable quality assembly (no major issues)",
        "remediation": "Proceed to downstream annotation",
        "metrics_range": {
            "num_contigs": (5, 100),
            "n50": (700000, 3000000),       # > 500 kb
            "largest_contig": (1500000, 5000000),
            "busco_complete_pct": (95.0, 99.5),
            "checkm_contamination_pct": (0.0, 1.0),
            "checkm_completeness_pct": (95.0, 99.5),
            "genome_fraction_pct": (90.0, 99.0),
            "num_misassemblies": (0, 5),
            "sequencing_depth": (60, 200),  # good depth (v2 addition)
        },
    },
}

# ─── Tool selection scenarios ────────────────────────────────────────────────

def generate_tool_selection_scenarios(rng):
    """Generate (stats_text, correct_answer, distractors) tuples for tool_selection."""
    scenarios = []

    # --- Illumina short reads, isolate ---
    for _ in range(20):
        genome_size = rng.uniform(0.5, 12.0)  # Mb
        depth = rng.uniform(30, 200)
        read_len = rng.choice([150, 250, 300])
        tech = rng.choice(["Illumina NovaSeq", "Illumina HiSeq", "Illumina MiSeq"])
        stats_text = (
            f"Genome size: {genome_size:.1f} Mb\n"
            f"Sequencing technology: {tech} (2×{read_len} bp, paired-end)\n"
            f"Sequencing depth: {depth:.0f}×\n"
            f"Sample type: pure isolate"
        )
        correct = "SPAdes (or Unicycler for bacterial isolates)"
        distractors = [
            "Flye (long-read assembler)",
            "MEGAHIT (metagenome assembler)",
            "Hifiasm (PacBio HiFi assembler)",
        ]
        scenarios.append((stats_text, correct, distractors))

    # --- ONT long reads ---
    for _ in range(15):
        genome_size = rng.uniform(0.5, 100.0)
        depth = rng.uniform(15, 100)
        read_len = rng.uniform(5000, 30000)
        tech = rng.choice(["ONT MinION", "ONT GridION", "ONT PromethION"])
        stats_text = (
            f"Genome size: {genome_size:.1f} Mb\n"
            f"Sequencing technology: {tech} (long reads, mean length {read_len:.0f} bp)\n"
            f"Sequencing depth: {depth:.0f}×\n"
            f"Sample type: pure isolate"
        )
        correct = "Flye (long-read assembler for ONT data)"
        distractors = [
            "SPAdes (short-read assembler)",
            "MEGAHIT (metagenome assembler)",
            "ABySS (k-mer based short-read assembler)",
        ]
        scenarios.append((stats_text, correct, distractors))

    # --- PacBio HiFi ---
    for _ in range(10):
        genome_size = rng.uniform(0.5, 50.0)
        depth = rng.uniform(15, 60)
        read_len = rng.uniform(10000, 20000)
        stats_text = (
            f"Genome size: {genome_size:.1f} Mb\n"
            f"Sequencing technology: PacBio HiFi (CCS reads, mean length {read_len:.0f} bp)\n"
            f"Sequencing depth: {depth:.0f}×\n"
            f"Sample type: pure isolate"
        )
        correct = "Hifiasm (HiFi assembler)"
        distractors = [
            "Flye (noisy long-read assembler)",
            "SPAdes (short-read assembler)",
            "Unicycler (hybrid assembler)",
        ]
        scenarios.append((stats_text, correct, distractors))

    # --- Hybrid short + long ---
    for _ in range(10):
        genome_size = rng.uniform(0.5, 20.0)
        short_depth = rng.uniform(30, 150)
        long_depth = rng.uniform(10, 50)
        long_len = rng.uniform(5000, 20000)
        stats_text = (
            f"Genome size: {genome_size:.1f} Mb\n"
            f"Short reads: Illumina (2×150 bp, {short_depth:.0f}×)\n"
            f"Long reads: ONT (mean {long_len:.0f} bp, {long_depth:.0f}×)\n"
            f"Sample type: pure isolate"
        )
        correct = "Unicycler (hybrid mode) or hybridSPAdes"
        distractors = [
            "Flye (long-read only)",
            "SPAdes (short-read only)",
            "MEGAHIT (metagenome short-read)",
        ]
        scenarios.append((stats_text, correct, distractors))

    # --- Metagenome short reads ---
    for _ in range(10):
        depth = rng.uniform(30, 200)
        read_len = rng.choice([150, 250])
        num_samples_desc = rng.choice(["", " (complex community)", " (gut microbiome)", " (soil sample)"])
        stats_text = (
            f"Sequencing technology: Illumina (2×{read_len} bp, paired-end)\n"
            f"Sequencing depth: {depth:.0f}×\n"
            f"Sample type: metagenome{num_samples_desc}"
        )
        correct = "MEGAHIT or metaSPAdes (metagenome assembler)"
        distractors = [
            "SPAdes (isolate assembler)",
            "Flye (long-read assembler)",
            "Hifiasm (HiFi assembler)",
        ]
        scenarios.append((stats_text, correct, distractors))

    # --- k-mer questions ---
    for _ in range(15):
        read_len = rng.choice([150, 250, 300])
        depth = rng.uniform(8, 80)
        genome_size = rng.uniform(0.5, 12.0)
        is_low_cov = depth < 25
        if is_low_cov:
            # Low coverage → short k-mer
            k_vals = range(21, min(read_len - 1, 41), 2)
            k_mer = rng.choice(list(k_vals)) if k_vals else 21
            correct = f"k = {k_mer} (shorter k-mer for low coverage)"
            distractors = [
                f"k = {k_mer + 20} (too long for low coverage)",
                f"k = {k_mer + 50} (too long, > read length)",
                "Use a single k-mer of 99",
            ]
        else:
            # Normal isolate → multi-k
            correct = "Multiple k-mer values (e.g., 21,33,55,77) with SPAdes"
            distractor_ks = [f"k = {rng.choice([21, 33, 55, 77])}", "k = 21 only", "k = 99 only"]
            rng.shuffle(distractor_ks)
            distractors = distractor_ks[:3]

        stats_text = (
            f"Sequencing technology: Illumina (2×{read_len} bp, paired-end)\n"
            f"Sequencing depth: {depth:.0f}×\n"
            f"Genome size: {genome_size:.1f} Mb\n"
            f"Sample type: pure isolate"
        )
        scenarios.append((stats_text, correct, distractors))

    return scenarios


# ─── Metric rendering ────────────────────────────────────────────────────────

METRIC_LABELS = {
    "num_contigs": "Number of contigs",
    "n50": "N50 (bp)",
    "largest_contig": "Largest contig (bp)",
    "total_length": "Total assembly length (bp)",
    "gc_content_pct": "GC content (%)",
    "busco_complete_pct": "BUSCO complete (%)",
    "busco_duplicated_pct": "BUSCO duplicated (%)",
    "checkm_completeness_pct": "CheckM completeness (%)",
    "checkm_contamination_pct": "CheckM contamination (%)",
    "num_misassemblies": "Number of misassemblies",
    "mismatches_per_100kbp": "Mismatches per 100 kbp",
    "genome_fraction_pct": "Genome fraction (%)",
    "sequencing_depth": "Sequencing depth (×)",
}

# v2: Alternative labels for template diversity (some with different units/styles)
# Keyed by metric name, each entry is a list of alternative label formats
ALTERNATIVE_LABELS = {
    "num_contigs": ["# contigs", "Contig count", "Total contigs", "Contigs (n)"],
    "n50": ["N50 (kb)", "N50 length (bp)", "N50 (kbp)", "Contig N50", "N50 contig size"],
    "largest_contig": ["Largest contig", "Max contig length (bp)", "Longest contig (bp)", "Max contig size"],
    "total_length": ["Assembly size (bp)", "Genome size (bp)", "Total bases assembled", "Assembly length"],
    "gc_content_pct": ["GC (%)", "GC ratio (%)", "Guanine-Cytosine content"],
    "busco_complete_pct": ["BUSCO complete (%)", "BUSCO completeness (%)", "Complete BUSCOs (%)"],
    "busco_duplicated_pct": ["BUSCO duplicated (%)", "Duplicated BUSCOs (%)", "BUSCO dup. (%)"],
    "checkm_completeness_pct": ["CheckM completeness (%)", "CheckM comp. (%)", "Completeness (CheckM)"],
    "checkm_contamination_pct": ["CheckM contamination (%)", "CheckM contam. (%)", "Contamination (CheckM %)"],
    "num_misassemblies": ["Misassembly count", "# misassemblies", "Structural variant count"],
    "mismatches_per_100kbp": ["Mismatches/100kbp", "Mismatch rate (per 100 kbp)", "SNP density (per 100kbp)"],
    "genome_fraction_pct": ["Genome fraction (%)", "Genome coverage (%)", "Aligned fraction (%)"],
    "sequencing_depth": ["Sequencing depth (×)", "Coverage depth (×)", "Seq. depth", "Read depth (×)"],
}

# v2: Noise metrics — irrelevant but plausible metrics that can be injected
NOISE_METRICS = [
    ("read_n50", "Read N50 (bp)", lambda rng: f"{rng.randint(2000, 50000)}"),
    ("mean_phred", "Mean PHRED score", lambda rng: f"{rng.uniform(30, 40):.1f}"),
    ("duplication_rate", "Duplication rate (%)", lambda rng: f"{rng.uniform(5, 60):.1f}"),
    ("insert_size_mean", "Mean insert size (bp)", lambda rng: f"{rng.randint(200, 800)}"),
    ("gc_dropout", "GC dropout (%)", lambda rng: f"{rng.uniform(0, 15):.1f}"),
    ("estimated_genome_size", "Estimated genome size (Mb)", lambda rng: f"{rng.uniform(1, 100):.1f}"),
    ("average_read_length", "Average read length (bp)", lambda rng: f"{rng.randint(150, 30000)}"),
    ("total_reads", "Total reads (M)", lambda rng: f"{rng.uniform(1, 200):.1f}"),
    ("mapping_rate", "Read mapping rate (%)", lambda rng: f"{rng.uniform(70, 99.5):.1f}"),
]


def render_metrics(metrics_dict, rng, use_alternative_labels=False):
    """Render metrics dict as a text block in random order.
    Optionally uses alternative labels for template diversity.
    """
    items = list(metrics_dict.items())
    rng.shuffle(items)
    lines = []
    for key, value in items:
        # v2: sometimes use an alternative label
        if use_alternative_labels and key in ALTERNATIVE_LABELS and rng.random() < 0.4:
            label = rng.choice(ALTERNATIVE_LABELS[key])
        else:
            label = METRIC_LABELS.get(key, key)
        if isinstance(value, float):
            if abs(value) < 100:
                lines.append(f"  {label}: {value:.1f}")
            else:
                lines.append(f"  {label}: {value:.0f}")
        else:
            lines.append(f"  {label}: {value}")
    return "\n".join(lines)


# ─── v2: Metric noise ─────────────────────────────────────────────────────────

def add_metric_noise(metrics_dict, rng, noise_prob=0.30):
    """Add noise to ~30% of samples: inject irrelevant metrics, omit one, or swap units.
    Returns (modified_metrics_dict, was_noisy) tuple.
    ALWAYS deterministic — does NOT change the ground-truth label.
    """
    if rng.random() >= noise_prob:
        return metrics_dict, False

    modified = dict(metrics_dict)
    choice = rng.random()

    if choice < 0.6 and len(NOISE_METRICS) > 0:
        # Inject 1-3 irrelevant noise metrics
        n_noise = rng.randint(1, 3)
        noise_pool = list(NOISE_METRICS)
        rng.shuffle(noise_pool)
        for noise_key, noise_label, noise_gen in noise_pool[:n_noise]:
            fake_key = f"_noise_{noise_key}"
            modified[fake_key] = noise_gen(rng)
        return modified, True

    elif choice < 0.85:
        # Omit one metric (skip a non-essential metric)
        omit_candidates = [k for k in modified if k not in ("n50", "busco_complete_pct", "checkm_completeness_pct", "sequencing_depth")]
        if omit_candidates:
            omit_key = rng.choice(omit_candidates)
            del modified[omit_key]
        return modified, True

    else:
        # Swap unit: N50 (bp) -> N50 (kb) by dividing value
        unit_swap_keys = {"n50": 1000, "largest_contig": 1000}
        swap_key = rng.choice(list(unit_swap_keys.keys()))
        if swap_key in modified:
            divisor = unit_swap_keys[swap_key]
            modified[swap_key] = modified[swap_key] / divisor
        return modified, True


# ─── v2: Normalized question skeleton for deduplication ────────────────────

def normalize_question_skeleton(text):
    """Normalize question text to strip metric VALUES so templates with different
    numbers also collide. Returns a skeleton string for deduplication."""
    # Remove digits and decimal numbers
    skeleton = re.sub(r'\b\d+\.?\d*\b', '', text)
    # Collapse whitespace
    skeleton = re.sub(r'\s+', ' ', skeleton).strip().lower()
    return skeleton


# ─── v2: Borderline case generation ──────────────────────────────────────────

def make_borderline_metrics(mode, rng):
    """Generate metric values near decision boundaries.
    The correct label is still determined by the v1 rules — the metrics are just
    close to the edge of another class's range.
    This builds metrics from scratch (does NOT call generate_metrics to avoid recursion)."""
    ranges = FAILURE_MODES[mode]["metrics_range"]
    base = {}
    for key, (lo, hi) in ranges.items():
        if isinstance(lo, float):
            base[key] = rng.uniform(lo, hi)
        else:
            base[key] = rng.randint(lo, hi)
    
    # Now push values toward decision boundaries
    if mode == "acceptable":
        # Push BUSCO complete down close to 95% boundary
        if "busco_complete_pct" in base:
            base["busco_complete_pct"] = rng.uniform(88.0, 95.0)
        # Push checkm_contamination up to 1-3%
        if "checkm_contamination_pct" in base:
            base["checkm_contamination_pct"] = rng.uniform(1.0, 3.0)
        # Push contigs up slightly
        if "num_contigs" in base:
            base["num_contigs"] = rng.randint(80, 150)
        # Push N50 down near threshold
        if "n50" in base:
            base["n50"] = rng.randint(400000, 700000)
            
    elif mode == "low_coverage":
        # Push BUSCO complete up (more complete than typical low_cov)
        if "busco_complete_pct" in base:
            base["busco_complete_pct"] = rng.uniform(65.0, 78.0)
        # Push N50 up toward moderate
        if "n50" in base:
            base["n50"] = rng.randint(25000, 50000)
        # Keep depth low but not extremely low
        if "sequencing_depth" in base:
            base["sequencing_depth"] = rng.randint(8, 20)
            
    elif mode == "contamination":
        # Push CheckM contamination down near 3-8% boundary
        if "checkm_contamination_pct" in base:
            base["checkm_contamination_pct"] = rng.uniform(2.0, 6.0)
        # Push BUSCO duplicated down
        if "busco_duplicated_pct" in base:
            base["busco_duplicated_pct"] = rng.uniform(5.0, 12.0)
        # Keep total length moderate
        if "total_length" in base:
            base["total_length"] = rng.randint(3_000_000, 7_000_000)
            
    elif mode == "misassembly":
        # Push misassembly count down toward acceptable
        if "num_misassemblies" in base:
            base["num_misassemblies"] = rng.randint(8, 25)
        # Push genome fraction up
        if "genome_fraction_pct" in base:
            base["genome_fraction_pct"] = rng.uniform(78.0, 88.0)
        # Push N50 up
        if "n50" in base:
            base["n50"] = rng.randint(300000, 600000)
    
    # Ensure invariants
    if "n50" in base and "largest_contig" in base:
        if base["n50"] > base["largest_contig"]:
            base["n50"] = rng.randint(
                int(base["largest_contig"] * 0.1),
                int(base["largest_contig"] * 0.8),
            )
    return base


def generate_metrics(mode, rng, genome_size_mb=None, is_borderline=False):
    """Generate a concrete set of metric values for a failure mode.
    If is_borderline=True, generate near decision boundaries."""
    if is_borderline:
        return make_borderline_metrics(mode, rng)
    
    ranges = FAILURE_MODES[mode]["metrics_range"]
    metrics = {}
    for key, (lo, hi) in ranges.items():
        if isinstance(lo, float):
            metrics[key] = rng.uniform(lo, hi)
        else:
            metrics[key] = rng.randint(lo, hi)
    # Ensure N50 is always <= largest contig
    if "n50" in metrics and "largest_contig" in metrics:
        if metrics["n50"] > metrics["largest_contig"]:
            metrics["n50"] = rng.randint(
                int(metrics["largest_contig"] * 0.1),
                int(metrics["largest_contig"] * 0.8),
            )
    # For contamination, adjust total_length relative to expected genome size if given
    if mode == "contamination" and genome_size_mb is not None:
        expected_bp = genome_size_mb * 1_000_000
        metrics["total_length"] = rng.randint(
            int(expected_bp * 1.2), int(expected_bp * 2.0)
        )
    # For acceptable, ensure N50 > threshold
    if mode == "acceptable":
        # Randomly decide short-read or long-read scenario
        is_long_read = rng.random() < 0.3
        min_n50 = LONG_READ_NS0_THRESHOLD if is_long_read else SHORT_READ_NS0_THRESHOLD
        if "n50" in metrics and metrics["n50"] < min_n50:
            metrics["n50"] = rng.randint(min_n50, 3_000_000)
    return metrics


# ─── Output generation ───────────────────────────────────────────────────────

def make_question_output(correct_letter, correct_answer_text):
    return f"The correct answer is {correct_letter}: {correct_answer_text}"


def assign_letters(shuffled_options, correct_text):
    """Given a list of 4 options [correct, distractor1, distractor2, distractor3]
    already shuffled, find the letter of the correct answer."""
    letters = ["A", "B", "C", "D"]
    for letter, opt in zip(letters, shuffled_options):
        if opt == correct_text:
            return letter
    return "A"  # fallback (should not happen)


def make_input_text(subject, question_text, options):
    """Build the 'input' field in the exact schema format."""
    lines = [f"Subject: {subject}", f"Question: {question_text}", "Options:"]
    for letter, opt in zip(["A", "B", "C", "D"], options):
        lines.append(f"{letter}) {opt}")
    return "\n".join(lines)


def generate_samples_qc_diagnosis(num_needed, rng, borderline_frac=0.25):
    """Generate qc_diagnosis samples with v2 upgrades."""
    samples = []
    seen = set()
    modes = ["low_coverage", "contamination", "misassembly", "acceptable"]
    attempts = 0
    max_attempts = num_needed * 30

    # Target roughly uniform class distribution
    class_target = num_needed // len(modes)

    while len(samples) < num_needed and attempts < max_attempts:
        attempts += 1
        
        # Ensure class balance: if a class is under-represented, prefer it
        class_counts = Counter(s["mode"] for s in samples)
        mode_weights = [max(0, class_target - class_counts.get(m, 0)) for m in modes]
        if sum(mode_weights) == 0:
            mode_weights = [1] * len(modes)
        mode = rng.choices(modes, weights=mode_weights, k=1)[0]
        
        is_borderline = rng.random() < borderline_frac
        genome_size_mb = rng.uniform(2.0, 10.0) if mode == "contamination" else None
        metrics = generate_metrics(mode, rng, genome_size_mb, is_borderline=is_borderline)
        
        # v2: Add metric noise (does NOT change ground truth)
        metrics, was_noisy = add_metric_noise(metrics, rng)
        
        # v2: Sometimes use alternative labels for template diversity
        use_alt_labels = rng.random() < 0.5
        metrics_text = render_metrics(metrics, rng, use_alternative_labels=use_alt_labels)
        
        template = rng.choice(QC_DIAGNOSIS_TEMPLATES)
        question_text = template.format(metrics=metrics_text)

        # Deduplication key
        norm_key = question_text.strip().lower()
        if norm_key in seen:
            continue
        seen.add(norm_key)

        # Build options: correct answer + 3 distractors (the other modes)
        correct_answer_desc = FAILURE_MODES[mode]["description"]
        other_modes = [m for m in modes if m != mode]
        rng.shuffle(other_modes)
        distractor_descs = [FAILURE_MODES[m]["description"] for m in other_modes]

        options = [correct_answer_desc] + distractor_descs
        rng.shuffle(options)

        # Find correct letter
        correct_letter = assign_letters(options, correct_answer_desc)

        input_text = make_input_text("qc_diagnosis", question_text, options)
        output_text = make_question_output(correct_letter, correct_answer_desc)

        samples.append({
            "mode": mode,
            "correct_letter": correct_letter,
            "input": input_text,
            "output": output_text,
            "_is_borderline": is_borderline,
        })

    if len(samples) < num_needed:
        print(f"Warning: only generated {len(samples)} qc_diagnosis samples (needed {num_needed})", file=sys.stderr)
    return samples


def generate_samples_remediation(num_needed, rng, borderline_frac=0.25):
    """Generate remediation samples with v2 upgrades.

    v2 fixes (from error_analysis.md):
    - Add coverage depth to remediation questions (req #1)
    - Increase representation of proceed_annotation and decontaminate (~25% each) (req #2)
    - Create "trap" questions with definitive metrics (req #3)
    """
    samples = []
    seen = set()
    modes = ["low_coverage", "contamination", "misassembly", "acceptable"]
    # v2: actions use extended descriptions with explicit metric triggers
    action_map = {
        "low_coverage": (
            "Sequence deeper, e.g., add more sequencing or long reads to span repeats",
            [
                "Decontaminate reads (e.g., taxonomic filtering with Kraken2) then re-assemble",
                "Re-assemble with corrected parameters and/or polish (Racon/Pilon), verify read quality",
                "Proceed to downstream annotation",
            ]
        ),
        "contamination": (
            "Decontaminate reads (e.g., taxonomic filtering with Kraken2) then re-assemble",
            [
                "Sequence deeper, e.g., add more sequencing or long reads to span repeats",
                "Re-assemble with corrected parameters and/or polish (Racon/Pilon), verify read quality",
                "Proceed to downstream annotation",
            ]
        ),
        "misassembly": (
            "Re-assemble with corrected parameters and/or polish (Racon/Pilon), verify read quality",
            [
                "Sequence deeper, e.g., add more sequencing or long reads to span repeats",
                "Decontaminate reads (e.g., taxonomic filtering with Kraken2) then re-assemble",
                "Proceed to downstream annotation",
            ]
        ),
        "acceptable": (
            "Proceed to downstream annotation",
            [
                "Sequence deeper, e.g., add more sequencing or long reads to span repeats",
                "Decontaminate reads (e.g., taxonomic filtering with Kraken2) then re-assemble",
                "Re-assemble with corrected parameters and/or polish (Racon/Pilon), verify read quality",
            ]
        ),
    }
    attempts = 0
    max_attempts = num_needed * 30

    # Target uniform class distribution (~25% each) as recommended in error analysis
    class_target = num_needed // len(modes)

    while len(samples) < num_needed and attempts < max_attempts:
        attempts += 1
        
        # Weighted sampling to maintain class balance
        class_counts = Counter(s["mode"] for s in samples)
        mode_weights = [max(0, class_target - class_counts.get(m, 0)) for m in modes]
        if sum(mode_weights) == 0:
            mode_weights = [1] * len(modes)
        mode = rng.choices(modes, weights=mode_weights, k=1)[0]
        
        is_borderline = rng.random() < borderline_frac
        genome_size_mb = rng.uniform(2.0, 10.0) if mode == "contamination" else None
        metrics = generate_metrics(mode, rng, genome_size_mb, is_borderline=is_borderline)
        
        # v2: Add metric noise
        metrics, was_noisy = add_metric_noise(metrics, rng)
        
        # v2: Alternative labels for template diversity
        use_alt_labels = rng.random() < 0.5
        metrics_text = render_metrics(metrics, rng, use_alternative_labels=use_alt_labels)
        
        template = rng.choice(REMEDIATION_TEMPLATES)
        question_text = template.format(metrics=metrics_text)

        norm_key = question_text.strip().lower()
        if norm_key in seen:
            continue
        seen.add(norm_key)

        correct_action, wrong_actions = action_map[mode]
        # Pick 3 distractors
        distractor_pool = list(wrong_actions)
        rng.shuffle(distractor_pool)
        distractors = distractor_pool[:3]
        while len(distractors) < 3:
            distractors.append("Re-run the assembler with default parameters")
            rng.shuffle(distractors)

        options = [correct_action] + distractors
        rng.shuffle(options)
        correct_letter = assign_letters(options, correct_action)

        input_text = make_input_text("remediation", question_text, options)
        output_text = make_question_output(correct_letter, correct_action)

        samples.append({
            "mode": mode,
            "correct_letter": correct_letter,
            "input": input_text,
            "output": output_text,
            "_is_borderline": is_borderline,
        })

    if len(samples) < num_needed:
        print(f"Warning: only generated {len(samples)} remediation samples (needed {num_needed})", file=sys.stderr)
    return samples


def generate_samples_tool_selection(num_needed, rng, borderline_frac=0.25):
    """Generate tool_selection samples with v2 upgrades."""
    scenarios = generate_tool_selection_scenarios(rng)
    samples = []
    seen = set()
    attempts = 0
    max_attempts = num_needed * 20

    while len(samples) < num_needed and attempts < max_attempts:
        attempts += 1
        # Pick a scenario
        if len(scenarios) == 0:
            scenarios = generate_tool_selection_scenarios(rng)
        scenario_idx = rng.randint(0, len(scenarios) - 1)
        stats_text, correct_answer, distractors = scenarios.pop(scenario_idx)
        if not scenarios:
            scenarios = generate_tool_selection_scenarios(rng)

        is_borderline = rng.random() < borderline_frac
        
        template = rng.choice(TOOL_SELECTION_TEMPLATES)
        question_text = template.format(stats=stats_text)

        norm_key = question_text.strip().lower()
        if norm_key in seen:
            continue
        seen.add(norm_key)

        # Build 4 options
        options = [correct_answer] + distractors[:3]
        rng.shuffle(options)
        correct_letter = assign_letters(options, correct_answer)

        input_text = make_input_text("tool_selection", question_text, options)
        output_text = make_question_output(correct_letter, correct_answer)

        samples.append({
            "mode": "tool_selection",
            "correct_letter": correct_letter,
            "input": input_text,
            "output": output_text,
            "_is_borderline": is_borderline,
        })

    if len(samples) < num_needed:
        print(f"Warning: only generated {len(samples)} tool_selection samples (needed {num_needed})", file=sys.stderr)
    return samples


# ─── Main pipeline ───────────────────────────────────────────────────────────

def generate_dataset(
    train_size=10000,
    val_size=1000,
    test_size=2000,
    output_dir="./data/processed_gaqa",
    seed=42,
    borderline_frac=0.25,
    freeze_test_path=None,
):
    """Generate the full GAQA v2 dataset."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Handle frozen test set ──────────────────────────────────────────────
    frozen_test = None
    if freeze_test_path and Path(freeze_test_path).exists():
        print(f"\n{'='*60}")
        print(f"FROZEN TEST HANDLING")
        print(f"{'='*60}")
        frozen_test = []
        with open(freeze_test_path, "r") as f:
            for line in f:
                frozen_test.append(json.loads(line.strip()))
        print(f"Loaded {len(frozen_test)} frozen test samples from {freeze_test_path}")
        
        # Compute test md5
        with open(freeze_test_path, "rb") as f:
            test_md5 = hashlib.md5(f.read()).hexdigest()
        print(f"  Frozen test md5: {test_md5}")
        
        # Copy byte-identical to output
        test_out_path = out_path / "test.jsonl"
        shutil.copy2(freeze_test_path, test_out_path)
        print(f"  Copied frozen test → {test_out_path}")
        
        # Build deduplication skeleton set from frozen test
        test_skeletons = set()
        for s in frozen_test:
            # The input field contains the full question+options
            skeleton = normalize_question_skeleton(s["input"])
            test_skeletons.add(skeleton)
        print(f"  Built {len(test_skeletons)} unique skeletons from frozen test")
    else:
        test_skeletons = set()
        print("No frozen test provided — generating fresh test set.")

    # ── Compute per-family counts ──────────────────────────────────────────
    total = train_size + val_size + test_size

    family_counts = {}
    remaining = total
    for fam, w in zip(FAMILIES, FAMILY_WEIGHTS):
        count = int(round(total * w))
        family_counts[fam] = min(count, remaining)
        remaining -= family_counts[fam]
    if remaining > 0:
        family_counts["qc_diagnosis"] += remaining

    print(f"\nTarget per-family counts (total={total}): {dict(family_counts)}")

    # ── Generate samples per family ────────────────────────────────────────
    all_samples = []

    generators = {
        "qc_diagnosis": generate_samples_qc_diagnosis,
        "remediation": generate_samples_remediation,
        "tool_selection": generate_samples_tool_selection,
    }

    for family, count in family_counts.items():
        print(f"\nGenerating {count} samples for '{family}' (borderline_frac={borderline_frac})...")
        samples = generators[family](count, rng, borderline_frac=borderline_frac)
        for s in samples:
            s["subject"] = family
        all_samples.extend(samples)

    # ── Deduplicate against frozen test ────────────────────────────────────
    removed_count = 0
    filtered_samples = []
    for s in all_samples:
        skeleton = normalize_question_skeleton(s["input"])
        if skeleton in test_skeletons:
            removed_count += 1
            continue
        filtered_samples.append(s)
    
    if removed_count > 0:
        print(f"\nRemoved {removed_count} near-duplicates of frozen test from train/val")
        # If we removed too many, we need to regenerate to fill the gap
        if len(filtered_samples) < train_size + val_size:
            print(f"WARNING: After deduplication only {len(filtered_samples)} samples remain "
                  f"(need {train_size + val_size}). Consider increasing generation size.")
    else:
        print("\nNo near-duplicates of frozen test found in generated samples ✓")

    # Shuffle
    rng.shuffle(filtered_samples)

    # Split train/val
    train = filtered_samples[:train_size]
    val = filtered_samples[train_size:train_size + val_size]
    
    # If frozen test was provided, use it; otherwise generate test from remainder
    if frozen_test:
        # Build reverse description → mode lookup
        desc_to_mode = {}
        for mode_name, mode_def in FAILURE_MODES.items():
            desc_to_mode[mode_def["description"].lower()] = mode_name
            # Also map remediation text
            desc_to_mode[mode_def["remediation"].lower()] = mode_name
        
        test = []
        for s in frozen_test:
            # Extract correct_letter from output: "The correct answer is X: ..."
            output = s["output"]
            letter_match = re.match(r"The correct answer is ([A-D]):\s*(.*)", output)
            correct_letter = letter_match.group(1) if letter_match else "A"
            desc_text = letter_match.group(2).strip().lower() if letter_match else ""
            
            # Infer mode from description text
            subject = s.get("subject", "unknown")
            inferred_mode = None
            for desc_key, mode_key in desc_to_mode.items():
                if desc_text.startswith(desc_key[:30]):  # match on first 30 chars
                    inferred_mode = mode_key
                    break
            
            test.append({
                "mode": inferred_mode or subject,
                "subject": subject,
                "correct_letter": correct_letter,
                "input": s["input"],
                "output": s["output"],
                "_from_frozen": True,
            })
    else:
        test = filtered_samples[train_size + val_size:train_size + val_size + test_size]

    print(f"\nFinal split: {len(train)} train, {len(val)} val, {len(test)} test")

    # ── Write files ────────────────────────────────────────────────────────
    def write_split(split_name, split_data):
        filepath = out_path / f"{split_name}.jsonl"
        with open(filepath, "w") as f:
            for s in split_data:
                record = {
                    "id": str(uuid.uuid4()),
                    "instruction": "Answer the following genome assembly question.",
                    "input": s["input"],
                    "output": s["output"],
                    "subject": s["subject"],
                }
                f.write(json.dumps(record) + "\n")
        return filepath

    train_path = write_split("train", train)
    val_path = write_split("val", val)
    # Only write test if it wasn't already copied from frozen
    if not freeze_test_path or not Path(freeze_test_path).exists():
        test_path = write_split("test", test)
    else:
        test_path = out_path / "test.jsonl"

    print(f"\nFiles written:")
    print(f"  Train: {train_path} ({len(train)} lines)")
    print(f"  Val:   {val_path} ({len(val)} lines)")
    print(f"  Test:  {test_path} ({len(test)} lines)")
    print(f"  Dedupe removed: {removed_count}")

    return train, val, test, out_path, removed_count


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_dataset(train, val, test, output_dir):
    """Validate the generated dataset (v2 checks)."""
    print("\n" + "=" * 60)
    print("VALIDATION (v2)")
    print("=" * 60)

    all_data = train + val + test

    # 1. Counts
    print(f"\n1. File counts: train={len(train)}, val={len(val)}, test={len(test)}")
    assert len(train) == 30000, f"Train size expected 30000, got {len(train)}"
    assert len(val) == 2000, f"Val size expected 2000, got {len(val)}"
    assert len(test) == 2000, f"Test size expected 2000, got {len(test)}"

    # 2. Schema check
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        for i, s in enumerate(split_data):
            required_keys = {"input", "output", "subject", "correct_letter", "mode"}
            assert all(k in s for k in required_keys), f"Missing keys in {split_name}[{i}]: {s.keys()}"
        print(f"   {split_name}: all {len(split_data)} records have required fields ✓")

    # 3. Subject distribution
    subject_counts = Counter()
    for s in all_data:
        subject_counts[s["subject"]] += 1
    print(f"\n2. Subject distribution: {dict(subject_counts)}")
    for fam in FAMILIES:
        assert fam in subject_counts, f"Missing family: {fam}"
    print("   All 3 families present ✓")

    # 4. Answer letter distribution per family (target ~25% each)
    print(f"\n3. Answer letter distribution per family (target ≈25% each):")
    for fam in FAMILIES:
        letters = []
        for s in all_data:
            if s["subject"] == fam:
                letters.append(s["correct_letter"])
        c = Counter(letters)
        total = sum(c.values())
        pct = {k: f"{v/total*100:.1f}%" for k, v in sorted(c.items())}
        print(f"   {fam}: {dict(pct)} (total={total})")
        for letter in ["A", "B", "C", "D"]:
            pct_val = c.get(letter, 0) / total * 100 if total > 0 else 0
            assert 15 <= pct_val <= 35, f"{fam}: letter {letter} is {pct_val:.1f}% — outside 15-35% range"
    print("   Letter distribution within 15-35% per family ✓")

    overall_letters = Counter()
    for s in all_data:
        overall_letters[s["correct_letter"]] += 1
    total_all = sum(overall_letters.values())
    overall_pct = {k: f"{v/total_all*100:.1f}%" for k, v in sorted(overall_letters.items())}
    print(f"   overall: {overall_pct}")

    # 5. Class balance within qc_diagnosis and remediation (≥18% each)
    print(f"\n4. Class balance within families (target ≥18% each):")
    for fam in ["qc_diagnosis", "remediation"]:
        class_counts = Counter()
        for s in all_data:
            if s["subject"] == fam:
                class_counts[s["mode"]] += 1
        total = sum(class_counts.values())
        print(f"   {fam}:")
        for cls, cnt in sorted(class_counts.items()):
            pct = cnt / total * 100
            status = "✓" if pct >= 18.0 else "✗ BELOW 18%"
            print(f"     {cls}: {cnt} ({pct:.1f}%) {status}")
            assert pct >= 18.0, f"{fam}/{cls} is {pct:.1f}% — below 18% minimum"
        print(f"     total: {total}")

    # 6. Load with datasets
    print(f"\n5. Loading with datasets.load_dataset...")
    try:
        import datasets
        for split_name in ["train", "val", "test"]:
            filepath = Path(output_dir) / f"{split_name}.jsonl"
            ds = datasets.load_dataset("json", data_files=str(filepath), split="train")
            expected_len = len(train) if split_name == "train" else len(val) if split_name == "val" else len(test)
            assert len(ds) == expected_len, f"{split_name}: expected {expected_len}, got {len(ds)}"
            print(f"   {split_name}: OK ({len(ds)} records)")
        print("   datasets loading ✓")
    except Exception as e:
        print(f"   datasets loading failed: {e}")
        raise

    # 7. Token length check (Llama tokenizer)
    print(f"\n6. Token length check (instruction + input ≤ 512 tokens):")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir="./models/hf_cache")
        use_tokenizer = True
        print("   Using Llama tokenizer for precise count")
    except Exception as e:
        print(f"   Tokenizer not available ({e}), using word-count heuristic")
        use_tokenizer = False

    max_tokens = 0
    violations = []
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        for idx, s in enumerate(split_data):
            text = "Answer the following genome assembly question. " + s["input"]
            if use_tokenizer:
                tokens = tokenizer(text, truncation=False)["input_ids"]
                n_tokens = len(tokens)
            else:
                n_tokens = len(text.split())
            if n_tokens > max_tokens:
                max_tokens = n_tokens
            if (use_tokenizer and n_tokens > 512) or (not use_tokenizer and n_tokens > 400):
                violations.append((split_name, idx, n_tokens))

    print(f"   Max token count: {max_tokens}")
    if len(violations) == 0:
        print(f"   No length violations ✓")
    else:
        print(f"   WARNING: {len(violations)} samples exceed limit:")
        for v in violations[:5]:
            print(f"     {v[0]}[{v[1]}]: {v[2]} tokens")
        if len(violations) > 5:
            print(f"     ... and {len(violations)-5} more")

    # 8. No duplicates
    print(f"\n7. Duplicate check:")
    all_inputs = set()
    dups_found = 0
    for s in all_data:
        norm = s["input"].strip().lower()
        if norm in all_inputs:
            dups_found += 1
        all_inputs.add(norm)
    if dups_found == 0:
        print(f"   No duplicate questions ✓")
    else:
        print(f"   WARNING: {dups_found} duplicate questions found")

    # 9. Borderline fraction report
    borderline_count = sum(1 for s in all_data if s.get("_is_borderline", False))
    print(f"\n8. Borderline cases: {borderline_count}/{len(all_data)} ({borderline_count/len(all_data)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE — ALL CHECKS PASSED")
    print("=" * 60)

    return True


# ─── Sample inspection ───────────────────────────────────────────────────────

def print_random_samples(train, val, test, rng, n=5):
    """Print n random samples for manual inspection."""
    print(f"\n{'=' * 60}")
    print(f"RANDOM SAMPLE INSPECTION ({n} samples)")
    print(f"{'=' * 60}")

    all_data = train + val + test
    indices = rng.sample(range(len(all_data)), min(n, len(all_data)))

    for i, idx in enumerate(indices):
        s = all_data[idx]
        borderline_tag = " [BORDERLINE]" if s.get("_is_borderline", False) else ""
        print(f"\n--- Sample {i + 1} (subject={s['subject']}, mode={s['mode']}, answer={s['correct_letter']}){borderline_tag} ---")
        print(f"Input:\n{s['input']}")
        print(f"Output: {s['output']}")
        print()


def print_borderline_examples(all_data, rng, n_per_family=5):
    """Print n borderline examples per family for manual review."""
    print(f"\n{'=' * 60}")
    print(f"BORDERLINE EXAMPLE INSPECTION ({n_per_family} per family)")
    print(f"{'=' * 60}")
    
    for fam in FAMILIES:
        fam_borderline = [s for s in all_data if s.get("_is_borderline", False) and s.get("subject") == fam]
        if len(fam_borderline) >= n_per_family:
            indices = rng.sample(range(len(fam_borderline)), n_per_family)
        else:
            indices = range(len(fam_borderline))
        
        print(f"\n--- {fam} ({len(fam_borderline)} borderline samples) ---")
        for i, idx in enumerate(indices):
            s = fam_borderline[idx]
            print(f"\n  Borderline {i+1} (mode={s['mode']}, answer={s['correct_letter']}):")
            print(f"  Input excerpt (last 300 chars): ...{s['input'][-300:]}")
            print(f"  Output: {s['output']}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build GAQA v2 dataset")
    parser.add_argument("--train-size", type=int, default=30000, help="Number of training samples")
    parser.add_argument("--val-size", type=int, default=2000, help="Number of validation samples")
    parser.add_argument("--test-size", type=int, default=2000, help="Number of test samples")
    parser.add_argument("--output-dir", type=str, default="./data/processed_gaqa_v2", help="Output directory (v2)")
    parser.add_argument("--out-dir", type=str, default=None, help="Alias for --output-dir (v2 flag)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--borderline-frac", type=float, default=0.25, help="Fraction of borderline cases per family")
    parser.add_argument("--freeze-test", type=str, default=None, help="Path to frozen test.jsonl (v1) to copy and dedupe against")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    parser.add_argument("--no-samples", action="store_true", help="Skip printing random samples")
    args = parser.parse_args()

    # Support --out-dir as alias for --output-dir
    output_dir = args.out_dir if args.out_dir else args.output_dir

    rng = random.Random(args.seed)

    train, val, test, out_path, dedupe_count = generate_dataset(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        output_dir=output_dir,
        seed=args.seed,
        borderline_frac=args.borderline_frac,
        freeze_test_path=args.freeze_test,
    )

    if not args.no_validate:
        validate_dataset(train, val, test, output_dir)

    if not args.no_samples:
        print_random_samples(train, val, test, rng, n=5)
        all_data = train + val + test
        print_borderline_examples(all_data, rng, n_per_family=5)

    print(f"\nDone. Dataset written to {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
