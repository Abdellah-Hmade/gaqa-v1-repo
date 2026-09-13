#!/usr/bin/env python3
"""Generate v3 realval conditions: subsample + contamination spike + write conditions.csv.

Runs ON THE SERVER after the pipeline's download stage. Idempotent: existing
subsampled files are reused, existing conditions.csv is not overwritten unless
--force.

Condition matrix per organism (11 conditions):
  C01  10x clean                    -> low_coverage   (spades default)
  C02  30x clean                    -> acceptable     (spades default)
  C03 100x clean                    -> acceptable     (spades default)
  C04  30x + 5% related donor       -> contamination
  C05 100x + 10% related donor      -> contamination
  C06 100x + 10% distant donor      -> contamination
  C07  30x + 10% distant donor      -> contamination
  C08 100x + 5% related donor       -> contamination (mild, high cov)
  C09  30x + 10% related donor      -> contamination
  C10 100x clean, SPAdes -k 21      -> misassembly    (assembler_config=wrong)
  C11 100x clean, MEGAHIT           -> misassembly    (assembler_config=megahit)

= 132 assemblies over 12 organisms (9 train split = 99, 3 heldout = 33).

Usage (on server):
  python3 scripts/realval_v3_conditions.py \
      --data-dir data/realval_v3 \
      --runner "micromamba run -n realval" \
      [--dry-run] [--force]
"""

import argparse
import csv
import gzip
import os
import re
import subprocess
import sys

SEED = 42

# Default data dir resolves relative to this script (data_pipeline/), not CWD,
# so the standalone command works from any directory.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data", "realval_v3")

CONDITIONS = [
    # suffix, cov, contam_pct, donor_kind, label_qc, assembler_config
    ("C01", 10,  0,  None,      "low_coverage",  "default"),
    ("C02", 30,  0,  None,      "acceptable",    "default"),
    ("C03", 100, 0,  None,      "acceptable",    "default"),
    ("C04", 30,  5,  "related", "contamination", "default"),
    ("C05", 100, 10, "related", "contamination", "default"),
    ("C06", 100, 10, "distant", "contamination", "default"),
    ("C07", 30,  10, "distant", "contamination", "default"),
    ("C08", 100, 5,  "related", "contamination", "default"),
    ("C09", 30,  10, "related", "contamination", "default"),
    ("C10", 100, 0,  None,      "misassembly",   "wrong"),
    ("C11", 100, 0,  None,      "misassembly",   "megahit"),
]


def slugify(organism: str) -> str:
    """Match the pipeline's slug rule: tr ' ' '_' | tr -d '().'"""
    return re.sub(r"[().]", "", organism.replace(" ", "_"))


def run(cmd, dry_run=False):
    if dry_run:
        print(f"  [dry] {cmd}")
        return
    subprocess.run(cmd, shell=True, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def total_bases(fastq_gz, runner):
    """Total bases via seqkit stats -T (fallback: count via gzip)."""
    cmd = f"{runner} seqkit stats -T {fastq_gz}".split() if runner else \
        ["seqkit", "stats", "-T", fastq_gz]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode == 0:
        lines = out.stdout.strip().split("\n")
        if len(lines) >= 2:
            hdr, vals = lines[0].split("\t"), lines[1].split("\t")
            row = dict(zip(hdr, vals))
            for k in ("sum_len", "Total_length", "total_length"):
                if k in row:
                    return int(float(row[k]))
            # seqkit >=2: column 'sum_len'
            return int(float(vals[4].replace(",", "")))
    raise RuntimeError(f"seqkit stats failed on {fastq_gz}: {out.stderr[:200]}")


def seqtk_sample(runner, inp, frac, seed, out_gz, dry_run):
    if os.path.exists(out_gz) and os.path.getsize(out_gz) > 0 and not dry_run:
        return
    frac = min(frac, 1.0)
    exe = f"{runner} seqtk" if runner else "seqtk"
    run(f"{exe} sample -s{seed} {inp} {frac:.6f} | gzip > {out_gz}", dry_run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--runner", default="",
                    help='env runner prefix, e.g. "micromamba run -n realval"')
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    manifest_path = os.path.join(data_dir, "manifest.csv")
    conditions_path = os.path.join(data_dir, "conditions.csv")
    sub_dir = os.path.join(data_dir, "subsampled")
    os.makedirs(sub_dir, exist_ok=True)

    if os.path.exists(conditions_path) and not args.force:
        print(f"conditions.csv already exists at {conditions_path} "
              f"(use --force to regenerate)")
        return

    with open(manifest_path) as f:
        orgs = list(csv.DictReader(f))
    by_key = {}
    for o in orgs:
        # derive a short key from the manifest row order / name
        by_key[o["organism"]] = o

    # map manifest 'related_donor'/'distant_donor' short keys to organism rows.
    # The manifest stores donor keys like 'styphi'; build key->organism lookup
    # from the organism list order in src.build_v3_manifest.
    key_map = {
        "ecoli": "Escherichia coli K-12 MG1655",
        "saureus": "Staphylococcus aureus NCTC 8325",
        "vcholerae": "Vibrio cholerae N16961",
        "bsub": "Bacillus subtilis 168",
        "mtb": "Mycobacterium tuberculosis H37Rv",
        "paeru": "Pseudomonas aeruginosa PAO1",
        "spneumo": "Streptococcus pneumoniae R6",
        "lmono": "Listeria monocytogenes EGD-e",
        "styphi": "Salmonella Typhimurium LT2",
        "kpneumo": "Klebsiella pneumoniae MGH 78578",
        "nmeningo": "Neisseria meningitidis MC58",
        "efaec": "Enterococcus faecalis V583",
    }

    header = ["organism", "condition_id", "coverage", "contamination_pct",
              "reads_pe1", "reads_pe2", "reference", "label_qc",
              "label_remediation", "label_tool", "assembler_config"]
    rows_out = []

    for org_idx, org in enumerate(orgs, start=1):
        name = org["organism"]
        slug = slugify(name)
        size_mb = float(org["genome_size_mb"])
        genome_bp = size_mb * 1_000_000
        pe1 = os.path.join(data_dir, "reads", slug, f"{slug}_1.fastq.gz")
        pe2 = os.path.join(data_dir, "reads", slug, f"{slug}_2.fastq.gz")
        ref = os.path.join(data_dir, "references", f"{slug}_genomic.fna.gz")
        if not (os.path.exists(pe1) and os.path.exists(pe2)):
            print(f"[skip] {name}: reads not downloaded yet ({pe1})")
            continue
        if not os.path.exists(ref):
            print(f"[skip] {name}: reference missing ({ref})")
            continue

        print(f"[{slug}] estimating depth ...")
        if args.dry_run:
            host_bases = int(genome_bp * 100)
        else:
            host_bases = total_bases(pe1, args.runner)
        print(f"  host bases: {host_bases/1e6:.0f} Mb "
              f"(~{host_bases/genome_bp:.0f}x)")

        donors = {"related": org.get("related_donor"), "distant": org.get("distant_donor")}
        donor_bases = {}
        for kind, dkey in donors.items():
            if not dkey or dkey == "none":
                donor_bases[kind] = None
                continue
            dname = key_map[dkey]
            dslug = slugify(dname)
            dpe1 = os.path.join(data_dir, "reads", dslug, f"{dslug}_1.fastq.gz")
            if not os.path.exists(dpe1):
                print(f"  [warn] donor {dname} reads missing — {kind} contamination "
                      f"conditions will be skipped")
                donor_bases[kind] = None
                continue
            donor_bases[kind] = (dpe1, dpe1.replace("_1.fastq.gz", "_2.fastq.gz"),
                                 dslug, None if args.dry_run else total_bases(dpe1, args.runner))

        for (suffix, cov, contam, donor_kind, label_qc, asm_cfg) in CONDITIONS:
            cond_id = f"{org_idx:02d}{suffix}"
            target_bases = cov * genome_bp
            host_frac_reads = 1.0 - (contam / 100.0)
            host_target = target_bases * host_frac_reads
            hfrac = host_target / host_bases

            label_tool = "correct_config"
            if asm_cfg == "wrong":
                label_tool = "wrong_config"
            elif asm_cfg == "megahit":
                label_tool = "wrong_tool"

            if contam == 0:
                # reuse the same subsampled file for identical (coverage, clean)
                # conditions (e.g. C03/C10/C11 all use the C03 file)
                base_cond = f"{org_idx:02d}" + {10: "C01", 30: "C02", 100: "C03"}[cov]
                out1 = os.path.join(sub_dir, f"{base_cond}_{slug}_{cov}x_1.fastq.gz")
                out2 = out1.replace("_1.fastq.gz", "_2.fastq.gz")
                print(f"  {cond_id}: {cov}x clean ({asm_cfg}) -> {os.path.basename(out1)}")
                seqtk_sample(args.runner, pe1, hfrac, SEED + org_idx, out1, args.dry_run)
                seqtk_sample(args.runner, pe2, hfrac, SEED + org_idx, out2, args.dry_run)
            else:
                d = donor_bases.get(donor_kind)
                if d is None:
                    print(f"  {cond_id}: SKIPPED (no {donor_kind} donor)")
                    continue
                dpe1, dpe2, dslug, dbases = d
                if args.dry_run:
                    dbases = host_bases
                # host fraction of target, donor fraction of target
                donor_target = target_bases * (contam / 100.0)
                dfrac = donor_target / dbases
                tmp_h1 = os.path.join(sub_dir, f"_tmp_{cond_id}_h1.fastq.gz")
                tmp_h2 = os.path.join(sub_dir, f"_tmp_{cond_id}_h2.fastq.gz")
                tmp_d1 = os.path.join(sub_dir, f"_tmp_{cond_id}_d1.fastq.gz")
                tmp_d2 = os.path.join(sub_dir, f"_tmp_{cond_id}_d2.fastq.gz")
                out1 = os.path.join(
                    sub_dir, f"{cond_id}_{slug}_{cov}x_spike{contam}pct_{dslug}_1.fastq.gz")
                out2 = out1.replace("_1.fastq.gz", "_2.fastq.gz")
                print(f"  {cond_id}: {cov}x + {contam}% {donor_kind} ({dslug})")
                if args.dry_run or not (os.path.exists(out1) and os.path.getsize(out1) > 0):
                    seqtk_sample(args.runner, pe1, hfrac, SEED + org_idx, tmp_h1, args.dry_run)
                    seqtk_sample(args.runner, pe2, hfrac, SEED + org_idx, tmp_h2, args.dry_run)
                    seqtk_sample(args.runner, dpe1, dfrac, SEED + org_idx + 7, tmp_d1, args.dry_run)
                    seqtk_sample(args.runner, dpe2, dfrac, SEED + org_idx + 7, tmp_d2, args.dry_run)
                    run(f"zcat {tmp_h1} {tmp_d1} | gzip > {out1}", args.dry_run)
                    run(f"zcat {tmp_h2} {tmp_d2} | gzip > {out2}", args.dry_run)
                    run(f"rm -f {tmp_h1} {tmp_h2} {tmp_d1} {tmp_d2}", args.dry_run)

            rows_out.append([name, cond_id, cov, contam, out1, out2, ref,
                             label_qc, label_qc, label_tool, asm_cfg])

    with open(conditions_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows_out)
    print(f"\nWrote {len(rows_out)} conditions -> {conditions_path}")


if __name__ == "__main__":
    main()
