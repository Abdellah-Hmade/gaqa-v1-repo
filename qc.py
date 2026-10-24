import csv
import gzip
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import constants
import data
from download import slug

RUNNER = shlex.split(constants.ENV_RUNNER) if constants.ENV_RUNNER else []


def run(args, stdout=None):
    subprocess.run(RUNNER + args, check=True, stdout=stdout, stderr=subprocess.DEVNULL)


def total_bases(fastq):
    try:
        out = subprocess.run(RUNNER + ["seqkit", "stats", "-T", str(fastq)],
                             capture_output=True, text=True)
        if out.returncode == 0:
            lines = out.stdout.strip().split("\n")
            if len(lines) >= 2:
                row = dict(zip(lines[0].split("\t"), lines[1].split("\t")))
                for k in ("sum_len", "Total_length", "total_length"):
                    if k in row:
                        return int(float(row[k]))
                return int(float(lines[1].split("\t")[4].replace(",", "")))
    except Exception:
        pass
    size = sum(Path(f).stat().st_size for f in (fastq,) if Path(f).exists())
    return size * 3 // 2


def sample_to_gz(inp, frac, out_gz, seed=42):
    if Path(out_gz).exists() and Path(out_gz).stat().st_size > 0:
        return
    Path(out_gz).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out_gz) + ".tmp"
    with open(tmp, "wb") as fh:
        run(["seqtk", "sample", f"-s{seed}", str(inp), f"{min(frac, 1.0):.6f}"], stdout=fh)
    with open(tmp, "rb") as src, gzip.open(out_gz, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove(tmp)


def merge_to_gz(parts, out_gz):
    if Path(out_gz).exists() and Path(out_gz).stat().st_size > 0:
        return
    with gzip.open(out_gz, "wb") as dst:
        for part in parts:
            with open(part, "rb") as src:
                shutil.copyfileobj(src, dst)
            os.remove(part)


def sample_to_plain(inp, frac, out, seed=42):
    with open(out, "wb") as fh:
        run(["seqtk", "sample", f"-s{seed}", str(inp), f"{min(frac, 1.0):.6f}"], stdout=fh)


def make_conditions(data_dir):
    data_dir = Path(data_dir)
    sub_dir = data_dir / "subsampled"
    sub_dir.mkdir(parents=True, exist_ok=True)
    manifest = list(csv.DictReader((data_dir / "manifest.csv").open()))
    by_key = {b["key"]: b for b in constants.BACTERIA}
    header = ["organism", "condition_id", "coverage", "contamination_pct", "reads_pe1",
              "reads_pe2", "reference", "label_qc", "label_remediation", "label_tool",
              "assembler_config"]
    rows_out = []
    for idx, org in enumerate(manifest, start=1):
        name = org["organism"]
        s = slug(name)
        genome_bp = float(org["genome_size_mb"]) * 1_000_000
        pe1 = data_dir / "reads" / s / f"{s}_1.fastq.gz"
        pe2 = data_dir / "reads" / s / f"{s}_2.fastq.gz"
        ref = data_dir / "references" / f"{s}_genomic.fna.gz"
        if not (pe1.exists() and pe2.exists() and ref.exists()):
            continue
        host_bases = total_bases(pe1)
        key = next((b["key"] for b in constants.BACTERIA if b["name"] == name), None)
        donors = {}
        if key:
            for kind in ("related", "distant"):
                dkey = by_key[key][kind]
                if dkey and dkey != "none" and dkey in by_key:
                    dname = by_key[dkey]["name"]
                    ds = slug(dname)
                    donors[kind] = (data_dir / "reads" / ds / f"{ds}_1.fastq.gz",
                                    data_dir / "reads" / ds / f"{ds}_2.fastq.gz")
        for suffix, cov, contam, donor_kind, label, asm_cfg in data.CONDITIONS:
            cid = f"{idx:02d}{suffix}"
            target = cov * genome_bp
            hfrac = target * (1 - contam / 100.0) / max(host_bases, 1)
            if contam == 0:
                out1 = sub_dir / f"{cid}_{s}_{cov}x_1.fastq.gz"
                out2 = sub_dir / f"{cid}_{s}_{cov}x_2.fastq.gz"
                sample_to_gz(pe1, hfrac, out1, constants.SEED + idx)
                sample_to_gz(pe2, hfrac, out2, constants.SEED + idx)
            else:
                d = donors.get(donor_kind)
                if not d:
                    continue
                dbases = total_bases(d[0])
                dfrac = target * (contam / 100.0) / max(dbases, 1)
                h1, h2 = f"{sub_dir}/_h_{cid}_1", f"{sub_dir}/_h_{cid}_2"
                d1, d2 = f"{sub_dir}/_d_{cid}_1", f"{sub_dir}/_d_{cid}_2"
                sample_to_plain(pe1, hfrac, h1, constants.SEED + idx)
                sample_to_plain(pe2, hfrac, h2, constants.SEED + idx)
                sample_to_plain(d[0], dfrac, d1, constants.SEED + idx + 7)
                sample_to_plain(d[1], dfrac, d2, constants.SEED + idx + 7)
                out1 = sub_dir / f"{cid}_{s}_{cov}x_spike{contam}pct_1.fastq.gz"
                out2 = sub_dir / f"{cid}_{s}_{cov}x_spike{contam}pct_2.fastq.gz"
                merge_to_gz([h1, d1], out1)
                merge_to_gz([h2, d2], out2)
            label_tool = "wrong_config" if asm_cfg == "wrong" else ("wrong_tool" if asm_cfg == "megahit" else "correct_config")
            rows_out.append([name, cid, cov, contam, str(out1), str(out2), str(ref),
                             label, label, label_tool, asm_cfg])
    with (data_dir / "conditions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows_out)
    print(f"conditions: {len(rows_out)} -> {data_dir / 'conditions.csv'}")


def assemble(data_dir):
    data_dir = Path(data_dir)
    conditions = list(csv.DictReader((data_dir / "conditions.csv").open()))
    for cond in conditions:
        cid = cond["condition_id"]
        coverage = int(cond["coverage"])
        cfg = cond["assembler_config"]
        cond_dir = data_dir / "assemblies" / cid
        if cfg == "megahit":
            out = cond_dir / "megahit"
            if (out / "final.contigs.fa").exists():
                continue
            run(["megahit", "-t", "4", "-o", str(out), "-1", cond["reads_pe1"], "-2", cond["reads_pe2"]])
            fasta = out / "contigs.fasta"
            if fasta.exists() or fasta.is_symlink():
                fasta.unlink()
            fasta.symlink_to("final.contigs.fa")
        else:
            out = cond_dir / "spades"
            if (out / "contigs.fasta").exists():
                continue
            kmers = "21,33,55" if coverage <= 20 else ("21" if cfg == "wrong" else "21,33,55,77")
            run(["spades.py", "-t", "4", "-o", str(out), "--pe1-1", cond["reads_pe1"],
                 "--pe1-2", cond["reads_pe2"], "-k", kmers])
    print(f"assembled conditions into {data_dir / 'assemblies'}")


def run_qc(data_dir):
    data_dir = Path(data_dir)
    qc_dir = data_dir / "qc_results"
    conditions = list(csv.DictReader((data_dir / "conditions.csv").open()))
    for cond in conditions:
        cid = cond["condition_id"]
        subdir = "megahit" if cond["assembler_config"] == "megahit" else "spades"
        fasta = data_dir / "assemblies" / cid / subdir / "contigs.fasta"
        if not fasta.exists():
            continue
        out = qc_dir / cid
        (out / "quast").mkdir(parents=True, exist_ok=True)
        if not (out / "quast" / "report.tsv").exists():
            run(["quast.py", "-o", str(out / "quast"), "-R", cond["reference"],
                 "--gene-finding", "-t", "2", str(fasta)])
        if not (out / "busco").exists():
            run(["busco", "-i", str(fasta), "-o", str(out / "busco"), "-m", "genome",
                 "-l", constants.BUSCO_LINEAGE, "-c", "2", "--offline"])
        bins = out / "checkm" / "bins"
        bins.mkdir(parents=True, exist_ok=True)
        linked = bins / "contigs.fasta"
        if not linked.exists():
            linked.symlink_to(fasta)
        if not (out / "checkm" / "qa.txt").exists():
            log = open(out / "checkm" / "lineage.log", "w")
            run(["checkm", "lineage_wf", "-t", "2", "-x", "fasta", str(bins), str(out / "checkm" / "wf")], stdout=log)
            run(["checkm", "qa", "-t", "2", "--tab_table", "-f", str(out / "checkm" / "qa.txt"),
                 str(out / "checkm" / "wf" / "lineage.ms"), str(out / "checkm" / "wf")], stdout=log)
            log.close()
    print(f"QC results into {qc_dir}")


def main():
    make_conditions(constants.DATA_DIR)
    assemble(constants.DATA_DIR)
    run_qc(constants.DATA_DIR)


if __name__ == "__main__":
    main()
