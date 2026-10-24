import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import constants

ENA_API = "https://www.ebi.ac.uk/ena/portal/api/search"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MIN_BASES = 80_000_000
MAX_BASES = 3_000_000_000


def slug(name):
    return re.sub(r"[().]", "", name.replace(" ", "_"))


def ena_fastq_urls(acc):
    prefix = acc[:6]
    digits = acc[3:]
    if len(digits) == 6:
        sub = ""
    elif len(digits) == 7:
        sub = f"/{digits[-1:].zfill(3)}"
    elif len(digits) == 8:
        sub = f"/{digits[-2:].zfill(3)}"
    else:
        sub = f"/{digits[-3:]}"
    base = f"https://ftp.sra.ebi.ac.uk/vol1/fastq/{prefix}{sub}/{acc}/{acc}"
    return f"{base}_1.fastq.gz", f"{base}_2.fastq.gz"


def http_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "gaqa-v1/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def http_ok(url, timeout=45):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "gaqa-v1/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status in (200, 301, 302)
    except Exception:
        return False


def find_ena_run(taxid, strain_re):
    query = f'tax_tree({taxid}) AND library_layout="PAIRED" AND instrument_platform="ILLUMINA"'
    fields = "run_accession,scientific_name,fastq_ftp,base_count,strain,sample_title,sample_alias"
    url = (f"{ENA_API}?result=read_run&query={urllib.parse.quote(query)}"
           f"&fields={fields}&format=tsv&limit=1000")
    try:
        tsv = http_get(url)
    except Exception:
        return None
    lines = tsv.strip().split("\n")
    if len(lines) < 2:
        return None
    hdr = lines[0].split("\t")
    rows = [dict(zip(hdr, line.split("\t"))) for line in lines[1:]]
    pattern = re.compile(strain_re, re.I)

    def usable(r):
        try:
            bases = int(r.get("base_count") or 0)
        except ValueError:
            return False
        return MIN_BASES <= bases <= MAX_BASES and ";" in (r.get("fastq_ftp") or "")

    strain_hits = [r for r in rows
                   if pattern.search(" ".join(r.get(k, "") or "" for k in
                                     ("scientific_name", "strain", "sample_title", "sample_alias")))
                   and usable(r)]
    pool = strain_hits or [r for r in rows if usable(r)]
    if not pool:
        return None
    pool.sort(key=lambda r: abs(int(r["base_count"]) - 400_000_000))
    r = pool[0]
    links = [f"https://{l}" for l in r["fastq_ftp"].split(";") if l.strip()]
    if len(links) < 2:
        return None
    return r["run_accession"], links[0], links[1]


def resolve_refseq_ftp(gcf):
    term = urllib.parse.quote(f"{gcf}[Assembly Accession]")
    try:
        res = json.loads(http_get(f"{EUTILS}/esearch.fcgi?db=assembly&term={term}&retmode=json"))
        ids = res["esearchresult"]["idlist"]
        if not ids:
            return None
        time.sleep(0.4)
        summ = json.loads(http_get(f"{EUTILS}/esummary.fcgi?db=assembly&id={ids[0]}&retmode=json"))
        doc = summ["result"][ids[0]]
        ftp = doc.get("ftppath_refseq") or doc.get("ftppath_genbank")
        if not ftp:
            return None
        name = ftp.rstrip("/").split("/")[-1]
        return f"{ftp.replace('ftp://', 'https://')}/{name}_genomic.fna.gz"
    except Exception:
        return None


def resolve_manifest():
    rows = []
    for b in constants.BACTERIA:
        key = b["key"]
        if key in constants.MANUAL_RUNS:
            acc = constants.MANUAL_RUNS[key]
            pe1, pe2 = ena_fastq_urls(acc)
            if not (http_ok(pe1) and http_ok(pe2)):
                continue
        else:
            run = find_ena_run(b["taxid"], b["strain"])
            if not run:
                continue
            acc, pe1, pe2 = run
        ref = resolve_refseq_ftp(b["gcf"])
        if not ref or not http_ok(ref):
            continue
        rows.append({**b, "run_accession": acc, "reads_url_pe1": pe1,
                     "reads_url_pe2": pe2, "reference_url": ref})
        time.sleep(0.4)
    return rows


def download_file(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return True


def download_all(data_dir, rows):
    data_dir = Path(data_dir)
    reads_dir = data_dir / "reads"
    ref_dir = data_dir / "references"
    for r in rows:
        s = slug(r["name"])
        download_file(r["reads_url_pe1"], reads_dir / s / f"{s}_1.fastq.gz")
        download_file(r["reads_url_pe2"], reads_dir / s / f"{s}_2.fastq.gz")
        download_file(r["reference_url"], ref_dir / f"{s}_genomic.fna.gz")


def write_manifest(data_dir, rows):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    columns = ["organism", "genome_size_mb", "gc_pct", "taxid", "split",
               "reads_url_pe1", "reads_url_pe2", "reference_url",
               "related_donor", "distant_donor", "run_accession"]
    with (data_dir / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({"organism": r["name"], "genome_size_mb": r["size_mb"],
                        "gc_pct": r["gc"], "taxid": r["taxid"], "split": r["split"],
                        "reads_url_pe1": r["reads_url_pe1"], "reads_url_pe2": r["reads_url_pe2"],
                        "reference_url": r["reference_url"], "related_donor": r["related"],
                        "distant_donor": r["distant"], "run_accession": r["run_accession"]})


def main():
    data_dir = constants.DATA_DIR
    rows = resolve_manifest()
    print(f"resolved {len(rows)}/{len(constants.BACTERIA)} organisms")
    write_manifest(data_dir, rows)
    download_all(data_dir, rows)
    print(f"downloaded reads + references into {data_dir}")


if __name__ == "__main__":
    main()
