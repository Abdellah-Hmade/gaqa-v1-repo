import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / 'data' / 'realval'
OUT = OUT_DIR / 'manifest_full.csv'
PIPELINE = OUT_DIR / 'manifest.csv'
ENA_API = 'https://www.ebi.ac.uk/ena/portal/api/search'
EUTILS = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'
ORGANISMS = [('ecoli', 'Escherichia coli K-12 MG1655', 562, 'MG1655|K-?12', 'GCF_000005845.2', 4.6, 50.8, 'train', 'styphi', 'mtb'), ('saureus', 'Staphylococcus aureus NCTC 8325', 1280, 'NCTC ?8325|8325', 'GCF_000013425.1', 2.9, 32.9, 'train', 'bsub', 'mtb'), ('vcholerae', 'Vibrio cholerae N16961', 666, 'N16961', 'GCF_000006745.1', 4.0, 47.5, 'train', 'paeru', 'mtb'), ('bsub', 'Bacillus subtilis 168', 1423, '168', 'GCF_000009045.1', 4.2, 43.5, 'heldout', 'saureus', 'mtb'), ('mtb', 'Mycobacterium tuberculosis H37Rv', 1773, 'H37Rv', 'GCF_000195955.2', 4.4, 65.6, 'heldout', 'none', 'ecoli'), ('paeru', 'Pseudomonas aeruginosa PAO1', 287, 'PAO1', 'GCF_000006765.1', 6.3, 66.6, 'train', 'vcholerae', 'mtb'), ('spneumo', 'Streptococcus pneumoniae R6', 1313, 'R6', 'GCF_000007045.1', 2.0, 39.7, 'train', 'efaec', 'mtb'), ('lmono', 'Listeria monocytogenes EGD-e', 1639, 'EGD-?e', 'GCF_000196035.1', 2.9, 38.0, 'train', 'efaec', 'mtb'), ('styphi', 'Salmonella Typhimurium LT2', 28901, 'LT2', 'GCF_000006945.2', 4.9, 52.2, 'train', 'ecoli', 'mtb'), ('kpneumo', 'Klebsiella pneumoniae MGH 78578', 573, 'MGH ?78578|78578', 'GCF_000016305.1', 5.3, 57.0, 'train', 'ecoli', 'mtb'), ('nmeningo', 'Neisseria meningitidis MC58', 487, 'MC58', 'GCF_000008805.1', 2.3, 51.5, 'heldout', 'paeru', 'mtb'), ('efaec', 'Enterococcus faecalis V583', 1351, 'V583', 'GCF_000007785.1', 3.2, 37.5, 'train', 'lmono', 'mtb')]
MIN_BASES = 80000000
MAX_BASES = 3000000000
MANUAL_RUNS = {'saureus': 'DRR1000166', 'kpneumo': 'SRR15422670', 'nmeningo': 'SRR23299964'}

def ena_fastq_urls(acc):
    prefix = acc[:6]
    digits = acc[3:]
    if len(digits) == 6:
        sub = ''
    elif len(digits) == 7:
        sub = f'/{digits[-1:].zfill(3)}'
    elif len(digits) == 8:
        sub = f'/{digits[-2:].zfill(3)}'
    else:
        sub = f'/{digits[-3:]}'
    base = f'https://ftp.sra.ebi.ac.uk/vol1/fastq/{prefix}{sub}/{acc}/{acc}'
    return (f'{base}_1.fastq.gz', f'{base}_2.fastq.gz')

def http_get(url, timeout=60):
    req = urllib.request.Request(url, headers={'User-Agent': 'gaqa-manifest/1.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode('utf-8', 'replace')

def http_ok(url, timeout=45):
    req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'gaqa-manifest/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status in (200, 301, 302)
    except Exception:
        return False

def find_ena_run(species_taxid, strain_re):
    query = f'tax_tree({species_taxid}) AND library_layout="PAIRED" AND instrument_platform="ILLUMINA"'
    fields = 'run_accession,scientific_name,fastq_ftp,base_count,library_layout,strain,sample_title,sample_alias'
    url = f'{ENA_API}?result=read_run&query={urllib.parse.quote(query)}&fields={fields}&format=tsv&limit=1000'
    try:
        tsv = http_get(url)
    except Exception as e:
        print(f'    ENA query failed: {e}')
        return None
    lines = tsv.strip().split('\n')
    if len(lines) < 2:
        return None
    hdr = lines[0].split('\t')
    rows = [dict(zip(hdr, l.split('\t'))) for l in lines[1:]]
    pat = re.compile(strain_re, re.I)

    def usable(r):
        try:
            bases = int(r.get('base_count') or 0)
        except ValueError:
            return False
        return MIN_BASES <= bases <= MAX_BASES and ';' in (r.get('fastq_ftp') or '')
    strain_hits = [r for r in rows if pat.search(' '.join((r.get(k, '') or '' for k in ('scientific_name', 'strain', 'sample_title', 'sample_alias')))) and usable(r)]
    pool = strain_hits or [r for r in rows if usable(r)]
    if not pool:
        return None
    pool.sort(key=lambda r: abs(int(r['base_count']) - 400000000))
    r = pool[0]
    links = r['fastq_ftp'].split(';')
    pe = [f'https://{l}' for l in links if l.strip()]
    if len(pe) < 2:
        return None
    return (r['run_accession'], pe[0], pe[1], int(r['base_count']), r['scientific_name'], bool(strain_hits))

def resolve_refseq_ftp(gcf):
    term = urllib.parse.quote(f'{gcf}[Assembly Accession]')
    try:
        res = json.loads(http_get(f'{EUTILS}/esearch.fcgi?db=assembly&term={term}&retmode=json'))
        ids = res['esearchresult']['idlist']
        if not ids:
            return None
        time.sleep(0.4)
        summ = json.loads(http_get(f'{EUTILS}/esummary.fcgi?db=assembly&id={ids[0]}&retmode=json'))
        doc = summ['result'][ids[0]]
        ftp = doc.get('ftppath_refseq') or doc.get('ftppath_genbank')
        if not ftp:
            return None
        name = ftp.rstrip('/').split('/')[-1]
        return f"{ftp.replace('ftp://', 'https://')}/{name}_genomic.fna.gz"
    except Exception as e:
        print(f'    eutils failed for {gcf}: {e}')
        return None

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for (key, name, taxid, strain_re, gcf, size, gc, split, rel, dist) in ORGANISMS:
        print(f'[{key}] {name}')
        if key in MANUAL_RUNS:
            acc = MANUAL_RUNS[key]
            (pe1, pe2) = ena_fastq_urls(acc)
            (ok1, ok2) = (http_ok(pe1), http_ok(pe2))
            print(f'  ENA (manual): {acc} urls_ok={ok1 and ok2}')
            if not (ok1 and ok2):
                continue
            strain_match = True
        else:
            run = find_ena_run(taxid, strain_re)
            if not run:
                print(f'  !! no ENA run found — SKIPPING (needs manual accession)')
                continue
            (acc, pe1, pe2, bases, sci, strain_match) = run
            print(f'  ENA: {acc} ({bases / 1000000.0:.0f} Mb) {sci!r} strain_match={strain_match}')
        time.sleep(0.4)
        ref = resolve_refseq_ftp(gcf)
        if not ref:
            print(f'  !! could not resolve {gcf} — SKIPPING')
            continue
        ok = http_ok(ref)
        print(f"  RefSeq: {ref} [{('OK' if ok else 'URL CHECK FAILED')}]")
        if not ok:
            continue
        rows.append({'organism': name, 'genome_size_mb': size, 'gc_pct': gc, 'taxid': taxid, 'split': split, 'reads_url_pe1': pe1, 'reads_url_pe2': pe2, 'reference_url': ref, 'related_donor': rel, 'distant_donor': dist, 'run_accession': acc})
        time.sleep(0.4)
    with OUT.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    pipe = OUT_DIR / 'manifest.csv'
    with pipe.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['organism', 'genome_size_mb', 'reads_url_pe1', 'reads_url_pe2', 'reference_url', 'technology', 'fallback_reads_url'])
        for r in rows:
            w.writerow([r['organism'], r['genome_size_mb'], r['reads_url_pe1'], r['reads_url_pe2'], r['reference_url'], 'Illumina', ''])
    print(f'Pipeline manifest: {pipe}')
    n_train = sum((1 for r in rows if r['split'] == 'train'))
    print(f'\nManifest: {OUT}  ({len(rows)} organisms: {n_train} train, {len(rows) - n_train} heldout)')
    return 0 if len(rows) >= 10 else 1
if __name__ == '__main__':
    sys.exit(main())
