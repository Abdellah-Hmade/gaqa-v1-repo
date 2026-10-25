import argparse
import csv
import gzip
import os
import re
import subprocess
import sys
import urllib.request
SEED = 42
LONG_READ_ASSEMBLERS = {'flye', 'canu', 'unicycler'}
CONDITIONS = [('C01', 10, 0, None, 'low_coverage', 'default'), ('C02', 30, 0, None, 'acceptable', 'default'), ('C03', 100, 0, None, 'acceptable', 'default'), ('C04', 30, 5, 'related', 'contamination', 'default'), ('C05', 100, 10, 'related', 'contamination', 'default'), ('C06', 100, 10, 'distant', 'contamination', 'default'), ('C07', 30, 10, 'distant', 'contamination', 'default'), ('C08', 100, 5, 'related', 'contamination', 'default'), ('C09', 30, 10, 'related', 'contamination', 'default'), ('C10', 100, 0, None, 'misassembly', 'wrong'), ('C11', 100, 0, None, 'misassembly', 'megahit'), ('C12', 50, 0, None, 'acceptable', 'default'), ('C13', 200, 0, None, 'acceptable', 'default'), ('C14', 30, 1, 'related', 'contamination', 'default'), ('C15', 100, 20, 'related', 'contamination', 'default'), ('C16', 30, 0, None, 'acceptable', 'flye'), ('C17', 100, 0, None, 'acceptable', 'flye'), ('C18', 30, 5, 'related', 'contamination', 'flye'), ('C19', 30, 0, None, 'acceptable', 'canu'), ('C20', 50, 0, None, 'acceptable', 'unicycler')]

def slugify(organism: str) -> str:
    return re.sub('[().]', '', organism.replace(' ', '_'))

def run(cmd, dry_run=False):
    if dry_run:
        print(f'  [dry] {cmd}')
        return
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode()[-3000:] if e.stderr else '(no stderr)'
        print(f'  [ERROR] Command failed (exit {e.returncode}): {cmd[:200]}')
        print(f'  stderr: {stderr}')
        raise

def total_bases(fastq_gz, runner):
    cmd = f'{runner} seqkit stats -T {fastq_gz}'.split() if runner else ['seqkit', 'stats', '-T', fastq_gz]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode == 0:
        lines = out.stdout.strip().split('\n')
        if len(lines) >= 2:
            (hdr, vals) = (lines[0].split('\t'), lines[1].split('\t'))
            row = dict(zip(hdr, vals))
            for k in ('sum_len', 'Total_length', 'total_length'):
                if k in row:
                    return int(float(row[k]))
            return int(float(vals[4].replace(',', '')))
    raise RuntimeError(f'seqkit stats failed on {fastq_gz}: {out.stderr[:200]}')

def subsample_paired(runner, in1, in2, frac, seed, out1, out2, dry_run):
    if os.path.exists(out1) and os.path.getsize(out1) > 1024 and (not dry_run):
        return
    if frac >= 1.0:
        run(f'cp {in1} {out1}', dry_run)
        run(f'cp {in2} {out2}', dry_run)
    else:
        exe = f'{runner} seqtk' if runner else 'seqtk'
        run(f'zcat {in1} | {exe} sample -s{seed} - {frac:.6f} | gzip > {out1}', dry_run)
        run(f'zcat {in2} | {exe} sample -s{seed} - {frac:.6f} | gzip > {out2}', dry_run)

def subsample_long_reads(runner, in_long, frac, seed, out_long, dry_run):
    if os.path.exists(out_long) and os.path.getsize(out_long) > 1024 and (not dry_run):
        return
    if frac >= 1.0:
        run(f'cp {in_long} {out_long}', dry_run)
    else:
        exe = f'{runner} seqtk' if runner else 'seqtk'
        run(f'zcat {in_long} | {exe} sample -s{seed} - {frac:.6f} | gzip > {out_long}', dry_run)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data/realval')
    ap.add_argument('--runner', default='', help='env runner prefix, e.g. "micromamba run -n realval"')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    data_dir = os.path.abspath(args.data_dir)
    manifest_path = os.path.join(data_dir, 'manifest_full.csv')
    conditions_path = os.path.join(data_dir, 'conditions.csv')
    sub_dir = os.path.join(data_dir, 'subsampled')
    os.makedirs(sub_dir, exist_ok=True)
    if os.path.exists(conditions_path) and (not args.force):
        print(f'conditions.csv already exists at {conditions_path} (use --force to regenerate)')
        return
    with open(manifest_path) as f:
        orgs = list(csv.DictReader(f))
    by_key = {}
    for o in orgs:
        by_key[o['organism']] = o
    key_map = {'ecoli': 'Escherichia coli K-12 MG1655', 'saureus': 'Staphylococcus aureus NCTC 8325', 'vcholerae': 'Vibrio cholerae N16961', 'bsub': 'Bacillus subtilis 168', 'mtb': 'Mycobacterium tuberculosis H37Rv', 'paeru': 'Pseudomonas aeruginosa PAO1', 'spneumo': 'Streptococcus pneumoniae R6', 'lmono': 'Listeria monocytogenes EGD-e', 'styphi': 'Salmonella Typhimurium LT2', 'kpneumo': 'Klebsiella pneumoniae MGH 78578', 'nmeningo': 'Neisseria meningitidis MC58', 'efaec': 'Enterococcus faecalis V583'}
    header = ['organism', 'condition_id', 'coverage', 'contamination_pct', 'reads_pe1', 'reads_pe2', 'reads_long', 'long_read_type', 'reference', 'label_qc', 'label_remediation', 'label_tool', 'assembler_config']
    rows_out = []
    for (org_idx, org) in enumerate(orgs, start=1):
        name = org['organism']
        slug = slugify(name)
        size_mb = float(org['genome_size_mb'])
        genome_bp = size_mb * 1000000
        pe1 = os.path.join(data_dir, 'reads', slug, f'{slug}_1.fastq.gz')
        pe2 = os.path.join(data_dir, 'reads', slug, f'{slug}_2.fastq.gz')
        ref = os.path.join(data_dir, 'references', f'{slug}_genomic.fna.gz')
        if not (os.path.exists(pe1) and os.path.exists(pe2)):
            print(f'[skip] {name}: reads not downloaded yet ({pe1})')
            continue
        if not os.path.exists(ref):
            print(f'[skip] {name}: reference missing ({ref})')
            continue
        long_reads_url = org.get('long_reads_url', '').strip()
        long_reads_path = ''
        long_read_type = ''
        tech = org.get('technology', 'Illumina')
        if long_reads_url:
            if 'nanopore' in tech.lower() or 'ont' in tech.lower():
                long_read_type = 'nanopore'
            else:
                long_read_type = 'pacbio'
        if long_reads_url:
            lr_dir = os.path.join(data_dir, 'longreads')
            os.makedirs(lr_dir, exist_ok=True)
            lr_file = os.path.join(lr_dir, f'{slug}_long.fastq.gz')
            if not os.path.exists(lr_file) and (not args.dry_run):
                print(f'[{slug}] downloading long reads: {long_reads_url}')
                try:
                    _last_pct = [0]

                    def _progress(n_blocks, block_size, total_size):
                        if total_size <= 0:
                            return
                        pct = min(int(n_blocks * block_size * 100 / total_size), 100)
                        if pct - _last_pct[0] >= 5:
                            _last_pct[0] = pct
                            mb = n_blocks * block_size / 1000000.0
                            sys.stderr.write(f'\r  [{slug}] {pct}%  ({mb:.0f} MB)\x1b[K')
                            sys.stderr.flush()
                    urllib.request.urlretrieve(long_reads_url.replace('https://', 'http://'), lr_file, _progress)
                    sys.stderr.write(f'\r  [{slug}] 100% — done\x1b[K\n')
                    sys.stderr.flush()
                except Exception as e:
                    print(f'  [warn] long-read download failed ({e}) — C16-C20 will be skipped')
                    long_reads_url = ''
            if os.path.exists(lr_file):
                long_reads_path = lr_file
        print(f'[{slug}] estimating depth ...')
        if args.dry_run:
            host_bases = int(genome_bp * 100)
        else:
            host_bases = total_bases(pe1, args.runner)
        print(f'  host bases: {host_bases / 1000000.0:.0f} Mb (~{host_bases / genome_bp:.0f}x)')
        donors = {'related': org.get('related_donor'), 'distant': org.get('distant_donor')}
        donor_bases = {}
        for (kind, dkey) in donors.items():
            if not dkey or dkey == 'none':
                donor_bases[kind] = None
                continue
            dname = key_map[dkey]
            dslug = slugify(dname)
            dpe1 = os.path.join(data_dir, 'reads', dslug, f'{dslug}_1.fastq.gz')
            if not os.path.exists(dpe1):
                print(f'  [warn] donor {dname} reads missing — {kind} contamination conditions will be skipped')
                donor_bases[kind] = None
                continue
            try:
                dbases_val = None if args.dry_run else total_bases(dpe1, args.runner)
            except Exception as e:
                print(f'  [warn] donor {dname} — total_bases failed ({e}), skipping {kind}')
                donor_bases[kind] = None
                continue
            donor_bases[kind] = (dpe1, dpe1.replace('_1.fastq.gz', '_2.fastq.gz'), dslug, dbases_val)
        for (suffix, cov, contam, donor_kind, label_qc, asm_cfg) in CONDITIONS:
            cond_id = f'{org_idx:02d}{suffix}'
            label_tool = 'correct_config'
            if asm_cfg == 'wrong':
                label_tool = 'wrong_config'
            elif asm_cfg == 'megahit':
                label_tool = 'wrong_tool'
            if asm_cfg in LONG_READ_ASSEMBLERS:
                if not long_reads_path:
                    print(f'  {cond_id}: SKIPPED (no long reads for {slug})')
                    continue
                if contam > 0:
                    print(f'  {cond_id}: SKIPPED (contamination not supported for long reads)')
                    continue
                print(f'  {cond_id}: {cov}x {asm_cfg} ({slug}, {long_read_type})')
                rows_out.append([name, cond_id, cov, contam, '', '', long_reads_path, long_read_type, ref, label_qc, label_qc, label_tool, asm_cfg])
                continue
            target_bases = cov * genome_bp
            host_frac_reads = 1.0 - contam / 100.0
            host_target = target_bases * host_frac_reads
            hfrac = host_target / host_bases
            if contam == 0:
                base_cond = f'{org_idx:02d}' + {10: 'C01', 30: 'C02', 100: 'C03', 50: 'C12', 200: 'C13'}[cov]
                out1 = os.path.join(sub_dir, f'{base_cond}_{slug}_{cov}x_1.fastq.gz')
                out2 = out1.replace('_1.fastq.gz', '_2.fastq.gz')
                print(f'  {cond_id}: {cov}x clean ({asm_cfg}) -> {os.path.basename(out1)}')
                try:
                    subsample_paired(args.runner, pe1, pe2, hfrac, SEED + org_idx, out1, out2, args.dry_run)
                except Exception as e:
                    print(f'  [SKIP] {cond_id}: subsample failed — {e}')
                    continue
            else:
                d = donor_bases.get(donor_kind)
                if d is None:
                    print(f'  {cond_id}: SKIPPED (no {donor_kind} donor)')
                    continue
                (dpe1, dpe2, dslug, dbases) = d
                if args.dry_run:
                    dbases = host_bases
                donor_target = target_bases * (contam / 100.0)
                dfrac = donor_target / dbases
                tmp_h1 = os.path.join(sub_dir, f'_tmp_{cond_id}_h1.fastq.gz')
                tmp_h2 = os.path.join(sub_dir, f'_tmp_{cond_id}_h2.fastq.gz')
                tmp_d1 = os.path.join(sub_dir, f'_tmp_{cond_id}_d1.fastq.gz')
                tmp_d2 = os.path.join(sub_dir, f'_tmp_{cond_id}_d2.fastq.gz')
                out1 = os.path.join(sub_dir, f'{cond_id}_{slug}_{cov}x_spike{contam}pct_{dslug}_1.fastq.gz')
                out2 = out1.replace('_1.fastq.gz', '_2.fastq.gz')
                print(f'  {cond_id}: {cov}x + {contam}% {donor_kind} ({dslug})')
                if args.dry_run or not (os.path.exists(out1) and os.path.getsize(out1) > 0):
                    try:
                        subsample_paired(args.runner, pe1, pe2, hfrac, SEED + org_idx, tmp_h1, tmp_h2, args.dry_run)
                        subsample_paired(args.runner, dpe1, dpe2, dfrac, SEED + org_idx + 7, tmp_d1, tmp_d2, args.dry_run)
                    except Exception as e:
                        print(f'  [SKIP] {cond_id}: subsample failed — {e}')
                        continue
                    run(f'zcat {tmp_h1} {tmp_d1} | gzip > {out1}', args.dry_run)
                    run(f'zcat {tmp_h2} {tmp_d2} | gzip > {out2}', args.dry_run)
                    run(f'rm -f {tmp_h1} {tmp_h2} {tmp_d1} {tmp_d2}', args.dry_run)
            rows_out.append([name, cond_id, cov, contam, out1, out2, '', '', ref, label_qc, label_qc, label_tool, asm_cfg])
    with open(conditions_path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        w.writerows(rows_out)
    print(f'\nWrote {len(rows_out)} conditions -> {conditions_path}')
if __name__ == '__main__':
    main()
