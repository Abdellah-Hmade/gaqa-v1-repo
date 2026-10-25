import html
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'data' / 'knowledge_sources'
OUT = ROOT / 'data' / 'gaqa_knowledge' / 'chunks.jsonl'
GTN_BASE = 'https://training.galaxyproject.org/training-material'
WORDS_PER_CHUNK = 380
OVERLAP_WORDS = 60
MIN_WORDS = 80
CC_BY = 'CC-BY-4.0'
TOOL_LICENSES = {'spades': 'GPLv2 (doc use with attribution)', 'quast': 'GPLv2 (doc use with attribution)', 'busco': 'MIT (doc use with attribution)', 'checkm': 'GPLv3 (doc use with attribution)', 'prokka': 'GPLv2 (doc use with attribution)'}
GTN_TOPICS = ['assembly', 'genome-annotation', 'sequence-analysis']
SEQ_ANALYSIS_KEEP = {'quality-control', 'quality-contamination-control', 'human-reads-removal', 'ncbi-fcs', 'mapping'}

def clean_gtn_md(text: str) -> str:
    text = re.sub('\\A---\\n.*?\\n---\\n', '', text, flags=re.S)
    text = re.sub('\\{%.*?%\\}', '', text, flags=re.S)
    text = re.sub('<!--.*?-->', '', text, flags=re.S)
    text = re.sub('!\\[[^\\]]*\\]\\([^)]*\\)', '', text)
    text = re.sub('\\{:[^}]*\\}', '', text)
    text = re.sub('^(\\s*> ?)+', '> ', text, flags=re.M)
    text = re.sub('^> $', '', text, flags=re.M)
    text = re.sub('\\n{3,}', '\n\n', text)
    return text.strip()

def html_to_text(path: Path) -> str:
    out = subprocess.run(['pandoc', '-f', 'html', '-t', 'plain', '--wrap=none', str(path)], capture_output=True, text=True)
    if out.returncode != 0:
        print(f'  [warn] pandoc failed on {path.name}: {out.stderr[:120]}')
        return _re_html_to_text(path.read_text(errors='replace'))
    text = html.unescape(out.stdout)
    text = re.sub('\\n{3,}', '\n\n', text)
    return text.strip()

def _re_html_to_text(html_str: str) -> str:
    text = re.sub('<script[^>]*>.*?</script>', '', html_str, flags=re.S | re.I)
    text = re.sub('<style[^>]*>.*?</style>', '', text, flags=re.S | re.I)
    text = re.sub('<[^>]+>', ' ', text)
    text = re.sub('&amp;', '&', text)
    text = re.sub('&lt;', '<', text)
    text = re.sub('&gt;', '>', text)
    text = re.sub('&#?\\w+;', ' ', text)
    text = re.sub('\\s+', ' ', text)
    return text.strip()

def chunk_text(text: str):
    paragraphs = [p.strip() for p in re.split('\\n\\s*\\n', text) if p.strip()]
    (chunks, buf, buf_len) = ([], [], 0)
    for para in paragraphs:
        words = para.split()
        if buf_len + len(words) > WORDS_PER_CHUNK and buf:
            chunks.append(' '.join(buf))
            tail = ' '.join(buf).split()[-OVERLAP_WORDS:]
            (buf, buf_len) = ([' '.join(tail)], len(tail))
        buf.append(para)
        buf_len += len(words)
    if buf and buf_len >= MIN_WORDS:
        chunks.append(' '.join(buf))
    if len(chunks) >= 2 and len(chunks[-1].split()) < MIN_WORDS:
        chunks[-2] += '\n\n' + chunks[-1]
        chunks.pop()
    return [c for c in chunks if len(c.split()) >= MIN_WORDS]

def iter_gtn():
    for topic in GTN_TOPICS:
        tdir = SRC / 'gtn' / 'topics' / topic / 'tutorials'
        if not tdir.exists():
            continue
        for tut in sorted(tdir.iterdir()):
            if not tut.is_dir():
                continue
            if topic == 'sequence-analysis' and tut.name not in SEQ_ANALYSIS_KEEP:
                continue
            for fname in ('tutorial.md', 'slides.md'):
                f = tut / fname
                if not f.exists():
                    continue
                text = clean_gtn_md(f.read_text(errors='replace'))
                if len(text.split()) < MIN_WORDS:
                    continue
                kind = 'tutorial' if fname == 'tutorial.md' else 'slides'
                url = f'{GTN_BASE}/topics/{topic}/tutorials/{tut.name}/{kind}.html'
                yield (f'gtn_{topic}_{tut.name}_{kind}', f'GTN {topic}: {tut.name} ({kind})', url, CC_BY, text)

def iter_tool_docs():
    td = SRC / 'tool_docs'
    for f in sorted(td.glob('*.html')):
        tool = f.name.split('_')[0]
        text = html_to_text(f)
        if len(text.split()) < MIN_WORDS:
            continue
        urls = {'spades': f"https://ablab.github.io/spades/{f.name[len('spades_'):]}", 'quast': 'http://quast.sourceforge.net/docs/manual.html', 'busco': 'https://busco.ezlab.org/busco_userguide.html'}
        yield (f'tool_{f.stem}', f'{tool} docs: {f.stem}', urls.get(tool, ''), TOOL_LICENSES.get(tool, ''), text)
    for f in sorted(td.glob('*.md')):
        tool = f.name.split('_')[0]
        text = f.read_text(errors='replace')
        if len(text.split()) < MIN_WORDS:
            continue
        yield (f'tool_{f.stem}', f'{tool} docs: {f.stem}', '', TOOL_LICENSES.get(tool, ''), text)
    wiki = td / 'checkm_wiki'
    if wiki.exists():
        for f in sorted(wiki.glob('*.md')):
            text = f.read_text(errors='replace')
            if len(text.split()) < MIN_WORDS:
                continue
            yield (f'tool_checkm_wiki_{f.stem}', f'CheckM wiki: {f.stem}', f'https://github.com/Ecogenomics/CheckM/wiki/{f.stem}', TOOL_LICENSES['checkm'], text)

def iter_ebi():
    ed = SRC / 'embl_ebi'
    base = 'https://www.ebi.ac.uk/training/online/courses/functional-genomics-ii-common-technologies-and-data-analysis-methods/next-generation-sequencing'
    for f in sorted(ed.glob('*.html')):
        text = html_to_text(f)
        if len(text.split()) < MIN_WORDS:
            continue
        slug = f.stem.replace('ngs_', '').replace('_', '/')
        url = base if slug == 'index' else f'{base}/{slug}/'
        yield (f'ebi_{f.stem}', f'EMBL-EBI NGS: {f.stem}', url, CC_BY, text)
EXTRA_SOURCES = [('flye', 'https://raw.githubusercontent.com/mikolmogorov/Flye/master/docs/USAGE.md', '', CC_BY), ('canu', '', 'https://canu.readthedocs.io/en/latest/quick-start.html', CC_BY), ('unicycler', 'https://raw.githubusercontent.com/rrwick/Unicycler/master/README.md', '', CC_BY), ('pilon', 'https://raw.githubusercontent.com/wiki/broadinstitute/pilon/Home.md', '', CC_BY), ('kraken2', 'https://raw.githubusercontent.com/DerrickWood/kraken2/master/docs/MANUAL.markdown', '', CC_BY), ('medaka', 'https://raw.githubusercontent.com/nanoporetech/medaka/master/README.md', '', CC_BY), ('racon', 'https://raw.githubusercontent.com/isovic/racon/master/README.md', '', CC_BY), ('bandage', 'https://raw.githubusercontent.com/wiki/rrwick/Bandage/Home.md', '', CC_BY), ('hifiasm', 'https://raw.githubusercontent.com/chhylp123/hifiasm/master/README.md', '', CC_BY)]

def _fetch_raw_md(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'GAQA-chunker/1.0'})
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.read().decode(errors='replace')
    except Exception:
        return ''

def iter_extra_tool_docs():
    for (name, raw_url, html_url, lic) in EXTRA_SOURCES:
        text = ''
        if raw_url:
            text = _fetch_raw_md(raw_url)
        if (not text or len(text.split()) < MIN_WORDS) and html_url:
            text = _fetch_text(html_url)
        if not text or len(text.split()) < MIN_WORDS:
            print(f'  [warn] extra_doc: could not fetch {name}')
            continue
        url = raw_url or html_url
        yield (f'extra_{name}', f'{name} docs', url, lic, text)

def _fetch_text(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'GAQA-chunker/1.0'})
        resp = urllib.request.urlopen(req, timeout=30)
        content = resp.read().decode(errors='replace')
        if '<html' in content[:200].lower() or url.endswith('.html'):
            return _re_html_to_text(content)
        return content
    except Exception as e:
        print(f'  [debug] fetch failed: {url[:80]} — {e}')
        return ''

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    (n_docs, n_chunks) = (0, 0)
    per_source = {}
    with OUT.open('w') as out:
        for it in (iter_gtn(), iter_tool_docs(), iter_ebi(), iter_extra_tool_docs()):
            for (doc_id, title, url, lic, text) in it:
                chunks = chunk_text(text)
                n_docs += 1
                per_source[doc_id] = len(chunks)
                for (i, c) in enumerate(chunks):
                    rec = {'chunk_id': f'{doc_id}#{i}', 'doc_id': doc_id, 'title': title, 'url': url, 'license': lic, 'text': c}
                    out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    n_chunks += 1
    total_words = sum((len(json.loads(l)['text'].split()) for l in OUT.open()))
    print(f'Documents: {n_docs}')
    print(f'Chunks:    {n_chunks}  (avg {total_words / max(n_chunks, 1):.0f} words)')
    print(f'Output:    {OUT}')
    print('\nTop docs by chunk count:')
    for (doc, c) in sorted(per_source.items(), key=lambda x: -x[1])[:15]:
        print(f'  {c:3d}  {doc}')
if __name__ == '__main__':
    sys.exit(main())
