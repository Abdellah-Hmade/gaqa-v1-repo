"""Chunk GAQA knowledge sources into ~500-token passages for grounded generation.

Sources:
  - GTN training-material (topics/assembly, topics/genome-annotation,
    topics/sequence-analysis): tutorial.md / slides.md — CC-BY 4.0
  - Tool docs: SPAdes (mkdocs html), QUAST (manual html), BUSCO (userguide html),
    CheckM (wiki md), Prokka (README md) — see per-file licenses
  - EMBL-EBI Train Online NGS course pages (html) — CC-BY 4.0

Output: data/gaqa_knowledge/chunks.jsonl
  {chunk_id, source, title, url, license, text}

Usage: python -m src.chunk_knowledge_sources
"""

import html
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "knowledge_sources"
OUT = ROOT / "data" / "gaqa_knowledge" / "chunks.jsonl"

GTN_BASE = "https://training.galaxyproject.org/training-material"
WORDS_PER_CHUNK = 380  # ~500 tokens
OVERLAP_WORDS = 60
MIN_WORDS = 80

CC_BY = "CC-BY-4.0"
TOOL_LICENSES = {
    "spades": "GPLv2 (doc use with attribution)",
    "quast": "GPLv2 (doc use with attribution)",
    "busco": "MIT (doc use with attribution)",
    "checkm": "GPLv3 (doc use with attribution)",
    "prokka": "GPLv2 (doc use with attribution)",
}

GTN_TOPICS = ["assembly", "genome-annotation", "sequence-analysis"]

# sequence-analysis is broad; keep only QC / contamination / NGS-relevant tutorials
SEQ_ANALYSIS_KEEP = {
    "quality-control", "quality-contamination-control", "human-reads-removal",
    "ncbi-fcs", "mapping",
}


def clean_gtn_md(text: str) -> str:
    """Strip GTN/jekyll markup that adds noise for the teacher model."""
    # YAML front matter
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)
    # Jekyll/liquid tags {% ... %} and includes
    text = re.sub(r"\{%.*?%\}", "", text, flags=re.S)
    # HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    # Images ![alt](url)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    # Kramdown attribute lists {: ...}
    text = re.sub(r"\{:[^}]*\}", "", text)
    # Collapse blockquote nesting markers left by hands-on/question boxes
    text = re.sub(r"^(\s*> ?)+", "> ", text, flags=re.M)
    text = re.sub(r"^> $", "", text, flags=re.M)
    # Collapse >3 blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_text(path: Path) -> str:
    out = subprocess.run(
        ["pandoc", "-f", "html", "-t", "plain", "--wrap=none", str(path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        print(f"  [warn] pandoc failed on {path.name}: {out.stderr[:120]}")
        return ""
    text = html.unescape(out.stdout)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str):
    """Split into overlapping word windows, preferring paragraph boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf, buf_len = [], [], 0
    for para in paragraphs:
        words = para.split()
        if buf_len + len(words) > WORDS_PER_CHUNK and buf:
            chunks.append(" ".join(buf))
            # overlap: keep tail words
            tail = " ".join(buf).split()[-OVERLAP_WORDS:]
            buf, buf_len = [" ".join(tail)], len(tail)
        buf.append(para)
        buf_len += len(words)
    if buf and buf_len >= MIN_WORDS:
        chunks.append(" ".join(buf))
    # merge a too-small final chunk into the previous one
    if len(chunks) >= 2 and len(chunks[-1].split()) < MIN_WORDS:
        chunks[-2] += "\n\n" + chunks[-1]
        chunks.pop()
    return [c for c in chunks if len(c.split()) >= MIN_WORDS]


def iter_gtn():
    for topic in GTN_TOPICS:
        tdir = SRC / "gtn" / "topics" / topic / "tutorials"
        if not tdir.exists():
            continue
        for tut in sorted(tdir.iterdir()):
            if not tut.is_dir():
                continue
            if topic == "sequence-analysis" and tut.name not in SEQ_ANALYSIS_KEEP:
                continue
            for fname in ("tutorial.md", "slides.md"):
                f = tut / fname
                if not f.exists():
                    continue
                text = clean_gtn_md(f.read_text(errors="replace"))
                if len(text.split()) < MIN_WORDS:
                    continue
                kind = "tutorial" if fname == "tutorial.md" else "slides"
                url = f"{GTN_BASE}/topics/{topic}/tutorials/{tut.name}/{kind}.html"
                yield (f"gtn_{topic}_{tut.name}_{kind}",
                       f"GTN {topic}: {tut.name} ({kind})", url, CC_BY, text)


def iter_tool_docs():
    td = SRC / "tool_docs"
    for f in sorted(td.glob("*.html")):
        tool = f.name.split("_")[0]
        text = html_to_text(f)
        if len(text.split()) < MIN_WORDS:
            continue
        urls = {
            "spades": f"https://ablab.github.io/spades/{f.name[len('spades_'):]}",
            "quast": "http://quast.sourceforge.net/docs/manual.html",
            "busco": "https://busco.ezlab.org/busco_userguide.html",
        }
        yield (f"tool_{f.stem}", f"{tool} docs: {f.stem}",
               urls.get(tool, ""), TOOL_LICENSES.get(tool, ""), text)
    for f in sorted(td.glob("*.md")):
        tool = f.name.split("_")[0]
        text = f.read_text(errors="replace")
        if len(text.split()) < MIN_WORDS:
            continue
        yield (f"tool_{f.stem}", f"{tool} docs: {f.stem}", "",
               TOOL_LICENSES.get(tool, ""), text)
    wiki = td / "checkm_wiki"
    if wiki.exists():
        for f in sorted(wiki.glob("*.md")):
            text = f.read_text(errors="replace")
            if len(text.split()) < MIN_WORDS:
                continue
            yield (f"tool_checkm_wiki_{f.stem}", f"CheckM wiki: {f.stem}",
                   f"https://github.com/Ecogenomics/CheckM/wiki/{f.stem}",
                   TOOL_LICENSES["checkm"], text)


def iter_ebi():
    ed = SRC / "embl_ebi"
    base = ("https://www.ebi.ac.uk/training/online/courses/"
            "functional-genomics-ii-common-technologies-and-data-analysis-methods/"
            "next-generation-sequencing")
    for f in sorted(ed.glob("*.html")):
        text = html_to_text(f)
        if len(text.split()) < MIN_WORDS:
            continue
        slug = f.stem.replace("ngs_", "").replace("_", "/")
        url = base if slug == "index" else f"{base}/{slug}/"
        yield (f"ebi_{f.stem}", f"EMBL-EBI NGS: {f.stem}", url, CC_BY, text)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_docs, n_chunks = 0, 0
    per_source = {}
    with OUT.open("w") as out:
        for it in (iter_gtn(), iter_tool_docs(), iter_ebi()):
            for doc_id, title, url, lic, text in it:
                chunks = chunk_text(text)
                n_docs += 1
                per_source[doc_id] = len(chunks)
                for i, c in enumerate(chunks):
                    rec = {
                        "chunk_id": f"{doc_id}#{i}",
                        "doc_id": doc_id,
                        "title": title,
                        "url": url,
                        "license": lic,
                        "text": c,
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_chunks += 1
    total_words = sum(len(json.loads(l)["text"].split()) for l in OUT.open())
    print(f"Documents: {n_docs}")
    print(f"Chunks:    {n_chunks}  (avg {total_words/max(n_chunks,1):.0f} words)")
    print(f"Output:    {OUT}")
    print("\nTop docs by chunk count:")
    for doc, c in sorted(per_source.items(), key=lambda x: -x[1])[:15]:
        print(f"  {c:3d}  {doc}")


if __name__ == "__main__":
    sys.exit(main())
