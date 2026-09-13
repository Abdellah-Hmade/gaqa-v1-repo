"""Dataset loading (local dir or auto-download from Zenodo) and helpers."""
import json
import re
from pathlib import Path

import requests

import config

_CACHE = Path(__file__).resolve().parent / "data_cache"


def _download(filename: str, cache_dir: Path) -> Path:
    dest = cache_dir / filename
    if dest.exists():
        return dest
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"{config.ZENODO_BASE}/{filename}?download=1"
    print(f"Downloading {filename} from {url} ...")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    return dest


def _pick_path(split: str, data_dir: str) -> Path:
    """Find the file for a split in data_dir. Accepts both the released
    naming (gaqa_v1_*.jsonl) and the data_pipeline builder naming
    (train.jsonl / val.jsonl / test.jsonl)."""
    candidates = [config.DATA_FILES[split]]
    alt = {"train": "train.jsonl", "val": "val.jsonl",
           "heldout": "test.jsonl", "knowledge": "test.jsonl"}[split]
    candidates.append(alt)
    for name in candidates:
        p = Path(data_dir) / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{split}: none of {candidates} found in {data_dir}")


def load_jsonl(split: str, data_dir: str | None = None):
    """Return a list of dicts for one GAQA v1 split.

    split in {"train", "val", "heldout", "knowledge"}.
    data_dir: optional local directory; when None the source is chosen by
    config.DATA_SOURCE (zenodo=auto-download, local=GAQA_DATA_DIR,
    rebuild=data_pipeline builder output).
    """
    filename = config.DATA_FILES[split]
    if data_dir is None:
        data_dir = config.effective_data_dir(split)
    if data_dir is not None:
        path = _pick_path(split, data_dir)
    else:
        path = _download(filename, _CACHE)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gold_letter(output: str) -> str:
    """Extract the gold answer letter from output: 'The correct answer is D: ...'."""
    m = re.search(r"The correct answer is\s+([A-D])", output)
    if not m:
        raise ValueError(f"cannot parse gold letter from output: {output[:80]!r}")
    return m.group(1)


def acceptable_letter(row: dict) -> str | None:
    """Letter of the 'Acceptable quality assembly' option in the input (qc_diagnosis)."""
    m = re.search(r"([A-D])\s*\)\s*Acceptable quality", row["input"])
    return m.group(1) if m else None