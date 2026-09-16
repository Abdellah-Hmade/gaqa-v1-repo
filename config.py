"""Configuration. Every value can be overridden via a `.env` file or an
environment variable of the same name (see `.env.example`). CLI flags override
these values. No external dotenv dependency required.
"""
import os
from pathlib import Path


def _load_dotenv(path: str | os.PathLike) -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_dotenv(Path(__file__).resolve().parent / ".env")
_load_dotenv(Path.cwd() / ".env")


def _get(name: str, default, cast=str):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return default


# --- dataset -------------------------------------------------------------
ZENODO_RECORD = _get("ZENODO_RECORD", "21762025")
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"
DATA_FILES = {
    "train":    "gaqa_v1_train.jsonl",
    "val":      "gaqa_v1_val.jsonl",
    "heldout":  "gaqa_v1_heldout_test.jsonl",
    "knowledge": "gaqa_v1_knowledge_test.jsonl",
}
DATA_DIR = _get("GAQA_DATA_DIR", None)  # optional local dir with gaqa_v1_*.jsonl

# --- models (Hugging Face) ------------------------------------------------
BITNET_MODEL = _get("BITNET_MODEL", "microsoft/bitnet-b1.58-2B-4T-bf16")
MISTRAL_MODEL = _get("MISTRAL_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
QWEN_MODEL = _get("QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = _get("HF_TOKEN", None)  # required for gated models (e.g. Mistral)

# Chat template used for evaluation (matches the paper's protocol / Llama-3
# special-token layout; the base BitNet tokenizer ships a different default).
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{{ message['content'] }}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] }}<|eot_id|>\n"
    "{% endif %}"
    "{% endfor %}"
)

# --- fine-tuning protocol -------------------------------------------------
SEED = _get("SEED", 42, int)
LORA_R = _get("LORA_R", 32, int)
LORA_ALPHA = _get("LORA_ALPHA", 64, int)
LORA_DROPOUT = _get("LORA_DROPOUT", 0.05, float)
NUM_EPOCHS = _get("NUM_EPOCHS", 3, int)
BATCH_SIZE = _get("BATCH_SIZE", 8, int)
GRAD_ACCUM = _get("GRAD_ACCUM", 1, int)
MAX_LENGTH = _get("MAX_LENGTH", 1024, int)
LEARNING_RATE = _get("LEARNING_RATE", 2e-4, float)
WARMUP_RATIO = _get("WARMUP_RATIO", 0.05, float)
WEIGHT_DECAY = _get("WEIGHT_DECAY", 0.01, float)
GRAD_CLIP = _get("GRAD_CLIP", 1.0, float)
MAX_TRAIN_SAMPLES = _get("MAX_TRAIN_SAMPLES", 0, int)  # 0 = all
MAX_STEPS = _get("MAX_STEPS", 0, int)                   # 0 = full epochs
EVAL_SAMPLES = _get("EVAL_SAMPLES", 256, int)
# Expected optimizer steps: NUM_EPOCHS x ceil(11043 / BATCH_SIZE) = 3 x 1381
import math as _math
EXPECTED_STEPS = _math.ceil(11043 / BATCH_SIZE) * NUM_EPOCHS

# --- evaluation / benchmark ----------------------------------------------
EVAL_BATCH_SIZE = _get("EVAL_BATCH_SIZE", 16, int)
DEVICE = _get("DEVICE", None)
if DEVICE is None:
    try:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        DEVICE = "cpu"

# --- outputs --------------------------------------------------------------
OUTPUT_DIR = _get("OUTPUT_DIR", "outputs/gaqa-v1-lora")
BENCHMARK_OUTPUT = _get("BENCHMARK_OUTPUT", "outputs/multi_model_comparison.csv")

# --- test mode (scripts/test.sh) ------------------------------------------
# smoke = fast, no GPU/no 2B download; full = venv -> data -> real-model eval
TEST_MODE = _get("TEST_MODE", "smoke").lower()

# --- data source (train.py / eval.py / benchmark.py) ----------------------
# zenodo : auto-download the released dataset from Zenodo (default)
# local  : read the released gaqa_v1_*.jsonl from GAQA_DATA_DIR
# rebuild: use the dataset built by data_pipeline (build_gaqa_dataset_v3.py)
DATA_SOURCE = _get("DATA_SOURCE", "zenodo").lower()
REBUILD_DATA_DIR = _get("REBUILD_DATA_DIR", "data_pipeline/data/processed_gaqa_v3")
REBUILD_HELDOUT_DIR = _get("REBUILD_HELDOUT_DIR", "data_pipeline/data/realval_v3_heldout")
REBUILD_KNOWLEDGE_DIR = _get("REBUILD_KNOWLEDGE_DIR", "data_pipeline/data/gaqa_knowledge")


def effective_data_dir(split: str | None = None) -> str | None:
    """Directory load_jsonl should read for a split, or None to use Zenodo."""
    if DATA_SOURCE == "local":
        if not DATA_DIR:
            raise SystemExit("DATA_SOURCE=local requires GAQA_DATA_DIR in .env")
        return DATA_DIR
    if DATA_SOURCE == "rebuild":
        return {"train": REBUILD_DATA_DIR, "val": REBUILD_DATA_DIR,
                "heldout": REBUILD_HELDOUT_DIR,
                "knowledge": REBUILD_KNOWLEDGE_DIR}[split or "train"]
    return None  # zenodo (auto-download)