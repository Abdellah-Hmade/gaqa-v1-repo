import os, ast
from pathlib import Path
from dotenv import load_dotenv
load_dotenv('.env')

def _bool(val):
    return str(val).lower() in ('true', '1', 'yes')

def _list(val):
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return [x.strip() for x in str(val).split(',') if x.strip()]

class Config:
    project_name: str = os.getenv('PROJECT_NAME', 'gaqa-bitnet-assembly')
    seed: int = int(os.getenv('SEED', '42'))
    output_dir: Path = Path(os.getenv('OUTPUT_DIR', './results'))
    hf_cache_dir: Path = Path(os.getenv('HF_CACHE_DIR', './models/hf_cache'))
    ntfy_topic: str = os.getenv('NTFY_TOPIC', 'gaqa-reference')
    bitnet_model: str = os.getenv('BITNET_MODEL', 'microsoft/bitnet-b1.58-2B-4T-bf16')
    bitnet_packed_adapter_dir: Path = Path(os.getenv('BITNET_PACKED_ADAPTER_DIR', './models/bitnet-packed-adapter'))
    bitnet_ste_adapter_dir: Path = Path(os.getenv('BITNET_STE_ADAPTER_DIR', './models/bitnet-ste-adapter'))
    bitnet_lora_r: int = int(os.getenv('BITNET_LORA_R', '8'))
    bitnet_lora_alpha: int = int(os.getenv('BITNET_LORA_ALPHA', '16'))
    bitnet_lora_dropout: float = float(os.getenv('BITNET_LORA_DROPOUT', '0.05'))
    bitnet_learning_rate: float = float(os.getenv('BITNET_LEARNING_RATE', '2e-4'))
    bitnet_num_epochs: int = int(os.getenv('BITNET_NUM_EPOCHS', '3'))
    bitnet_optimizer: str = os.getenv('BITNET_OPTIMIZER', 'adamw_torch')
    log_dir: Path = Path(os.getenv('LOG_DIR', './logs'))
    data_processed_dir: Path = Path(os.getenv('DATA_PROCESSED_DIR', './data/processed'))
    max_train_samples: int = int(os.getenv('MAX_TRAIN_SAMPLES', '20000'))
    max_eval_samples: int = int(os.getenv('MAX_EVAL_SAMPLES', '500'))
    max_length: int = int(os.getenv('MAX_LENGTH', '1024'))
    batch_size: int = int(os.getenv('BATCH_SIZE', '1'))
    grad_accum: int = int(os.getenv('GRAD_ACCUM', '4'))
    warmup_ratio: float = float(os.getenv('WARMUP_RATIO', '0.1'))
    scheduler: str = os.getenv('SCHEDULER', 'cosine')
    gradient_checkpointing: bool = _bool(os.getenv('GRADIENT_CHECKPOINTING', 'true'))
    dataloader_num_workers: int = int(os.getenv('DATALOADER_NUM_WORKERS', '4'))
    lora_target_modules: list = _list(os.getenv('LORA_TARGET_MODULES', '["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]'))
    eval_num_samples: int = int(os.getenv('EVAL_NUM_SAMPLES', '2000'))
    eval_batch_size: int = int(os.getenv('EVAL_BATCH_SIZE', '1'))
    eval_max_new_tokens: int = int(os.getenv('EVAL_MAX_NEW_TOKENS', '128'))
cfg = Config()
