import os
import random
import hashlib
import json

import numpy as np
import torch


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data", "Multi")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

EMBED_DIM = 256
GAT_DIM = 128
CONTRAST_DIM = 64
TEMPERATURE = 0.2
BATCH_SIZE = 32
PRETRAIN_EPOCHS = 100
FINETUNE_EPOCHS = 500
PATIENCE = 50


def ensure_dirs() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def setup_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def training_config_tag(config: dict) -> str:
    """Return a readable, stable tag without hashing machine-specific paths."""
    defaults = {
        "skip_pretrain": False,
        "graph_aux_weight": 0.2,
        "graph_warmup_epochs": 30,
        "graph_warmup_lr": 1e-4,
        "num_classes": 6,
    }
    relevant = {key: config[key] for key in defaults}
    if relevant == defaults:
        return "baseline"
    digest = hashlib.sha256(json.dumps(relevant, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"baseline_c{int(relevant['num_classes'])}_{digest}"


def training_run_name(fold: int, seed: int, config: dict) -> str:
    return f"fold{int(fold)}_seed{int(seed)}_{training_config_tag(config)}"


if __name__ == "__main__":
    ensure_dirs()
    print("Directories initialized.")
