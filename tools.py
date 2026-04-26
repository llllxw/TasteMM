import os
import random

import numpy as np
import torch


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data", "Multi")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_SAVE_DIR = os.path.join(ROOT_DIR, "models", "saved")
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
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def setup_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    ensure_dirs()
    print("Directories initialized.")
