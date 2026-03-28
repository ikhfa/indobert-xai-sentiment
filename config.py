"""
Global configuration for IndoBERT XAI Sentiment Analysis project.
All hyperparameters and paths are defined here — no hardcoded values elsewhere.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_DIR = OUTPUTS_DIR / "results"

# Create output dirs at import time
for _d in [OUTPUTS_DIR, CHECKPOINTS_DIR, PLOTS_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME: str = "indobenchmark/indobert-base-p1"
NUM_LABELS: int = 3          # positive=2, neutral=1, negative=0
MAX_SEQ_LEN: int = 128

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 32
EVAL_BATCH_SIZE: int = 64
LEARNING_RATE: float = 2e-5
EPOCHS: int = 5
WARMUP_RATIO: float = 0.1
WEIGHT_DECAY: float = 0.01
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME: str = "indonlp/indonlu"
DATASET_SUBSET: str = "smsa"

LABEL_MAP: dict = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_NAMES: list = ["negative", "neutral", "positive"]

# ---------------------------------------------------------------------------
# XAI
# ---------------------------------------------------------------------------
LIME_NUM_FEATURES: int = 15
LIME_NUM_SAMPLES: int = 500
IG_STEPS: int = 50
ATTENTION_TOP_K: int = 10

# Faithfulness evaluation token masking ratios
AOPC_K_RATIOS: list = [0.01, 0.05, 0.10, 0.20]

# ---------------------------------------------------------------------------
# Device (auto-detect)
# ---------------------------------------------------------------------------
import torch  # noqa: E402

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
