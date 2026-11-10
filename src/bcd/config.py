"""Central configuration: paths, random seed and evaluation settings."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = Path(os.getenv("BCD_MODELS_DIR", PROJECT_ROOT / "models"))
REPORTS_DIR = Path(os.getenv("BCD_REPORTS_DIR", PROJECT_ROOT / "reports"))
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = Path(os.getenv("BCD_MODEL_PATH", MODELS_DIR / "model.joblib"))

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Positive class is "malignant" (1). In a screening setting a missed malignant
# case is far more costly than a false alarm, so models are tuned and selected
# on the F2-score, which weights recall twice as much as precision. Plain recall
# is not used on its own because a model that flags everyone as malignant
# would score a perfect 1.0. ROC-AUC breaks ties.
POSITIVE_LABEL = "malignant"
NEGATIVE_LABEL = "benign"
SELECTION_METRIC = "f2"
TIEBREAK_METRIC = "roc_auc"
