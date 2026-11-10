"""Load and split the Breast Cancer Wisconsin (Diagnostic) dataset.

The dataset ships with scikit-learn, so no download is needed. It contains 569
fine-needle-aspirate samples described by 30 numeric features computed from
digitised images of cell nuclei (radius, texture, perimeter, area, ...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from bcd.config import RANDOM_STATE, TEST_SIZE


def to_snake_case(name: str) -> str:
    """Turn a scikit-learn feature name like 'mean radius' into 'mean_radius'."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


FEATURE_NAMES: list[str] = [to_snake_case(n) for n in load_breast_cancer().feature_names]


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Return features X and target y, where y = 1 means malignant.

    scikit-learn encodes malignant as 0 and benign as 1. We flip it so the
    class we care about detecting is the positive class, which makes recall
    and precision read naturally.
    """
    raw = load_breast_cancer(as_frame=True)
    X = raw.data.copy()
    X.columns = FEATURE_NAMES
    y = (1 - raw.target).rename("malignant")
    return X, y


@dataclass(frozen=True)
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Split:
    """Stratified train/test split so both sets keep the same class balance."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return Split(X_train, X_test, y_train, y_test)


def describe(X: pd.DataFrame, y: pd.Series) -> dict:
    """Small summary used in the README and the training report."""
    counts = y.value_counts().to_dict()
    return {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_malignant": int(counts.get(1, 0)),
        "n_benign": int(counts.get(0, 0)),
        "missing_values": int(X.isna().sum().sum()),
    }
