"""Candidate models and their hyperparameter search spaces.

Every candidate is a scikit-learn Pipeline, so scaling is fitted inside each
cross-validation fold and never leaks information from the validation data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from bcd.config import RANDOM_STATE


@dataclass
class Candidate:
    name: str
    pipeline: Pipeline
    param_grid: dict = field(default_factory=dict)


def get_candidates(random_state: int = RANDOM_STATE) -> list[Candidate]:
    return [
        Candidate(
            name="Logistic Regression",
            pipeline=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=5000, random_state=random_state)),
                ]
            ),
            param_grid={
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        Candidate(
            # SVC has no predict_proba of its own, so it is wrapped in a
            # calibrator (Platt scaling) to get probabilities for ROC-AUC.
            name="SVM (RBF)",
            pipeline=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", CalibratedClassifierCV(SVC(), method="sigmoid", ensemble=False)),
                ]
            ),
            param_grid={
                "clf__estimator__C": [0.1, 1.0, 10.0],
                "clf__estimator__gamma": ["scale", 0.01, 0.001],
                "clf__estimator__class_weight": [None, "balanced"],
            },
        ),
        Candidate(
            name="K-Nearest Neighbours",
            pipeline=Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
            param_grid={
                "clf__n_neighbors": [3, 5, 7, 11],
                "clf__weights": ["uniform", "distance"],
            },
        ),
        Candidate(
            name="Random Forest",
            pipeline=Pipeline([("clf", RandomForestClassifier(random_state=random_state))]),
            param_grid={
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 6],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        Candidate(
            name="Gradient Boosting",
            pipeline=Pipeline([("clf", GradientBoostingClassifier(random_state=random_state))]),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [2, 3],
            },
        ),
    ]
