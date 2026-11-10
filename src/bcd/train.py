"""Train, compare and save the best model.

Usage:
    python -m bcd.train            # full grid search (about a minute)
    python -m bcd.train --fast     # small grids, for CI and quick checks
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from bcd import __version__
from bcd.config import (
    CV_FOLDS,
    MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    SELECTION_METRIC,
    TIEBREAK_METRIC,
)
from bcd.data import FEATURE_NAMES, describe, load_dataset, split_dataset
from bcd.evaluate import evaluate_on_test, make_figures
from bcd.models import Candidate, get_candidates

SCORING = {
    "f2": make_scorer(fbeta_score, beta=2),
    "recall": "recall",
    "precision": "precision",
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
}


def shrink_grid(grid: dict) -> dict:
    """Keep only the first value of every parameter (used by --fast)."""
    return {key: values[:1] for key, values in grid.items()}


def search(candidate: Candidate, X, y, fast: bool) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = shrink_grid(candidate.param_grid) if fast else candidate.param_grid
    gs = GridSearchCV(
        candidate.pipeline,
        param_grid=grid,
        scoring=SCORING,
        refit=SELECTION_METRIC,
        cv=cv,
        n_jobs=-1,
    )
    gs.fit(X, y)
    return gs


def cv_summary(name: str, gs: GridSearchCV) -> dict:
    i = gs.best_index_
    row = {"model": name}
    for metric in SCORING:
        row[f"cv_{metric}"] = float(gs.cv_results_[f"mean_test_{metric}"][i])
        row[f"cv_{metric}_std"] = float(gs.cv_results_[f"std_test_{metric}"][i])
    row["best_params"] = {
        k.removeprefix("clf__").removeprefix("estimator__"): v for k, v in gs.best_params_.items()
    }
    return row


def run(fast: bool = False, model_path: Path = MODEL_PATH, reports_dir: Path = REPORTS_DIR) -> dict:
    started = time.perf_counter()
    X, y = load_dataset()
    split = split_dataset(X, y)

    results, fitted = [], {}
    for candidate in get_candidates():
        gs = search(candidate, split.X_train, split.y_train, fast)
        results.append(cv_summary(candidate.name, gs))
        fitted[candidate.name] = gs.best_estimator_
        print(
            f"  {candidate.name:<22} CV F2 {results[-1]['cv_f2']:.3f}  "
            f"recall {results[-1]['cv_recall']:.3f}  ROC-AUC {results[-1]['cv_roc_auc']:.3f}"
        )

    comparison = pd.DataFrame(results).sort_values(
        [f"cv_{SELECTION_METRIC}", f"cv_{TIEBREAK_METRIC}"], ascending=False
    )
    best_name = comparison.iloc[0]["model"]
    best_model = fitted[best_name]

    # The test set is only used here, after every model has been tuned.
    test_metrics = {}
    for name, model in fitted.items():
        test_metrics[name] = evaluate_on_test(model, split.X_test, split.y_test)
    for metric in ("f2", "recall", "precision", "roc_auc", "accuracy"):
        comparison[f"test_{metric}"] = [test_metrics[n][metric] for n in comparison["model"]]

    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)

    summary = {
        "project_version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "fast_mode": fast,
        "dataset": describe(X, y),
        "split": {"train": int(len(split.X_train)), "test": int(len(split.X_test))},
        "selection_metric": SELECTION_METRIC,
        "best_model": best_name,
        "best_params": comparison.iloc[0]["best_params"],
        "test_metrics": test_metrics[best_name],
        "all_models": {r["model"]: {**r, "test": test_metrics[r["model"]]} for r in results},
    }
    (reports_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))

    make_figures(fitted, best_name, split, X, y, reports_dir / "figures")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "feature_names": FEATURE_NAMES,
            "test_metrics": test_metrics[best_name],
            "trained_at": summary["trained_at"],
            "version": __version__,
        },
        model_path,
    )

    elapsed = time.perf_counter() - started
    print(f"\nBest model: {best_name}")
    m = test_metrics[best_name]
    print(
        f"Test set: recall {m['recall']:.3f}  precision {m['precision']:.3f}  "
        f"F2 {m['f2']:.3f}  ROC-AUC {m['roc_auc']:.3f}  accuracy {m['accuracy']:.3f}"
    )
    print(f"Saved model to {model_path} ({elapsed:.1f}s)")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare breast cancer classifiers.")
    parser.add_argument("--fast", action="store_true", help="use tiny grids (for CI)")
    args = parser.parse_args()
    print("Training candidate models with stratified 5-fold cross-validation...")
    run(fast=args.fast)


if __name__ == "__main__":
    main()
