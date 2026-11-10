"""Load the saved model and make predictions.

Can be used from Python or from the command line:
    python -m bcd.predict --sample 0     # predict one row of the dataset
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from bcd.config import MODEL_PATH, NEGATIVE_LABEL, POSITIVE_LABEL


class ModelNotFoundError(FileNotFoundError):
    pass


@lru_cache(maxsize=4)
def load_bundle(path: str | Path = MODEL_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(f"No trained model at {path}. Run `python -m bcd.train` first.")
    return joblib.load(path)


def predict(rows: list[dict], path: str | Path = MODEL_PATH) -> list[dict]:
    """Predict a list of samples, each a dict of the 30 feature values."""
    bundle = load_bundle(str(path))
    features = bundle["feature_names"]
    missing = [f for f in features if any(f not in row for row in rows)]
    if missing:
        raise ValueError(f"Missing features: {sorted(set(missing))}")

    X = pd.DataFrame(rows)[features]
    probs = bundle["model"].predict_proba(X)[:, 1]
    return [
        {
            "prediction": POSITIVE_LABEL if p >= 0.5 else NEGATIVE_LABEL,
            "probability_malignant": round(float(p), 4),
        }
        for p in probs
    ]


def main() -> None:
    from bcd.data import load_dataset

    parser = argparse.ArgumentParser(description="Predict one sample from the dataset.")
    parser.add_argument("--sample", type=int, default=0, help="row index (0-568)")
    args = parser.parse_args()

    X, y = load_dataset()
    row = X.iloc[args.sample].to_dict()
    result = predict([row])[0]
    actual = POSITIVE_LABEL if y.iloc[args.sample] == 1 else NEGATIVE_LABEL
    print(
        f"Sample {args.sample}: predicted {result['prediction']} "
        f"(p = {result['probability_malignant']:.3f}), actual {actual}"
    )


if __name__ == "__main__":
    main()
