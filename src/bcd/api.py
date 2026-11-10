"""REST API that serves the trained model with FastAPI.

Run locally:
    uvicorn bcd.api:app --reload
Then open http://127.0.0.1:8000/docs for the interactive Swagger page.
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from bcd import __version__
from bcd.config import MODEL_PATH
from bcd.data import FEATURE_NAMES, load_dataset
from bcd.predict import ModelNotFoundError, load_bundle, predict

# Build the request schema from the 30 feature names, so it can never drift
# from the training data. All features are measurements, so none is negative.
_example = load_dataset()[0].iloc[0].round(4).to_dict()

Sample = create_model(
    "Sample",
    **{name: (float, Field(..., ge=0, examples=[_example[name]])) for name in FEATURE_NAMES},
)


class Prediction(BaseModel):
    prediction: Literal["malignant", "benign"]
    probability_malignant: float = Field(..., ge=0, le=1)


class BatchRequest(BaseModel):
    samples: list[Sample] = Field(..., min_length=1, max_length=100)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


app = FastAPI(
    title="Breast Cancer Diagnosis API",
    version=__version__,
    description=(
        "Predicts whether a breast mass is malignant or benign from 30 features "
        "of a fine-needle aspirate. Educational project, not a medical device."
    ),
)


def _bundle() -> dict:
    try:
        return load_bundle(str(MODEL_PATH))
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    bundle = _bundle()
    return {
        "model": bundle["model_name"],
        "trained_at": bundle["trained_at"],
        "version": bundle["version"],
        "n_features": len(bundle["feature_names"]),
        "test_metrics": bundle["test_metrics"],
    }


@app.post("/predict", response_model=Prediction)
def predict_one(sample: Sample) -> dict:
    _bundle()
    return predict([sample.model_dump()], MODEL_PATH)[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest) -> dict:
    _bundle()
    rows = [s.model_dump() for s in request.samples]
    return {"predictions": predict(rows, MODEL_PATH)}
