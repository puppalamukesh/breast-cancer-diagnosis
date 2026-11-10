import pytest
from fastapi.testclient import TestClient

from bcd.data import load_dataset


@pytest.fixture(scope="module")
def client(trained):
    import bcd.api

    return TestClient(bcd.api.app)


@pytest.fixture(scope="module")
def samples():
    X, y = load_dataset()
    malignant = X[y == 1].iloc[0].to_dict()
    benign = X[y == 0].iloc[0].to_dict()
    return malignant, benign


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info(client):
    info = client.get("/model-info").json()
    assert info["n_features"] == 30
    assert 0.9 < info["test_metrics"]["roc_auc"] <= 1.0


def test_predict_known_cases(client, samples):
    malignant, benign = samples
    r1 = client.post("/predict", json=malignant)
    r2 = client.post("/predict", json=benign)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["prediction"] == "malignant"
    assert r2.json()["prediction"] == "benign"
    assert 0 <= r1.json()["probability_malignant"] <= 1


def test_batch_predict(client, samples):
    r = client.post("/predict/batch", json={"samples": list(samples)})
    assert r.status_code == 200
    assert [p["prediction"] for p in r.json()["predictions"]] == ["malignant", "benign"]


def test_rejects_missing_feature(client, samples):
    bad = dict(samples[0])
    bad.pop("mean_radius")
    assert client.post("/predict", json=bad).status_code == 422


def test_rejects_negative_value(client, samples):
    bad = dict(samples[0], mean_area=-1.0)
    assert client.post("/predict", json=bad).status_code == 422


def test_rejects_empty_batch(client):
    assert client.post("/predict/batch", json={"samples": []}).status_code == 422
