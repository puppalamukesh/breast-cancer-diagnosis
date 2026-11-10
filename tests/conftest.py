"""Shared test fixtures.

A small model is trained once per test session into a temporary folder, so
the tests never depend on (or overwrite) the real files in models/ and reports/.
"""

import importlib

import pytest


@pytest.fixture(scope="session")
def trained(tmp_path_factory, monkeypatch_session):
    workdir = tmp_path_factory.mktemp("run")
    model_path = workdir / "model.joblib"
    monkeypatch_session.setenv("BCD_MODEL_PATH", str(model_path))
    monkeypatch_session.setenv("BCD_REPORTS_DIR", str(workdir / "reports"))

    import bcd.api
    import bcd.config
    import bcd.predict
    import bcd.train

    for module in (bcd.config, bcd.train, bcd.predict, bcd.api):
        importlib.reload(module)

    summary = bcd.train.run(fast=True, model_path=model_path, reports_dir=workdir / "reports")
    return {"summary": summary, "model_path": model_path, "reports_dir": workdir / "reports"}


@pytest.fixture(scope="session")
def monkeypatch_session():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()
