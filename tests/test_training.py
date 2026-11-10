import json

from bcd.models import get_candidates


def test_every_candidate_is_a_pipeline_with_a_grid():
    candidates = get_candidates()
    assert len(candidates) == 5
    for c in candidates:
        assert c.param_grid, f"{c.name} has no hyperparameters to tune"
        assert all(key.startswith("clf__") for key in c.param_grid)


def test_training_saves_model_and_reports(trained):
    assert trained["model_path"].exists()
    reports = trained["reports_dir"]
    assert (reports / "metrics.json").exists()
    assert (reports / "model_comparison.csv").exists()
    for fig in ("roc_curves", "confusion_matrix", "feature_importance", "class_balance"):
        assert (reports / "figures" / f"{fig}.png").exists()

    metrics = json.loads((reports / "metrics.json").read_text())
    assert metrics["best_model"] in {c.name for c in get_candidates()}


def test_best_model_is_good_enough(trained):
    # Even the small CI grids should comfortably clear these bars.
    m = trained["summary"]["test_metrics"]
    assert m["roc_auc"] > 0.95
    assert m["recall"] > 0.85
    assert m["accuracy"] > 0.90
