"""Test-set metrics and the figures saved to reports/figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed, works on servers and in CI

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Colours: one fixed order for the models so every chart matches.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def evaluate_on_test(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "f2": float(fbeta_score(y_test, y_pred, beta=2)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=13, color=TEXT, pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.xaxis.label.set_color(TEXT_MUTED)
    ax.yaxis.label.set_color(TEXT_MUTED)


def _save(fig, path: Path) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_class_balance(y, path: Path) -> None:
    counts = [int((y == 0).sum()), int((y == 1).sum())]
    labels = ["Benign", "Malignant"]
    fig, ax = plt.subplots(figsize=(6, 2.4))
    bars = ax.barh(labels, counts, color=SERIES[:2], height=0.5)
    for bar, n in zip(bars, counts, strict=True):
        share = n / sum(counts)
        ax.text(
            bar.get_width() + 5,
            bar.get_y() + bar.get_height() / 2,
            f"{n}  ({share:.0%})",
            va="center",
            fontsize=9,
            color=TEXT,
        )
    ax.set_xlim(0, max(counts) * 1.25)
    ax.set_xlabel("Number of samples")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax, "Class balance: 569 samples")
    _save(fig, path)


def plot_roc_curves(fitted: dict, best_name: str, X_test, y_test, path: Path) -> None:
    # All models are close to perfect, so the full 0-1 square hides the
    # differences. Zoom into the top-left corner where the curves separate,
    # and draw the selected model last and thicker so it stays visible.
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    colours = dict(zip(fitted, SERIES, strict=False))
    draw_order = [n for n in fitted if n != best_name] + [best_name]
    lines = {}
    for name in draw_order:
        prob = fitted[name].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        is_best = name == best_name
        label = f"{name} (AUC {auc:.3f})" + ("  - selected" if is_best else "")
        (lines[name],) = ax.plot(
            fpr,
            tpr,
            color=colours[name],
            linewidth=3 if is_best else 1.6,
            drawstyle="steps-post",
            label=label,
        )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_xlim(-0.005, 0.3)
    ax.set_ylim(0.7, 1.01)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    # Legend in the usual model order, not the drawing order.
    leg = ax.legend(handles=[lines[n] for n in fitted], loc="lower right", fontsize=8.5, frameon=False)
    for text in leg.get_texts():
        text.set_color(TEXT)
    _style(ax, "ROC curves on the test set (top-left corner)")
    _save(fig, path)


def plot_confusion_matrix(model, name: str, X_test, y_test, path: Path) -> None:
    cm = confusion_matrix(y_test, model.predict(X_test))
    labels = ["Benign", "Malignant"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap=cmap)
    for i in range(2):
        for j in range(2):
            dark = cm[i, j] > cm.max() * 0.55
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center", fontsize=16, color="white" if dark else TEXT
            )
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    _style(ax, f"Confusion matrix: {name}")
    for spine in ax.spines.values():
        spine.set_visible(False)
    _save(fig, path)


def plot_feature_importance(model, name: str, X_test, y_test, path: Path, top: int = 10) -> None:
    result = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=42)
    order = np.argsort(result.importances_mean)[::-1][:top][::-1]
    names = [X_test.columns[i].replace("_", " ") for i in order]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.barh(
        names,
        result.importances_mean[order],
        xerr=result.importances_std[order],
        color=SERIES[0],
        height=0.6,
        error_kw={"ecolor": TEXT_MUTED, "linewidth": 1},
    )
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax, f"Top {top} features: {name}")
    _save(fig, path)


def plot_model_comparison(fitted: dict, X_test, y_test, path: Path) -> None:
    names = list(fitted)
    recall = [recall_score(y_test, fitted[n].predict(X_test)) for n in names]
    precision = [precision_score(y_test, fitted[n].predict(X_test)) for n in names]
    y_pos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.barh(y_pos + 0.18, recall, height=0.34, color=SERIES[0], label="Recall (malignant)")
    ax.barh(y_pos - 0.18, precision, height=0.34, color=SERIES[1], label="Precision (malignant)")
    for y, r, p in zip(y_pos, recall, precision, strict=True):
        ax.text(r + 0.005, y + 0.18, f"{r:.2f}", va="center", fontsize=8, color=TEXT)
        ax.text(p + 0.005, y - 0.18, f"{p:.2f}", va="center", fontsize=8, color=TEXT)
    ax.set_yticks(y_pos, names)
    ax.set_xlim(0.8, 1.04)
    ax.set_xlabel("Score on the test set (axis starts at 0.8)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8.5, frameon=False)
    for text in leg.get_texts():
        text.set_color(TEXT)
    _style(ax, "Recall vs precision by model")
    ax.set_title("Recall vs precision by model", loc="left", fontsize=13, color=TEXT, pad=30)
    _save(fig, path)


def make_figures(fitted: dict, best_name: str, split, X, y, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    best = fitted[best_name]
    plot_class_balance(y, out_dir / "class_balance.png")
    plot_roc_curves(fitted, best_name, split.X_test, split.y_test, out_dir / "roc_curves.png")
    plot_confusion_matrix(best, best_name, split.X_test, split.y_test, out_dir / "confusion_matrix.png")
    plot_feature_importance(best, best_name, split.X_test, split.y_test, out_dir / "feature_importance.png")
    plot_model_comparison(fitted, split.X_test, split.y_test, out_dir / "model_comparison.png")
