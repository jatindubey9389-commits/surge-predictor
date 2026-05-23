"""Generate confusion matrix and feature importance charts, saved to reports/."""

import pathlib
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from train import FEATURES, load_and_prepare

REPORTS_DIR = pathlib.Path(__file__).parent.parent / "reports"
MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "surge_model.pkl"


def _load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def plot_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Save a confusion matrix PNG to reports/confusion_matrix.png."""
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Surge", "Surge"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    out = REPORTS_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


def plot_feature_importance(model, feature_names: list[str]) -> None:
    """Save a horizontal bar chart of XGBoost feature importances to reports/."""
    importances = model.feature_importances_
    indices = np.argsort(importances)
    sorted_names = [feature_names[i] for i in indices]
    sorted_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(sorted_names, sorted_vals, color="steelblue")
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    out = REPORTS_DIR / "feature_importance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


def plot_precision_recall(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Save a precision-recall curve to reports/precision_recall.png.

    More informative than accuracy for imbalanced surge/no-surge classes;
    average precision (area under the PR curve) is the headline metric.
    """
    proba = model.predict_proba(X_test)[:, 1]
    ap = average_precision_score(y_test, proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(
        y_test, proba, ax=ax, name=f"AP = {ap:.3f}"
    )
    ax.set_title("Precision–Recall Curve")
    fig.tight_layout()
    out = REPORTS_DIR / "precision_recall.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


def plot_roc_curve(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Save a ROC curve PNG to reports/roc_curve.png.

    AUC-ROC complements the PR curve: the latter is more informative under
    class imbalance, while AUC-ROC gives an imbalance-agnostic view of
    ranking quality.
    """
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name=f"AUC = {auc:.3f}")
    ax.set_title("ROC Curve")
    fig.tight_layout()
    out = REPORTS_DIR / "roc_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


def evaluate() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model = _load_model()

    X, y = load_and_prepare()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    plot_confusion_matrix(model, X_test, y_test)
    plot_feature_importance(model, list(X_test.columns))
    plot_precision_recall(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)


if __name__ == "__main__":
    evaluate()
