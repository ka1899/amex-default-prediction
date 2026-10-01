"""Evaluation metrics, including the official AMEX competition metric."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def _weighted_gini(y: np.ndarray, w: np.ndarray) -> float:
    random = np.cumsum(w / w.sum())
    lorentz = np.cumsum(y * w) / (y * w).sum()
    return float(((lorentz - random) * w).sum())


def amex_metric(y_true, y_pred) -> float:
    """AMEX M = 0.5 * (normalized Gini + default capture rate in the top 4%).

    Negatives are weighted 20x to undo the competition's subsampling of non-defaulters.
    """
    y_true = np.asarray(y_true, dtype=float)
    order = np.argsort(-np.asarray(y_pred, dtype=float), kind="mergesort")
    y = y_true[order]
    w = np.where(y == 0, 20.0, 1.0)

    top = np.cumsum(w) <= 0.04 * w.sum()
    capture = y[top].sum() / y.sum()

    y_best = np.sort(y_true)[::-1]
    gini = _weighted_gini(y, w) / _weighted_gini(y_best, np.where(y_best == 0, 20.0, 1.0))
    return 0.5 * (gini + capture)


def evaluate(y_true, y_pred) -> dict[str, float]:
    auc = roc_auc_score(y_true, y_pred)
    p = np.clip(y_pred, 1e-6, 1 - 1e-6)
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "amex_m": amex_metric(y_true, y_pred),
        "brier": brier_score_loss(y_true, p),
        "log_loss": log_loss(y_true, p),
    }


def calibration_table(y_true, y_pred, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs. observed default rate per predicted-probability decile."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    return df.groupby("bin").agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
