"""Figures for the analysis. Each function returns a matplotlib Figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

BLUE, ORANGE, GREEN, RED, GREY = "#2a6fdb", "#e8833a", "#2e9e6a", "#d64545", "#8a8f98"


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def roc_curves(y_test, preds: dict[str, np.ndarray], aucs: dict[str, float]):
    fig, ax = plt.subplots(figsize=(6, 5))
    for (name, p), c in zip(preds.items(), [GREY, ORANGE, BLUE]):
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {aucs[name]:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#bbb", lw=1)
    ax.legend(loc="lower right", frameon=False)
    _style(ax, "ROC curves — held-out test set", "False positive rate", "True positive rate")
    return fig


def feature_ablation(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.barh(table.index, table["auc"], color=[GREY, ORANGE, BLUE])
    for b, v in zip(bars, table["auc"]):
        ax.text(v + 0.001, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center")
    lo = table["auc"].min() - 0.02
    ax.set_xlim(lo, min(1, table["auc"].max() + 0.01))
    ax.invert_yaxis()
    _style(ax, "Does trajectory information help? (LightGBM, test AUC)", "AUC")
    return fig


def horizon(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(table.index, table["auc"], marker="o", color=BLUE, lw=2)
    for k, v in table["auc"].items():
        ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xticks(table.index)
    _style(ax, "Early-warning horizon", "Most recent statements withheld (months)", "Test AUC")
    return fig


def calibration(raw: pd.DataFrame, calibrated: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="#bbb", lw=1, label="Perfect calibration")
    ax.plot(raw["predicted"], raw["observed"], marker="o", color=ORANGE, label="Uncalibrated")
    ax.plot(calibrated["predicted"], calibrated["observed"], marker="o", color=BLUE, label="Isotonic-calibrated")
    ax.legend(frameon=False)
    _style(ax, "Reliability diagram (test deciles)", "Mean predicted default probability", "Observed default rate")
    return fig


def loss_ratio_by_band(table: pd.DataFrame, target: float):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(table))
    w = 0.38
    ax.bar(x - w / 2, table["loss_ratio_flat"], w, color=GREY, label="Flat premium")
    ax.bar(x + w / 2, table["loss_ratio_risk"], w, color=BLUE, label="Risk-based premium")
    ax.axhline(target, color=RED, ls="--", lw=1.5, label=f"Target loss ratio ({target:.0%})")
    ax.set_xticks(x, table.index)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(frameon=False)
    _style(ax, "Experience loss ratio by risk quintile (test set)", "Predicted risk quintile", "Claims / premium")
    return fig


def threshold_curve(curve: pd.DataFrame, chosen: float):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(curve["threshold"], curve["net_savings"] / 1e3, color=BLUE, lw=2)
    ax.axvline(chosen, color=RED, ls="--", lw=1.5, label=f"Chosen on validation: {chosen:.2f}")
    ax.axhline(0, color="#999", lw=1)
    ax.legend(frameon=False)
    _style(ax, "Intervention net savings vs. threshold (test set)", "Risk-score threshold", "Net savings ($k)")
    return fig


def sensitivity_heatmap(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    vals = table.to_numpy() / 1e3
    lim = np.abs(vals).max()
    im = ax.imshow(vals, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:,.0f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Net savings ($k)")
    ax.set_title("Net savings under alternative assumptions", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Cost per contacted customer")
    ax.set_ylabel("Share of true defaults prevented")
    return fig


def shap_importance(mean_abs: pd.Series, top: int = 20):
    s = mean_abs.sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    colors = [BLUE if any(k in n for k in ("_slope", "_delta", "_recent_shift", "_diff_std", "_pct_change", "_3m_avg"))
              else GREY for n in s.index]
    ax.barh(s.index, s.values, color=colors)
    ax.tick_params(axis="y", labelsize=8)
    _style(ax, f"Top {top} features by mean |SHAP| (blue = trajectory)", "Mean |SHAP value| (log-odds)")
    return fig


def payment_trajectory(traj: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for (label, col), c in zip([("Non-default", 0), ("Default", 1)], [BLUE, RED]):
        ax.plot(traj.index, traj[col], marker="o", color=c, lw=2, label=label)
    ax.legend(frameon=False)
    _style(ax, "Mean payment ratio P_2 by statement month", "Months before last statement", "Mean P_2")
    ax.invert_xaxis()
    return fig
