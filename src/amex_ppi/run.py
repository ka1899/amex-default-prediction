"""End-to-end analysis. ``python -m amex_ppi.run --data data/amex_sample.parquet``

Each step is a function so the notebook can run and narrate them one by one.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import intervention as iv
from . import models as M
from . import plots
from .data import DATE_COL, ID_COL, TARGET_COL, load_statements, make_synthetic
from .features import customer_features, truncate_history
from .metrics import calibration_table, evaluate
from .pricing import PricingAssumptions, experience_loss_ratio, gross_premium, pricing_table


@dataclass
class Config:
    data_path: str | None = "data/amex_sample.parquet"  # None -> synthetic demo data
    out_dir: str = "reports"
    seed: int = 42
    horizons: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    pricing: PricingAssumptions = field(default_factory=PricingAssumptions)
    intervention: iv.InterventionAssumptions = field(default_factory=iv.InterventionAssumptions)


def load(cfg: Config) -> pd.DataFrame:
    if cfg.data_path is None:
        print("WARNING: using synthetic demo data. Results are NOT meaningful.")
        return make_synthetic(seed=cfg.seed)
    return load_statements(cfg.data_path)


def payment_trajectory(statements: pd.DataFrame, col: str = "P_2") -> pd.DataFrame:
    months_before_last = statements.groupby(ID_COL).cumcount(ascending=False)
    return statements.groupby([months_before_last, TARGET_COL])[col].mean().unstack()


def compare_models(X, y, ids, seed=42):
    fitted = M.fit_models(X, y, ids, seed)
    rows, test_preds = [], {}
    for name, model in fitted.items():
        pv, pt = M.predict(model, X.loc[ids["val"]]), M.predict(model, X.loc[ids["test"]])
        test_preds[name] = pt
        rows.append({"model": name, "split": "validation", **evaluate(y.loc[ids["val"]], pv)})
        rows.append({"model": name, "split": "test", **evaluate(y.loc[ids["test"]], pt)})
    table = pd.DataFrame(rows).set_index(["model", "split"])
    # Select on validation only, so the test numbers stay unbiased.
    best = table.xs("validation", level="split")["auc"].idxmax()
    return fitted, table, test_preds, best


def feature_ablation(statements, y, ids, X_traj, seed=42) -> pd.DataFrame:
    rows = {}
    for fs in ("snapshot", "aggregate", "trajectory"):
        X = X_traj if fs == "trajectory" else customer_features(statements, fs)[0]
        m = M.fit_lightgbm(X.loc[ids["train"]], y.loc[ids["train"]], X.loc[ids["val"]], y.loc[ids["val"]], seed)
        rows[fs] = {"n_features": X.shape[1], **evaluate(y.loc[ids["test"]], M.predict(m, X.loc[ids["test"]]))}
    return pd.DataFrame(rows).T


def horizon_analysis(statements, ids, horizons, seed=42) -> pd.DataFrame:
    """Withhold the last k statements and re-fit: how far ahead does the signal reach?

    Restricted to customers with a full 13 statements so every horizon scores the
    same population.
    """
    counts = statements.groupby(ID_COL).size()
    full = counts.index[counts == 13]
    base = statements[statements[ID_COL].isin(full)]
    split = {k: v.intersection(full) for k, v in ids.items()}

    rows = {}
    for k in horizons:
        X, y = customer_features(truncate_history(base, k), "trajectory")
        m = M.fit_lightgbm(X.loc[split["train"]], y.loc[split["train"]], X.loc[split["val"]], y.loc[split["val"]], seed)
        rows[k] = {"statements_used": 13 - k, **evaluate(y.loc[split["test"]], M.predict(m, X.loc[split["test"]]))}
        print(f"  horizon {k}: AUC {rows[k]['auc']:.4f}", flush=True)
    out = pd.DataFrame(rows).T
    out.index.name = "months_withheld"
    return out


def shap_values(model, X: pd.DataFrame, n: int = 3000, seed: int = 42) -> pd.Series:
    import shap

    sample = X.sample(min(n, len(X)), random_state=seed)
    sv = shap.TreeExplainer(model).shap_values(sample)
    if isinstance(sv, list):
        sv = sv[1]
    return pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)


def run(cfg: Config) -> dict:
    out = Path(cfg.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(out / "figures" / f"{name}.png", dpi=150, bbox_inches="tight")

    statements = load(cfg)
    print(f"Loaded {len(statements):,} statements / {statements[ID_COL].nunique():,} customers")
    save(plots.payment_trajectory(payment_trajectory(statements)), "payment_trajectory")

    print("Building trajectory features...")
    X, y = customer_features(statements, "trajectory")
    ids = M.split_ids(y, cfg.seed)
    yv, yt = y.loc[ids["val"]], y.loc[ids["test"]]
    print(f"  {X.shape[1]} features; default rate {y.mean():.1%}")

    print("Comparing models...")
    fitted, model_table, test_preds, best = compare_models(X, y, ids, cfg.seed)
    model_table.to_csv(out / "tables" / "model_comparison.csv")
    test_auc = model_table.xs("test", level="split")["auc"].to_dict()
    save(plots.roc_curves(yt, test_preds, test_auc), "roc_curves")
    print(model_table.round(4).to_string())

    print("Feature-set ablation...")
    ablation = feature_ablation(statements, y, ids, X, cfg.seed)
    ablation.to_csv(out / "tables" / "feature_ablation.csv")
    save(plots.feature_ablation(ablation), "feature_ablation")

    print("Horizon analysis...")
    horizon = horizon_analysis(statements, ids, cfg.horizons, cfg.seed)
    horizon.to_csv(out / "tables" / "horizon.csv")
    save(plots.horizon(horizon), "horizon")

    print("Calibrating best model on validation...")
    best_model = fitted[best]
    calibrated = M.calibrate(best_model, X.loc[ids["val"]], yv)
    pv = M.predict(calibrated, X.loc[ids["val"]])
    pt = M.predict(calibrated, X.loc[ids["test"]])
    pt_raw = test_preds[best]
    cal_metrics = {"uncalibrated": evaluate(yt, pt_raw), "calibrated": evaluate(yt, pt)}
    save(plots.calibration(calibration_table(yt, pt_raw), calibration_table(yt, pt)), "calibration")

    print("Pricing...")
    a = cfg.pricing
    flat_p = float(y.loc[ids["train"]].mean())
    price = pricing_table(pt, yt, flat_p, a)
    price.to_csv(out / "tables" / "pricing_by_band.csv")
    save(plots.loss_ratio_by_band(price, a.target_loss_ratio), "loss_ratio_by_band")
    pricing_summary = {
        "target_loss_ratio": a.target_loss_ratio,
        "experience_loss_ratio_calibrated": experience_loss_ratio(gross_premium(pt, a), yt, a),
        "experience_loss_ratio_uncalibrated": experience_loss_ratio(gross_premium(pt_raw, a), yt, a),
        "experience_loss_ratio_flat": experience_loss_ratio(gross_premium(np.full(len(yt), flat_p), a), yt, a),
        "avg_premium_risk_based": float(gross_premium(pt, a).mean()),
        "avg_premium_flat": float(gross_premium(flat_p, a)),
    }

    print("Intervention economics...")
    ia = cfg.intervention
    threshold = iv.best_threshold(pv, yv, ia)
    econ = iv.economics(pt, yt, threshold, ia)
    curve = iv.threshold_curve(pt, yt, ia)
    sens = iv.sensitivity(pt, yt, threshold, ia)
    sens.to_csv(out / "tables" / "intervention_sensitivity.csv")
    save(plots.threshold_curve(curve, threshold), "threshold_curve")
    save(plots.sensitivity_heatmap(sens), "intervention_sensitivity")

    shap_top = None
    if best == "LightGBM":
        print("SHAP...")
        mean_abs = shap_values(best_model, X.loc[ids["test"]], seed=cfg.seed)
        mean_abs.head(50).to_csv(out / "tables" / "shap_top50.csv")
        save(plots.shap_importance(mean_abs), "shap_importance")
        shap_top = mean_abs.head(10).round(4).to_dict()

    results = {
        "data": "synthetic" if cfg.data_path is None else str(cfg.data_path),
        "n_customers": int(len(y)),
        "n_features": int(X.shape[1]),
        "default_rate": float(y.mean()),
        "split_sizes": {k: int(len(v)) for k, v in ids.items()},
        "best_model": best,
        "model_comparison": {f"{m}|{s}": r for (m, s), r in model_table.round(4).to_dict("index").items()},
        "feature_ablation": ablation.round(4).to_dict("index"),
        "horizon": {int(k): v for k, v in horizon.round(4).to_dict("index").items()},
        "calibration": cal_metrics,
        "pricing_assumptions": asdict(a),
        "pricing": pricing_summary,
        "intervention_assumptions": asdict(ia),
        "intervention": econ,
        "shap_top10": shap_top,
    }
    (out / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"Done. Results in {out}/")
    return results


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/amex_sample.parquet", help="Sample parquet from amex_ppi.data")
    p.add_argument("--synthetic", action="store_true", help="Run on synthetic demo data instead")
    p.add_argument("--out", default="reports")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    run(Config(data_path=None if a.synthetic else a.data, out_dir=a.out, seed=a.seed))


if __name__ == "__main__":
    _main()
