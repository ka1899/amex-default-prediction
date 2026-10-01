"""Cost-benefit of proactively contacting high-risk customers.

Every flagged customer costs ``cost``; only flagged customers who would really
have defaulted can be saved, and only a ``success_rate`` share of those are.
The ROI is driven almost entirely by these assumptions, so always report it
alongside :func:`sensitivity` and the break-even success rate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InterventionAssumptions:
    cost: float = 200.0
    success_rate: float = 0.30
    claim_amount: float = 5_000.0


def economics(p, y_true, threshold: float, a: InterventionAssumptions = InterventionAssumptions()) -> dict:
    p, y = np.asarray(p, dtype=float), np.asarray(y_true, dtype=float)
    flagged = p >= threshold
    n_flagged = int(flagged.sum())
    tp = int((flagged & (y == 1)).sum())
    cost = n_flagged * a.cost
    savings = tp * a.success_rate * a.claim_amount
    net = savings - cost
    return {
        "threshold": threshold,
        "flagged": n_flagged,
        "flagged_pct": n_flagged / len(p),
        "true_positives": tp,
        "precision": tp / n_flagged if n_flagged else np.nan,
        "recall": tp / max(int(y.sum()), 1),
        "prevented_defaults": tp * a.success_rate,
        "program_cost": cost,
        "claims_avoided": savings,
        "net_savings": net,
        "roi": net / cost if cost else np.nan,
        # Success rate at which the program just pays for itself.
        "breakeven_success_rate": cost / (tp * a.claim_amount) if tp else np.nan,
    }


def threshold_curve(p, y_true, a: InterventionAssumptions = InterventionAssumptions(),
                    grid=np.round(np.arange(0.02, 0.99, 0.01), 2)) -> pd.DataFrame:
    return pd.DataFrame([economics(p, y_true, t, a) for t in grid])


def best_threshold(p, y_true, a: InterventionAssumptions = InterventionAssumptions()) -> float:
    """Threshold maximizing expected net savings (choose on validation data)."""
    curve = threshold_curve(p, y_true, a)
    return float(curve.loc[curve["net_savings"].idxmax(), "threshold"])


def sensitivity(p, y_true, threshold: float, base: InterventionAssumptions = InterventionAssumptions(),
                success_rates=(0.05, 0.10, 0.20, 0.30, 0.40),
                costs=(100, 200, 400, 800)) -> pd.DataFrame:
    """Net savings ($) at a fixed threshold across success-rate × cost scenarios."""
    rows = {}
    for s in success_rates:
        rows[f"{s:.0%}"] = {
            f"${c}": economics(p, y_true, threshold, replace(base, success_rate=s, cost=c))["net_savings"]
            for c in costs
        }
    out = pd.DataFrame(rows).T
    out.index.name = "success rate"
    out.columns.name = "cost / contact"
    return out
