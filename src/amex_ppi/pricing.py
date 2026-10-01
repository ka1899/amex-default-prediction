"""Risk-based premium calculation for a Payment Protection Insurance product.

Gross premium follows the standard loading formula

    gross = pure_premium / (1 - expense_ratio - profit_margin)

so expenses and profit are *shares of the premium*. The *target* loss ratio
is therefore ``1 - expense_ratio - profit_margin`` by construction. It is an
input, not a result. The meaningful check is the **experience** loss ratio:
actual claims divided by premiums charged, which only lands on target if the
model's probabilities are calibrated.

Note: PPI pays out on involuntary unemployment, disability, etc., not on default
itself. Default probability is used here as a proxy for claim probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PricingAssumptions:
    claim_amount: float = 5_000.0  # assumed avg. benefit paid per claim (AMEX features are anonymized)
    expense_ratio: float = 0.25
    profit_margin: float = 0.15

    @property
    def target_loss_ratio(self) -> float:
        return 1 - self.expense_ratio - self.profit_margin


def gross_premium(p, a: PricingAssumptions = PricingAssumptions()) -> np.ndarray:
    return np.asarray(p, dtype=float) * a.claim_amount / a.target_loss_ratio


def experience_loss_ratio(premium, y_true, a: PricingAssumptions = PricingAssumptions()) -> float:
    return float((np.asarray(y_true, dtype=float) * a.claim_amount).sum() / np.asarray(premium).sum())


def pricing_table(p, y_true, flat_rate_p: float, a: PricingAssumptions = PricingAssumptions(),
                  n_bands: int = 5) -> pd.DataFrame:
    """Compare risk-based vs. flat pricing by predicted-risk quintile.

    ``flat_rate_p`` is the portfolio default rate the flat premium is set on
    (estimate it on training data, not on the customers being priced).
    """
    df = pd.DataFrame({"p": np.asarray(p, dtype=float), "y": np.asarray(y_true, dtype=float)})
    df["risk_premium"] = gross_premium(df["p"], a)
    df["flat_premium"] = gross_premium(flat_rate_p, a)
    df["claims"] = df["y"] * a.claim_amount
    labels = ["Very Low", "Low", "Medium", "High", "Very High"][:n_bands]
    df["band"] = pd.qcut(df["p"].rank(method="first"), n_bands, labels=labels)

    t = df.groupby("band", observed=True).agg(
        customers=("y", "size"),
        predicted_rate=("p", "mean"),
        actual_rate=("y", "mean"),
        risk_premium=("risk_premium", "mean"),
        flat_premium=("flat_premium", "mean"),
        claims=("claims", "sum"),
        risk_prem_total=("risk_premium", "sum"),
        flat_prem_total=("flat_premium", "sum"),
    )
    t["loss_ratio_risk"] = t["claims"] / t["risk_prem_total"]
    t["loss_ratio_flat"] = t["claims"] / t["flat_prem_total"]
    return t.drop(columns=["claims", "risk_prem_total", "flat_prem_total"])
