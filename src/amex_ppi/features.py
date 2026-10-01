"""Customer-level feature engineering from monthly statements.

Three nested feature sets let us test whether behavioral *trajectories* add
signal beyond what a single snapshot or static aggregates already carry:

* ``snapshot``   – the most recent statement only (the "each month in isolation" baseline)
* ``aggregate``  – snapshot + mean/std/min/max over the full history
* ``trajectory`` – aggregate + trend features (slope, first→last change,
  recent-vs-history shift, month-over-month volatility, mean % change)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CATEGORICAL_COLS, DATE_COL, ID_COL, TARGET_COL

FEATURE_SETS = ("snapshot", "aggregate", "trajectory")
PCT_CHANGE_CLIP = 10.0


def numeric_cols(df: pd.DataFrame) -> list[str]:
    skip = {ID_COL, DATE_COL, TARGET_COL, *CATEGORICAL_COLS}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def truncate_history(df: pd.DataFrame, drop_last: int) -> pd.DataFrame:
    """Drop each customer's last ``drop_last`` statements.

    The AMEX label is "default within 18 months after the *last* statement", so
    training on truncated histories measures how far ahead the signal reaches.
    Customers left with no statements are removed.
    """
    if drop_last <= 0:
        return df
    pos_from_end = df.groupby(ID_COL, sort=False).cumcount(ascending=False)
    return df[pos_from_end >= drop_last]


def _slope(values: pd.DataFrame, t: pd.Series, ids: pd.Series) -> pd.DataFrame:
    """Per-customer OLS slope of each column on statement index, ignoring NaNs."""
    mask = values.notna().astype("float32")
    x = values.fillna(0)
    tm = mask.mul(t, axis=0)
    g = lambda frame: frame.groupby(ids, sort=False).sum()  # noqa: E731
    n, st, sx = g(mask), g(tm), g(x)
    stt, stx = g(tm.mul(t, axis=0)), g(x.mul(t, axis=0))
    denom = n * stt - st**2
    return ((n * stx - st * sx) / denom.where(denom > 0)).astype("float32")


def customer_features(df: pd.DataFrame, feature_set: str = "trajectory") -> tuple[pd.DataFrame, pd.Series]:
    """Aggregate statements to one row per customer. Returns ``(X, y)`` indexed by customer."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}")

    df = df.sort_values([ID_COL, DATE_COL])
    num = numeric_cols(df)
    cats = [c for c in CATEGORICAL_COLS if c in df.columns]
    g = df.groupby(ID_COL, sort=False)

    blocks = [g[num].last().add_suffix("_last"), g.size().rename("n_statements").to_frame()]

    # Categoricals: one-hot of the latest value, plus how often it changed.
    if cats:
        last_cat = g[cats].last()
        blocks.append(pd.get_dummies(last_cat.astype("string"), dummy_na=True, dtype="float32"))

    if feature_set in ("aggregate", "trajectory"):
        blocks += [
            g[num].mean().add_suffix("_mean"),
            g[num].std().add_suffix("_std"),
            g[num].min().add_suffix("_min"),
            g[num].max().add_suffix("_max"),
        ]
        if cats:
            blocks.append(g[cats].nunique().add_suffix("_nunique"))

    if feature_set == "trajectory":
        ids = df[ID_COL]
        t = g.cumcount().astype("float32")
        values = df[num]
        diffs = g[num].diff()
        prev = g[num].shift()
        # Guard against divide-by-zero: pct_change on a 0 balance would give ±inf.
        pct = (diffs / prev.abs().where(prev.abs() > 1e-6)).clip(-PCT_CHANGE_CLIP, PCT_CHANGE_CLIP)

        recent = df.groupby(ID_COL, sort=False).tail(3).groupby(ID_COL, sort=False)[num].mean()
        blocks += [
            _slope(values, t, ids).add_suffix("_slope"),
            (g[num].last() - g[num].first()).add_suffix("_delta"),
            (recent - g[num].mean()).add_suffix("_recent_shift"),
            recent.add_suffix("_3m_avg"),
            diffs.groupby(ids, sort=False).std().add_suffix("_diff_std"),
            pct.groupby(ids, sort=False).mean().add_suffix("_pct_change_mean"),
        ]

    X = pd.concat(blocks, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).astype("float32")
    X.columns = [str(c) for c in X.columns]
    y = g[TARGET_COL].first().astype("int8")
    return X, y
