"""Loading, sampling and (for tests/demos only) synthesizing AMEX-shaped data.

The raw Kaggle ``train_data.csv`` is ~16 GB, so :func:`build_sample` streams it
in chunks, keeps a random subset of customers and writes a compact float32
parquet file that the rest of the pipeline reads.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ID_COL = "customer_ID"
DATE_COL = "S_2"
TARGET_COL = "target"

# Categorical columns as documented by the competition hosts.
CATEGORICAL_COLS = [
    "B_30", "B_38", "D_114", "D_116", "D_117", "D_120",
    "D_126", "D_63", "D_64", "D_66", "D_68",
]


def build_sample(
    raw_dir: str | Path,
    out_path: str | Path,
    n_customers: int = 50_000,
    seed: int = 42,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Stream ``train_data.csv`` and keep every statement for a random customer subset."""
    raw_dir = Path(raw_dir).expanduser()
    labels = pd.read_csv(raw_dir / "train_labels.csv")

    rng = np.random.default_rng(seed)
    n = min(n_customers, len(labels))
    keep = set(rng.choice(labels[ID_COL].to_numpy(), size=n, replace=False))

    header = pd.read_csv(raw_dir / "train_data.csv", nrows=0).columns
    dtypes = {c: "float32" for c in header if c not in (ID_COL, DATE_COL, "D_63", "D_64")}
    dtypes.update({ID_COL: "string", DATE_COL: "string", "D_63": "string", "D_64": "string"})

    parts = []
    reader = pd.read_csv(raw_dir / "train_data.csv", dtype=dtypes, chunksize=chunksize)
    for i, chunk in enumerate(reader):
        parts.append(chunk[chunk[ID_COL].isin(keep)])
        if i % 4 == 0:
            print(f"  read {(i + 1) * chunksize:,} rows", flush=True)

    df = pd.concat(parts, ignore_index=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.merge(labels, on=ID_COL, how="left")
    df[TARGET_COL] = df[TARGET_COL].astype("int8")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} rows / {df[ID_COL].nunique():,} customers to {out_path}")
    return df


def load_statements(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)


def make_synthetic(n_customers: int = 3_000, seed: int = 0) -> pd.DataFrame:
    """AMEX-shaped toy data for tests and quick demos. **Not** evidence of anything.

    Default risk is driven by a noisy latent factor; deterioration starts at a
    random month (not a fixed one), statement counts vary, values go missing and
    two categorical columns are included so every pipeline branch is exercised.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cid in range(n_customers):
        latent = rng.normal()
        default = int(rng.random() < 1 / (1 + np.exp(-(latent * 1.5 - 1.1))))
        n_stmt = 13 if rng.random() < 0.85 else int(rng.integers(1, 13))
        onset = int(rng.integers(3, 13))
        base_pay = rng.uniform(0.3, 0.9)
        for m in range(13 - n_stmt, 13):
            drift = max(0, m - onset) * (0.03 if default else 0.0) + rng.normal(0, 0.01)
            rows.append({
                ID_COL: f"c{cid:06d}",
                DATE_COL: pd.Timestamp("2017-03-01") + pd.DateOffset(months=m),
                "P_2": base_pay - drift + 0.05 * (-latent) + rng.normal(0, 0.08),
                "B_1": max(0.0, 0.2 + drift * 1.5 + 0.05 * latent + rng.normal(0, 0.06)),
                "B_2": rng.uniform(0, 1),
                "D_39": float(rng.poisson(1 + 2 * drift * 10)),
                "S_3": np.nan if rng.random() < 0.2 else rng.normal(0.2, 0.05),
                "R_1": max(0.0, rng.normal(0.02 + drift, 0.02)),
                "D_63": rng.choice(["CO", "CR", "CL"], p=[0.75, 0.17, 0.08]),
                "B_30": float(rng.choice([0, 1, 2], p=[0.8, 0.15, 0.05])),
                TARGET_COL: default,
            })
    df = pd.DataFrame(rows)
    df[TARGET_COL] = df[TARGET_COL].astype("int8")
    return df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)


def _main() -> None:
    p = argparse.ArgumentParser(description="Build a customer-level sample of AMEX train data.")
    p.add_argument("--raw-dir", default="~/amex_data", help="Folder with train_data.csv and train_labels.csv")
    p.add_argument("--out", default="data/amex_sample.parquet")
    p.add_argument("--n-customers", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    build_sample(a.raw_dir, a.out, a.n_customers, a.seed)


if __name__ == "__main__":
    _main()
