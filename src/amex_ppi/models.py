"""Model definitions, data splitting and probability calibration."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def split_ids(y: pd.Series, seed: int = 42) -> dict[str, pd.Index]:
    """Stratified 60/20/20 train/validation/test split of customer IDs.

    Validation is used for model selection, early stopping, calibration and
    threshold choice; the test set is touched once, for final reporting.
    """
    idx = y.index.to_numpy(dtype=object)
    train, rest = train_test_split(idx, test_size=0.4, stratify=y.to_numpy(), random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=y.loc[rest].to_numpy(), random_state=seed)
    return {"train": pd.Index(train), "val": pd.Index(val), "test": pd.Index(test)}


def make_logistic(seed: int = 42):
    # Unscaled features make L2-regularized LR a straw-man baseline; scale first.
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.05, max_iter=2000, random_state=seed),
    )


def make_random_forest(seed: int = 42):
    # No class_weight: reweighting distorts probabilities, and we price off them.
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=20, max_features="sqrt", n_jobs=-1, random_state=seed
    )


def fit_lightgbm(X_tr, y_tr, X_val, y_val, seed: int = 42) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.3,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_X=(X_val,),
        eval_y=(y_val,),
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def fit_models(X: pd.DataFrame, y: pd.Series, ids: dict, seed: int = 42) -> dict:
    tr, va = ids["train"], ids["val"]
    models = {
        "Logistic Regression": make_logistic(seed).fit(X.loc[tr], y.loc[tr]),
        "Random Forest": make_random_forest(seed).fit(X.loc[tr], y.loc[tr]),
        "LightGBM": fit_lightgbm(X.loc[tr], y.loc[tr], X.loc[va], y.loc[va], seed),
    }
    return models


def predict(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def calibrate(model, X_val, y_val, method: str = "isotonic"):
    """Calibrate an already-fitted model on held-out validation data."""
    return CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(X_val, y_val)
