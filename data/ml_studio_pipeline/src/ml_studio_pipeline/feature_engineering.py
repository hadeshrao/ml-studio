"""
Feature engineering transformations.

Every function in STEP_REGISTRY takes (df: pd.DataFrame, **params) and returns a
transformed DataFrame. Steps are recorded in project.steps under
stage="feature_engineering" and replayed by state.current_df(). The kedro
exporter generates one node per step.

Naming convention for new columns:
  - transforms  →  {col}_log1p, {col}_sqrt, {col}_power
  - bins        →  {col}_bin
  - interactions→  {col_a}_x_{col_b}

Scalers replace column values in-place (fitted on the full df passed at replay
time; the kedro export wraps them in a proper train/test-safe node).

No Streamlit imports — these functions are imported in the generated kedro project.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

STEP_REGISTRY: dict[str, Callable[..., pd.DataFrame]] = {}


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_onehot(df: pd.DataFrame, columns: list, drop_first: bool = False) -> pd.DataFrame:
    return pd.get_dummies(df, columns=[c for c in columns if c in df.columns],
                          drop_first=drop_first, dtype=int)


STEP_REGISTRY["encode_onehot"] = encode_onehot


def encode_label(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes
    return df


STEP_REGISTRY["encode_label"] = encode_label


# ── Scaling ───────────────────────────────────────────────────────────────────

def scale_standard(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    from sklearn.preprocessing import StandardScaler
    df = df.copy()
    valid = [c for c in columns if c in df.columns]
    df[valid] = StandardScaler().fit_transform(df[valid].astype(float))
    return df


STEP_REGISTRY["scale_standard"] = scale_standard


def scale_minmax(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    from sklearn.preprocessing import MinMaxScaler
    df = df.copy()
    valid = [c for c in columns if c in df.columns]
    df[valid] = MinMaxScaler().fit_transform(df[valid].astype(float))
    return df


STEP_REGISTRY["scale_minmax"] = scale_minmax


def scale_robust(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    from sklearn.preprocessing import RobustScaler
    df = df.copy()
    valid = [c for c in columns if c in df.columns]
    df[valid] = RobustScaler().fit_transform(df[valid].astype(float))
    return df


STEP_REGISTRY["scale_robust"] = scale_robust


# ── Numeric transforms ────────────────────────────────────────────────────────

def transform_log1p(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[f"{col}_log1p"] = np.log1p(df[col].clip(lower=0))
    return df


STEP_REGISTRY["transform_log1p"] = transform_log1p


def transform_sqrt(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[f"{col}_sqrt"] = np.sqrt(df[col].clip(lower=0))
    return df


STEP_REGISTRY["transform_sqrt"] = transform_sqrt


def transform_power(df: pd.DataFrame, columns: list, method: str = "yeo-johnson") -> pd.DataFrame:
    from sklearn.preprocessing import PowerTransformer
    df = df.copy()
    valid = [c for c in columns if c in df.columns]
    new_cols = [f"{c}_power" for c in valid]
    df[new_cols] = PowerTransformer(method=method).fit_transform(df[valid].astype(float))
    return df


STEP_REGISTRY["transform_power"] = transform_power


# ── Feature creation ──────────────────────────────────────────────────────────

def bin_numeric(df: pd.DataFrame, column: str, n_bins: int = 5,
                strategy: str = "quantile") -> pd.DataFrame:
    df = df.copy()
    if column not in df.columns:
        return df
    out = f"{column}_bin"
    if strategy == "quantile":
        df[out] = pd.qcut(df[column], q=n_bins, labels=False, duplicates="drop")
    else:
        df[out] = pd.cut(df[column], bins=n_bins, labels=False)
    return df


STEP_REGISTRY["bin_numeric"] = bin_numeric


def create_interaction(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    df = df.copy()
    if col_a in df.columns and col_b in df.columns:
        df[f"{col_a}_x_{col_b}"] = df[col_a] * df[col_b]
    return df


STEP_REGISTRY["create_interaction"] = create_interaction
