"""
Data preprocessing transformations.

Every function registered in STEP_REGISTRY is a pure function:
  (df: pd.DataFrame, **params) -> pd.DataFrame

Steps are recorded in project.steps under stage="preprocessing" and replayed
by state.current_df() in the same order they were added. The kedro exporter
generates one pipeline node per step from this registry.

No Streamlit imports — these functions are imported in the generated kedro project.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

STEP_REGISTRY: dict[str, Callable[..., pd.DataFrame]] = {}


def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return df.drop(columns=[c for c in columns if c in df.columns])


STEP_REGISTRY["drop_columns"] = drop_columns


def impute_mean(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
    return df


STEP_REGISTRY["impute_mean"] = impute_mean


def impute_median(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


STEP_REGISTRY["impute_median"] = impute_median


def impute_mode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])
    return df


STEP_REGISTRY["impute_mode"] = impute_mode


def impute_constant(df: pd.DataFrame, columns: list, value: str) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                fill: object = float(value)
            except ValueError:
                fill = value
        else:
            fill = value
        df[col] = df[col].fillna(fill)
    return df


STEP_REGISTRY["impute_constant"] = impute_constant


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


STEP_REGISTRY["remove_duplicates"] = remove_duplicates


def cast_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


STEP_REGISTRY["cast_numeric"] = cast_numeric
