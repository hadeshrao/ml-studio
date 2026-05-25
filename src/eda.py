"""
Exploratory data analysis utilities.

Pure functions only — no Streamlit imports. The EDA page imports these for
computation; Plotly rendering lives in the page itself. Functions here are
read-only (they never mutate the DataFrame) so they are not registered as
pipeline steps, but the module keeps a STEP_REGISTRY for structural consistency.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

STEP_REGISTRY: dict[str, Callable[..., pd.DataFrame]] = {}


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype, non-null count, missing %, and cardinality."""
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "non_null": int(s.notna().sum()),
                "missing_pct": round(s.isna().mean() * 100, 1),
                "n_unique": int(s.nunique()),
            }
        )
    return pd.DataFrame(rows)


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Columns with at least one missing value, sorted by count descending."""
    counts = df.isna().sum()
    pct = (counts / len(df) * 100).round(2)
    result = pd.DataFrame(
        {"column": df.columns, "missing_count": counts.values, "missing_pct": pct.values}
    )
    return result[result["missing_count"] > 0].sort_values("missing_count", ascending=False)


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Correlation matrix for numeric columns only."""
    return df.select_dtypes(include="number").corr(method=method)
