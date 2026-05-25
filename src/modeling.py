"""
Model training and evaluation.

Training a model is recorded as a single Step under stage="modeling" with params
capturing the algorithm name and hyperparameters. The fitted model object is stored
directly on project.model (not in the steps list) since it is not serialisable as JSON.

STEP_REGISTRY here maps algorithm keys to factory functions that return unfitted
sklearn-compatible estimators: (df, **params) -> fitted estimator.

No Streamlit imports — these functions must be importable in the generated kedro project.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

STEP_REGISTRY: dict[str, Callable[..., pd.DataFrame]] = {}

# Example pattern:
# from sklearn.ensemble import RandomForestClassifier
# def train_random_forest(df: pd.DataFrame, target: str, n_estimators: int = 100, **kwargs):
#     X = df.drop(columns=[target])
#     y = df[target]
#     model = RandomForestClassifier(n_estimators=n_estimators, **kwargs)
#     model.fit(X, y)
#     return model
# STEP_REGISTRY["random_forest_classifier"] = train_random_forest
