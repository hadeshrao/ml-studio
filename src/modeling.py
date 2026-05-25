"""
Model training and evaluation.

The fitted model is stored on project.model; a Step is recorded in project.steps
(stage="modeling") so the kedro exporter can generate the corresponding node.
The STEP_REGISTRY is kept for structural consistency; for kedro export it maps
algorithm keys to their constructor defaults.

No Streamlit imports — these functions are importable in the generated kedro project.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

STEP_REGISTRY: dict[str, Callable[..., pd.DataFrame]] = {}


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    model: Any
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray]   # predict_proba output; None for regression/clustering
    metrics: dict
    cv_scores: np.ndarray          # empty array for clustering
    feature_names: list[str]
    task_type: str


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(algorithm_key: str, hyperparams: dict, task_type: str) -> Any:
    """Instantiate an unfitted sklearn-compatible estimator from a key + param dict."""
    p = {k: v for k, v in hyperparams.items() if v is not None}

    if algorithm_key == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C=p.get("C", 1.0),
            penalty=p.get("penalty", "l2"),
            max_iter=p.get("max_iter", 1000),
            solver=p.get("solver", "saga"),
            random_state=p.get("random_state", 42),
            n_jobs=-1,
        )

    if algorithm_key == "random_forest":
        if task_type == "classification":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=p.get("n_estimators", 100),
                max_depth=p.get("max_depth"),
                min_samples_split=p.get("min_samples_split", 2),
                max_features=p.get("max_features", "sqrt"),
                random_state=p.get("random_state", 42),
                n_jobs=-1,
            )
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=p.get("n_estimators", 100),
            max_depth=p.get("max_depth"),
            min_samples_split=p.get("min_samples_split", 2),
            max_features=p.get("max_features", 1.0),
            random_state=p.get("random_state", 42),
            n_jobs=-1,
        )

    if algorithm_key == "xgboost":
        if task_type == "classification":
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=p.get("n_estimators", 100),
                max_depth=p.get("max_depth", 6),
                learning_rate=p.get("learning_rate", 0.1),
                subsample=p.get("subsample", 1.0),
                colsample_bytree=p.get("colsample_bytree", 1.0),
                random_state=p.get("random_state", 42),
                eval_metric="logloss",
                verbosity=0,
            )
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=p.get("n_estimators", 100),
            max_depth=p.get("max_depth", 6),
            learning_rate=p.get("learning_rate", 0.1),
            subsample=p.get("subsample", 1.0),
            colsample_bytree=p.get("colsample_bytree", 1.0),
            random_state=p.get("random_state", 42),
            verbosity=0,
        )

    if algorithm_key == "lightgbm":
        if task_type == "classification":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=p.get("n_estimators", 100),
                num_leaves=p.get("num_leaves", 31),
                learning_rate=p.get("learning_rate", 0.1),
                min_child_samples=p.get("min_child_samples", 20),
                random_state=p.get("random_state", 42),
                verbose=-1,
                n_jobs=-1,
            )
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=p.get("n_estimators", 100),
            num_leaves=p.get("num_leaves", 31),
            learning_rate=p.get("learning_rate", 0.1),
            min_child_samples=p.get("min_child_samples", 20),
            random_state=p.get("random_state", 42),
            verbose=-1,
            n_jobs=-1,
        )

    if algorithm_key == "knn":
        if task_type == "classification":
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier(
                n_neighbors=p.get("n_neighbors", 5),
                weights=p.get("weights", "uniform"),
                metric=p.get("metric", "minkowski"),
                n_jobs=-1,
            )
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(
            n_neighbors=p.get("n_neighbors", 5),
            weights=p.get("weights", "uniform"),
            metric=p.get("metric", "minkowski"),
            n_jobs=-1,
        )

    if algorithm_key == "svm":
        from sklearn.svm import SVC, SVR
        if task_type == "classification":
            return SVC(
                C=p.get("C", 1.0),
                kernel=p.get("kernel", "rbf"),
                degree=p.get("degree", 3),
                probability=True,
                random_state=p.get("random_state", 42),
            )
        return SVR(C=p.get("C", 1.0), kernel=p.get("kernel", "rbf"), degree=p.get("degree", 3))

    if algorithm_key == "linear_regression":
        from sklearn.linear_model import LinearRegression
        return LinearRegression(fit_intercept=p.get("fit_intercept", True), n_jobs=-1)

    if algorithm_key == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=p.get("alpha", 1.0))

    if algorithm_key == "lasso":
        from sklearn.linear_model import Lasso
        return Lasso(alpha=p.get("alpha", 1.0), max_iter=p.get("max_iter", 1000))

    if algorithm_key == "elastic_net":
        from sklearn.linear_model import ElasticNet
        return ElasticNet(
            alpha=p.get("alpha", 1.0),
            l1_ratio=p.get("l1_ratio", 0.5),
            max_iter=p.get("max_iter", 1000),
        )

    if algorithm_key == "kmeans":
        from sklearn.cluster import KMeans
        return KMeans(
            n_clusters=p.get("n_clusters", 5),
            n_init=p.get("n_init", 10),
            max_iter=p.get("max_iter", 300),
            random_state=p.get("random_state", 42),
        )

    if algorithm_key == "dbscan":
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=p.get("eps", 0.5), min_samples=p.get("min_samples", 5), n_jobs=-1)

    if algorithm_key == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        return AgglomerativeClustering(
            n_clusters=p.get("n_clusters", 5), linkage=p.get("linkage", "ward")
        )

    raise ValueError(f"Unknown algorithm key: '{algorithm_key}'")


# ── Training ──────────────────────────────────────────────────────────────────

def train_supervised(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    algorithm_key: str,
    hyperparams: dict,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    task_type: str = "classification",
) -> TrainResult:
    from sklearn.model_selection import (
        KFold,
        StratifiedKFold,
        cross_val_score,
        train_test_split,
    )

    X = df[features].copy()
    y = df[target].copy()

    # Drop rows where any feature or the target is NaN
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    if len(X) == 0:
        raise ValueError("No rows remain after dropping NaN. Add imputation steps first.")

    # Train / test split
    try:
        stratify = y if task_type == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    model = build_model(algorithm_key, hyperparams, task_type)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_prob: Optional[np.ndarray] = None
    if task_type == "classification" and hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)

    # Cross-validation
    if task_type == "classification":
        min_class = int(y_train.value_counts().min())
        n_splits = min(cv_folds, min_class)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scoring = "accuracy"
    else:
        n_splits = min(cv_folds, len(X_train))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scoring = "r2"

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    metrics = _supervised_metrics(y_test, y_pred, y_prob, task_type)
    metrics.update(
        {
            "algorithm": algorithm_key,
            "task_type": task_type,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "cv_scoring": scoring,
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
        }
    )

    return TrainResult(
        model=model,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        metrics=metrics,
        cv_scores=cv_scores,
        feature_names=features,
        task_type=task_type,
    )


def train_unsupervised(
    df: pd.DataFrame,
    features: list[str],
    algorithm_key: str,
    hyperparams: dict,
) -> TrainResult:
    from sklearn.metrics import silhouette_score

    X = df[features].dropna().reset_index(drop=True)
    model = build_model(algorithm_key, hyperparams, task_type="clustering")
    labels = model.fit_predict(X)

    unique_valid = np.unique(labels[labels >= 0])
    n_clusters = len(unique_valid)
    n_noise = int((labels == -1).sum())

    metrics: dict = {
        "algorithm": algorithm_key,
        "task_type": "clustering",
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_samples": len(X),
    }
    if n_clusters > 1:
        try:
            metrics["silhouette"] = float(
                silhouette_score(X, labels, sample_size=min(5000, len(X)))
            )
        except Exception:
            pass
    if hasattr(model, "inertia_"):
        metrics["inertia"] = float(model.inertia_)

    return TrainResult(
        model=model,
        X_test=X,
        y_test=pd.Series(labels, name="cluster"),
        y_pred=labels,
        y_prob=None,
        metrics=metrics,
        cv_scores=np.array([]),
        feature_names=features,
        task_type="clustering",
    )


# ── Metrics helpers ───────────────────────────────────────────────────────────

def _supervised_metrics(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    task_type: str,
) -> dict:
    if task_type == "classification":
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        m = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(
                precision_score(y_test, y_pred, average="weighted", zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, y_pred, average="weighted", zero_division=0)
            ),
            "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        }
        n_classes = len(np.unique(y_test))
        if y_prob is not None:
            try:
                if n_classes == 2:
                    m["roc_auc"] = float(roc_auc_score(y_test, y_prob[:, 1]))
                else:
                    m["roc_auc"] = float(
                        roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
                    )
            except Exception:
                pass
        return m

    # regression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    m = {
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }
    nonzero = y_test.values != 0
    if nonzero.any():
        m["mape"] = float(
            np.mean(np.abs((y_test.values[nonzero] - y_pred[nonzero]) / y_test.values[nonzero]))
            * 100
        )
    return m


# ── Feature importance ────────────────────────────────────────────────────────

def compute_feature_importance(model: Any, feature_names: list[str]) -> Optional[pd.DataFrame]:
    """Returns DataFrame(feature, importance) sorted descending, or None."""
    imp: Optional[np.ndarray] = None

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 2:
            imp = np.abs(coef).mean(axis=0)
        else:
            imp = np.abs(coef)

    if imp is None or len(imp) != len(feature_names):
        return None

    return (
        pd.DataFrame({"feature": feature_names, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
