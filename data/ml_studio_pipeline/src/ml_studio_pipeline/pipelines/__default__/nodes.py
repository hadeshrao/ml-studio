
"""Auto-generated model training node for the ML Studio pipeline."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def train_model(df: pd.DataFrame) -> object:
    """Train Random Forest and return the fitted estimator.

    Hyperparameters were recorded from the ML Studio training session.
    To retune, edit conf/base/parameters.yml and re-run `kedro run`.
    """
    feature_cols: list[str] = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()

    
    y = df["median_house_value"].copy()
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]
    

    model = RandomForestRegressor(
        
        random_state=42,
        
        n_estimators=100,
        
        max_depth=null,
        
        min_samples_split=2,
        
        max_features="sqrt",
        
    )

    
    model.fit(X, y)
    

    return model
