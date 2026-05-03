from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

from src.feature_extraction.extract_features_v1 import FEATURE_COLUMNS

DATA_PATH = Path("data/processed/v1_features.csv")


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    return pd.read_csv(path)


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df[FEATURE_COLUMNS]
    y = df["efficiency"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_test, preds),
    }


def save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved model to: {path}")


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best RF params: {grid_search.best_params_}")
    return grid_search.best_estimator_


def tune_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingRegressor:
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
    }
    gb = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(gb, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best GB params: {grid_search.best_params_}")
    return grid_search.best_estimator_
