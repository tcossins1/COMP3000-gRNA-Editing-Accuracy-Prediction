from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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
