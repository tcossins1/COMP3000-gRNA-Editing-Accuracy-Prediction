from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.feature_extraction.extract_features_v1 import FEATURE_COLUMNS

DATA_PATH = Path("data/processed/v1_features.csv")
MODEL_DIR = Path("models")
RESULTS_PATH = Path("data/processed/model_results.csv")
MODEL_PATHS = {
    "linear_regression_v1": MODEL_DIR / "linear_regression_v1.joblib",
    "random_forest_v1": MODEL_DIR / "random_forest_v1.joblib",
    "gradient_boosting_v1": MODEL_DIR / "gradient_boosting_v1.joblib",
}


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


def load_model(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def evaluate_all() -> pd.DataFrame:
    df = load_dataset()
    _, X_test, _, y_test = split_dataset(df)

    rows = []
    for model_name, model_path in MODEL_PATHS.items():
        model = load_model(model_path)
        metrics = evaluate_model(model, X_test, y_test)
        rows.append({"model": model_name, **metrics})

    df_results = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    df_results.to_csv(RESULTS_PATH, index=False)
    print(f"Saved evaluation results to: {RESULTS_PATH}")
    return df_results


def main() -> None:
    results = evaluate_all()
    print("\nEvaluation complete. Results:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
