from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.training.training_utils import (
    evaluate_model,
    load_dataset,
    save_model,
    split_dataset,
    tune_gradient_boosting,
    tune_random_forest,
)

MODEL_DIR = Path("models")
MODEL_TYPE_MAP = {
    "linear": "linear_regression",
    "rf": "random_forest",
    "gb": "gradient_boosting",
}


def build_model(model_type: str) -> Any:
    if model_type == "linear":
        return LinearRegression()
    if model_type == "rf":
        return RandomForestRegressor(random_state=42)
    if model_type == "gb":
        return GradientBoostingRegressor(random_state=42)
    raise ValueError(f"Unsupported model type: {model_type}")


def default_model_name(model_type: str, data_path: Path) -> str:
    suffix = data_path.stem.replace("_features", "")
    return f"{MODEL_TYPE_MAP[model_type]}_{suffix}.joblib"


def train_model(model_type: str, data_path: Path, output_path: Path) -> dict[str, float]:
    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test = split_dataset(df)

    if model_type == "rf":
        model = tune_random_forest(X_train, y_train)
    elif model_type == "gb":
        model = tune_gradient_boosting(X_train, y_train)
    else:
        model = build_model(model_type)
        model.fit(X_train, y_train)

    if model_type in {"rf", "gb"}:
        # For tuned models, training already happened in tune_* functions.
        pass
    else:
        model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    save_model(model, output_path)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a regression model for gRNA efficiency prediction")
    parser.add_argument(
        "--model-type",
        choices=["linear", "rf", "gb"],
        default="linear",
        help="Type of model to train (linear, rf, gb)",
    )
    parser.add_argument(
        "--data-path",
        default="data/processed/v1_features.csv",
        help="Path to the processed feature dataset",
    )
    parser.add_argument(
        "--output-path",
        help="Path to save the trained model. Defaults to models/<model_type>_<dataset>.joblib",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    output_path = Path(args.output_path) if args.output_path else MODEL_DIR / default_model_name(args.model_type, data_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = train_model(args.model_type, data_path, output_path)
    print(f"Trained model saved to: {output_path}")
    print("Evaluation metrics:")
    print(f"  MAE : {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R^2 : {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
