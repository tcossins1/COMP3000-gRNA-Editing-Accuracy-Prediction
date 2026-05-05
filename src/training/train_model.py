from __future__ import annotations

import argparse
from pathlib import Path

from src.training.training_utils import (
    default_model_name,
    save_model,
    train_model as train_model_shared,
)

MODEL_DIR = Path("models")


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

    model, metrics = train_model_shared(args.model_type, data_path)
    save_model(model, output_path)
    
    print(f"Trained model saved to: {output_path}")
    print("Evaluation metrics:")
    print(f"  MAE : {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R^2 : {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
