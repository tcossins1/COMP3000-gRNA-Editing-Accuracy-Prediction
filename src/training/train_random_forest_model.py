from __future__ import annotations

from pathlib import Path

from src.evaluation.evaluate_models import evaluate_model
from src.training.training_utils import load_dataset, save_model, split_dataset, tune_random_forest

MODEL_PATH = Path("models/random_forest_v1.joblib")


def train_random_forest() -> None:
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df)

    model = tune_random_forest(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    save_model(model, MODEL_PATH)

    print("Random Forest evaluation:")
    print(f"  MAE : {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R^2 : {metrics['r2']:.4f}")


def main() -> None:
    train_random_forest()


if __name__ == "__main__":
    main()
