from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor
from src.training.training_utils import evaluate_model, load_dataset, save_model, split_dataset

MODEL_PATH = Path("models/gradient_boosting_v1.joblib")


def train_gradient_boosting() -> None:
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    save_model(model, MODEL_PATH)

    print("Gradient Boosting evaluation:")
    print(f"  MAE : {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R^2 : {metrics['r2']:.4f}")


def main() -> None:
    train_gradient_boosting()


if __name__ == "__main__":
    main()
