"""Train and save a V1 linear regression model"""

from __future__ import annotations

from pathlib import Path

from sklearn.linear_model import LinearRegression
from src.evaluation.evaluate_models import evaluate_model
from src.training.training_utils import load_dataset, save_model, split_dataset

MODEL_PATH = Path("models/linear_regression_v1.joblib")


def train_linear_regression() -> None:
    # train on the default V1 feature dataset and evaluate
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df)

    model = LinearRegression()
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    save_model(model, MODEL_PATH)

    print("Linear Regression evaluation:")
    print(f"  MAE : {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R^2 : {metrics['r2']:.4f}")


def main() -> None:
    train_linear_regression()


if __name__ == "__main__":
    main()