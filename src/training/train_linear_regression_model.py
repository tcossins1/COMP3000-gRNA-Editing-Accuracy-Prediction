# Train a Ridge regression model using v1_features.csv (V1 Azimuth features)
# Uses multiple simple sequence-derived features (GC windows, poly-T, homopolymer length)

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Must match feature extraction + API inference order
FEATURE_COLUMNS = ["gc_content", "gc_1_10", "gc_11_20", "has_poly_t4", "max_homopolymer"]


def load_dataset():
    return pd.read_csv("data/processed/v1_features.csv")


def train_ridge_model(df):
    X = df[FEATURE_COLUMNS]
    y = df["efficiency"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Basic evaluation (prints to terminal)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    print("Ridge v2 evaluation:")
    print(f"  MAE : {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R^2 : {r2:.4f}")

    return model


def save_model(model, path="models/ridge_regression_v2.joblib"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved model to: {path}")


def main():
    df = load_dataset()
    model = train_ridge_model(df)
    save_model(model)


if __name__ == "__main__":
    main()