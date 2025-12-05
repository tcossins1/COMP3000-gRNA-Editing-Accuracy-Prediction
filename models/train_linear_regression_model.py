# A basic linear regression model using the V1_features.csv from the V1 Azimuth dataset only
# Currently only using GC content as an extracted feature

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib


def load_dataset():
    # Load the processed feature dataset
    return pd.read_csv("data/processed/v1_features.csv")


def train_linear_model(df):
    # Train linear regression model using GC content
    X = df[["gc_content"]]   # features (currently only GC content)
    y = df["efficiency"]     # target variable

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

# TODO: add proper evaluation (MSE, R²) once more features are included


def save_model(model, path="models/linear_regression_v1.joblib"):
    # Save trained model to the 'models/' directory
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def main():
    df = load_dataset()
    model = train_linear_model(df)
    save_model(model)


if __name__ == "__main__":
    main()
