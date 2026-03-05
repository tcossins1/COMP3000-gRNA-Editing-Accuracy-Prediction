from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "ridge_regression_v2.joblib"
MODEL_NAME = "ridge_regression_v2"