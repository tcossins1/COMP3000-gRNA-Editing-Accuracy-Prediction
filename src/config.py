import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = os.getenv("FORECAS9_MODEL", "gradient_boosting_v2")
MODEL_PATH = REPO_ROOT / "models" / f"{MODEL_NAME}.joblib"