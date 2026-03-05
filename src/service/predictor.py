from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd

from src.feature_extraction.extract_features_v1 import extract_features_from_sequence

VALID_NTS = set("ATGC")


@dataclass
class PredictionResult:
    sequence: str
    prediction: float
    features: Dict[str, Any]


class EfficiencyPredictor:
    """
    Loads a trained sklearn model once and exposes predict(sequence).
    """

    def __init__(self, model_path: str | Path, feature_columns: list[str]):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at: {self.model_path}")

        self.model = joblib.load(self.model_path)
        self.feature_columns = feature_columns

    def validate_sequence(self, seq: str) -> str:
        if seq is None:
            raise ValueError("Sequence is required.")

        seq = seq.strip().upper()

        if len(seq) != 20:
            raise ValueError("Sequence must be exactly 20 nucleotides long.")

        if not set(seq).issubset(VALID_NTS):
            raise ValueError("Sequence must contain only A, T, G, C.")

        return seq

    def predict(self, seq: str) -> PredictionResult:
        seq = self.validate_sequence(seq)
        features = extract_features_from_sequence(seq)

        X = pd.DataFrame(
            [[features.get(col, 0.0) for col in self.feature_columns]],
            columns=self.feature_columns,
        )

        pred = float(self.model.predict(X)[0])

        return PredictionResult(sequence=seq, prediction=pred, features=features)