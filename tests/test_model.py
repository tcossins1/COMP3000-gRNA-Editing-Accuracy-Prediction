"""Unit tests for trained models."""
import pytest
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from src.config import MODEL_PATH, MODEL_NAME
from src.feature_extraction.extract_features_v1 import FEATURE_COLUMNS


class TestModelArtifacts:
    """Test that trained models are properly saved and loadable."""

    def test_v2_gradient_boosting_model_exists(self):
        """Test that V2 gradient boosting model exists."""
        model_path = Path("models/gradient_boosting_v2.joblib")
        assert model_path.exists(), f"V2 GB model not found at {model_path}"

    def test_v2_linear_regression_model_exists(self):
        """Test that V2 linear regression model exists."""
        model_path = Path("models/linear_regression_v2.joblib")
        assert model_path.exists(), f"V2 LR model not found at {model_path}"

    def test_v2_random_forest_model_exists(self):
        """Test that V2 random forest model exists."""
        model_path = Path("models/random_forest_v2.joblib")
        assert model_path.exists(), f"V2 RF model not found at {model_path}"

    def test_default_model_path_valid(self):
        """Test that default model path from config is valid."""
        assert MODEL_PATH.exists(), f"Default model not found at {MODEL_PATH}"
        print(f"Default model: {MODEL_NAME} at {MODEL_PATH}")

    def test_model_can_be_loaded(self):
        """Test that the default model can be loaded."""
        model = joblib.load(str(MODEL_PATH))
        assert model is not None, "Model failed to load"
        assert hasattr(model, "predict"), "Model should have predict method"

    def test_model_has_expected_features(self):
        """Test that model has feature_names_in_ attribute."""
        model = joblib.load(str(MODEL_PATH))
        
        if hasattr(model, "feature_names_in_"):
            feature_names = model.feature_names_in_
            assert len(feature_names) == len(FEATURE_COLUMNS), \
                f"Model expects {len(feature_names)} features, but {len(FEATURE_COLUMNS)} are available"

    def test_model_prediction_output_type(self):
        """Test that model returns numpy array predictions."""
        model = joblib.load(str(MODEL_PATH))
        
        # Create dummy input matching feature count
        n_features = len(FEATURE_COLUMNS)
        X_dummy = np.random.randn(1, n_features)
        
        pred = model.predict(X_dummy)
        
        assert isinstance(pred, np.ndarray), "Prediction should be numpy array"
        assert pred.shape == (1,), "Prediction should have shape (1,)"

    def test_model_prediction_is_numeric(self):
        """Test that model predictions are numeric values."""
        model = joblib.load(str(MODEL_PATH))
        
        n_features = len(FEATURE_COLUMNS)
        X_dummy = np.random.randn(5, n_features)
        
        preds = model.predict(X_dummy)
        
        assert all(isinstance(p, (float, np.floating)) for p in preds), \
            "All predictions should be numeric"

    def test_model_consistency(self):
        """Test that model produces consistent predictions."""
        model = joblib.load(str(MODEL_PATH))
        
        n_features = len(FEATURE_COLUMNS)
        X_test = np.random.randn(10, n_features)
        
        pred1 = model.predict(X_test)
        pred2 = model.predict(X_test)
        
        np.testing.assert_array_equal(pred1, pred2, \
            "Model should produce consistent predictions")

    def test_v2_models_have_compatible_features(self):
        """Test that all V2 models use the same features."""
        models = [
            Path("models/gradient_boosting_v2.joblib"),
            Path("models/linear_regression_v2.joblib"),
            Path("models/random_forest_v2.joblib"),
        ]
        
        for model_path in models:
            if model_path.exists():
                model = joblib.load(str(model_path))
                if hasattr(model, "feature_names_in_"):
                    assert len(model.feature_names_in_) == len(FEATURE_COLUMNS), \
                        f"{model_path.name} has incompatible feature count"


class TestModelEvaluation:
    """Test model evaluation results."""

    def test_v2_model_results_exist(self):
        """Test that V2 model results CSV exists."""
        results_path = Path("data/processed/v2_model_results.csv")
        assert results_path.exists(), f"V2 model results not found at {results_path}"

    def test_v2_model_results_valid(self):
        """Test that V2 model results have valid format."""
        df = pd.read_csv(Path("data/processed/v2_model_results.csv"))
        
        assert "model" in df.columns, "Results should have 'model' column"
        assert "mae" in df.columns, "Results should have 'mae' column"
        assert "rmse" in df.columns, "Results should have 'rmse' column"
        assert "r2" in df.columns, "Results should have 'r2' column"
        
        assert len(df) > 0, "Results should have at least one row"

    def test_v2_model_metrics_reasonable(self):
        """Test that V2 model metrics are in reasonable ranges."""
        df = pd.read_csv(Path("data/processed/v2_model_results.csv"))
        
        # MAE should be positive
        assert (df["mae"] > 0).all(), "MAE should be positive"
        
        # RMSE should be positive and >= MAE
        assert (df["rmse"] > 0).all(), "RMSE should be positive"
        assert (df["rmse"] >= df["mae"]).all(), "RMSE should be >= MAE"
        
        # R² should be between -inf and 1, typically we'd expect it in [-1, 1]
        assert (df["r2"] <= 1).all(), "R² should not exceed 1"

    def test_combined_model_results_exist(self):
        """Test that combined model results CSV exists."""
        results_path = Path("data/processed/combined_model_results.csv")
        assert results_path.exists(), f"Combined model results not found at {results_path}"

    def test_combined_model_results_valid(self):
        """Test that combined model results have valid format."""
        df = pd.read_csv(Path("data/processed/combined_model_results.csv"))
        
        assert "model" in df.columns, "Results should have 'model' column"
        assert "mae" in df.columns, "Results should have 'mae' column"
        assert "rmse" in df.columns, "Results should have 'rmse' column"
        assert "r2" in df.columns, "Results should have 'r2' column"
        
        assert len(df) > 0, "Results should have at least one row"
