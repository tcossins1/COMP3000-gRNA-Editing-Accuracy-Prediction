"""Unit tests for the prediction service."""
import pytest
from pathlib import Path
import joblib
from sklearn.linear_model import LinearRegression
import numpy as np

from src.service.predictor import EfficiencyPredictor
from src.feature_extraction.extract_features_v1 import FEATURE_COLUMNS


@pytest.fixture
def dummy_model(tmp_path):
    """Create a simple dummy model for testing."""
    X = np.random.randn(10, len(FEATURE_COLUMNS))
    y = np.random.randn(10)
    
    model = LinearRegression()
    model.fit(X, y)
    
    model_path = tmp_path / "test_model.joblib"
    joblib.dump(model, str(model_path))
    
    return model_path


class TestEfficiencyPredictor:
    """Test the EfficiencyPredictor service."""

    def test_predictor_initialization_valid_model(self, dummy_model):
        """Test predictor initialization with a valid model."""
        predictor = EfficiencyPredictor(dummy_model, feature_columns=FEATURE_COLUMNS)
        assert predictor.model is not None
        assert len(predictor.feature_columns) == len(FEATURE_COLUMNS)

    def test_predictor_initialization_missing_model(self):
        """Test predictor initialization fails with missing model."""
        with pytest.raises(FileNotFoundError):
            EfficiencyPredictor("nonexistent_model.joblib", feature_columns=FEATURE_COLUMNS)

    def test_sequence_validation_valid(self, dummy_model):
        """Test sequence validation for a valid 20-nt sequence."""
        predictor = EfficiencyPredictor(dummy_model, feature_columns=FEATURE_COLUMNS)
        seq = "ATGCGTAGCTAAGCTAGCAC"
        validated = predictor.validate_sequence(seq)
        assert validated == seq

    def test_sequence_validation_case_insensitive(self, dummy_model):
        """Test sequence validation converts to uppercase."""
        predictor = EfficiencyPredictor(dummy_model, feature_columns=FEATURE_COLUMNS)
        seq_lower = "atgcgtagctaagctagcac"
        validated = predictor.validate_sequence(seq_lower)
        assert validated == "ATGCGTAGCTAAGCTAGCAC"

    def test_sequence_validation_whitespace_stripped(self, dummy_model):
        """Test sequence validation strips whitespace."""
        predictor = EfficiencyPredictor(dummy_model, feature_columns=FEATURE_COLUMNS)
        seq_with_spaces = "  ATGCGTAGCTAAGCTAGCAC  "
        validated = predictor.validate_sequence(seq_with_spaces)
        assert validated == "ATGCGTAGCTAAGCTAGCAC"

    def test_sequence_validation_wrong_length(self, dummy_model):
        """Test sequence validation fails for non-20-nt sequences."""
        predictor = EfficiencyPredictor(dummy_model, feature_columns=FEATURE_COLUMNS)
        
        with pytest.raises(ValueError, match="exactly 20 nucleotides"):
            predictor.validate_sequence("ATGCGTAGCTAA")
        
        with pytest.raises(ValueError, match="exactly 20 nucleotides"):
            predictor.validate_sequence("ATGCGTAGCTAAGCTAGCACAA")

    def test_sequence_validation_invalid_nucleotides(self, dummy_model):
        """Test sequence validation fails for invalid nucleotides."""
        predictor = EfficiencyPredictor(dummy_model, feature_columns=FEATURE_COLUMNS)
        
        with pytest.raises(ValueError, match="only A, T, G, C"):
            predictor.validate_sequence("ATGCGTAGCTAAGCTAGCAXN")

    def test_sequence_validation_none(self, dummy_model):
        """Test sequence validation fails for None input."""
        predictor = EfficiencyPredictor(dummy_model, feature_columns=FEATURE_COLUMNS)
        
        with pytest.raises(ValueError, match="required"):
            predictor.validate_sequence(None)

    def test_predict_returns_valid_result(self, dummy_model):
        """Test prediction returns a valid PredictionResult."""
        predictor = EfficiencyPredictor(dummy_model, feature_columns=FEATURE_COLUMNS)
        seq = "ATGCGTAGCTAAGCTAGCAC"
        result = predictor.predict(seq)

        assert result.sequence == seq
        assert isinstance(result.prediction, float)
        assert isinstance(result.features, dict)
        assert len(result.features) > 0

    def test_predict_prediction_range(self, dummy_model):
        """Test prediction is a reasonable float value."""
        predictor = EfficiencyPredictor(dummy_model, feature_columns=FEATURE_COLUMNS)
        seq = "ATGCGTAGCTAAGCTAGCAC"
        result = predictor.predict(seq)

        # Predictions should be numeric and not NaN
        assert not np.isnan(result.prediction)
        assert isinstance(result.prediction, float)

    def test_predict_invalid_sequence(self, dummy_model):
        """Test prediction raises error for invalid sequence."""
        predictor = EfficiencyPredictor(dummy_model, feature_columns=FEATURE_COLUMNS)
        
        with pytest.raises(ValueError):
            predictor.predict("INVALIDSEQUENCE")

    def test_predict_consistency(self, dummy_model):
        """Test same sequence produces same prediction."""
        predictor = EfficiencyPredictor(dummy_model, feature_columns=FEATURE_COLUMNS)
        seq = "ATGCGTAGCTAAGCTAGCAC"
        
        result1 = predictor.predict(seq)
        result2 = predictor.predict(seq)
        
        assert result1.prediction == result2.prediction
