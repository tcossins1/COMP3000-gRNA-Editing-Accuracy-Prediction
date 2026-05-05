"""Integration tests for the full ML pipeline."""
import pytest
from pathlib import Path
import pandas as pd
import numpy as np

from src.feature_extraction.extract_features_v1 import build_feature_dataset


class TestFullPipeline:
    """Integration tests for the complete pipeline."""

    def test_cleaned_v1_dataset_exists(self):
        """Test that V1 cleaned dataset exists and is valid."""
        v1_path = Path("data/processed/v1_cleaned.csv")
        assert v1_path.exists(), f"V1 cleaned dataset not found at {v1_path}"
        
        df = pd.read_csv(v1_path)
        assert len(df) > 0, "V1 cleaned dataset is empty"
        assert "sequence" in df.columns, "V1 cleaned dataset missing 'sequence' column"
        assert "efficiency" in df.columns, "V1 cleaned dataset missing 'efficiency' column"

    def test_cleaned_v2_dataset_exists(self):
        """Test that V2 cleaned dataset exists and is valid."""
        v2_path = Path("data/processed/v2_cleaned.csv")
        assert v2_path.exists(), f"V2 cleaned dataset not found at {v2_path}"
        
        df = pd.read_csv(v2_path)
        assert len(df) > 0, "V2 cleaned dataset is empty"
        assert "sequence" in df.columns, "V2 cleaned dataset missing 'sequence' column"
        assert "efficiency" in df.columns, "V2 cleaned dataset missing 'efficiency' column"

    def test_combined_dataset_exists(self):
        """Test that combined dataset exists and is valid."""
        combined_path = Path("data/processed/combined_cleaned.csv")
        assert combined_path.exists(), f"Combined cleaned dataset not found at {combined_path}"
        
        df = pd.read_csv(combined_path)
        assert len(df) > 0, "Combined cleaned dataset is empty"
        assert "sequence" in df.columns, "Combined dataset missing 'sequence' column"
        assert "efficiency" in df.columns, "Combined dataset missing 'efficiency' column"
        assert "source_dataset" in df.columns, "Combined dataset missing 'source_dataset' column"

    def test_v1_features_dataset_exists(self):
        """Test that V1 features dataset exists and is valid."""
        v1_features_path = Path("data/processed/v1_features.csv")
        assert v1_features_path.exists(), f"V1 features dataset not found at {v1_features_path}"
        
        df = pd.read_csv(v1_features_path)
        assert len(df) > 0, "V1 features dataset is empty"
        assert df.shape[1] > 1, "V1 features dataset should have multiple columns"

    def test_v2_features_dataset_exists(self):
        """Test that V2 features dataset exists and is valid."""
        v2_features_path = Path("data/processed/v2_features.csv")
        assert v2_features_path.exists(), f"V2 features dataset not found at {v2_features_path}"
        
        df = pd.read_csv(v2_features_path)
        assert len(df) > 0, "V2 features dataset is empty"
        assert df.shape[1] > 1, "V2 features dataset should have multiple columns"

    def test_combined_features_dataset_exists(self):
        """Test that combined features dataset exists and is valid."""
        combined_features_path = Path("data/processed/combined_features.csv")
        assert combined_features_path.exists(), f"Combined features dataset not found at {combined_features_path}"
        
        df = pd.read_csv(combined_features_path)
        assert len(df) > 0, "Combined features dataset is empty"
        assert df.shape[1] > 1, "Combined features dataset should have multiple columns"

    def test_feature_columns_consistency(self):
        """Test that all feature datasets have consistent feature columns."""
        v1_df = pd.read_csv(Path("data/processed/v1_features.csv"))
        v2_df = pd.read_csv(Path("data/processed/v2_features.csv"))
        combined_df = pd.read_csv(Path("data/processed/combined_features.csv"))

        # All should have the same number of columns (features + target)
        assert v1_df.shape[1] == v2_df.shape[1], "V1 and V2 feature columns mismatch"
        assert v1_df.shape[1] == combined_df.shape[1], "V1 and combined feature columns mismatch"

    def test_sequence_quality_v1(self):
        """Test V1 sequences are all exactly 20 nucleotides."""
        df = pd.read_csv(Path("data/processed/v1_cleaned.csv"))
        
        seq_lengths = df["sequence"].astype(str).str.len()
        assert (seq_lengths == 20).all(), "V1 sequences should all be 20 nt"

    def test_sequence_quality_v2(self):
        """Test V2 sequences are all exactly 20 nucleotides."""
        df = pd.read_csv(Path("data/processed/v2_cleaned.csv"))
        
        seq_lengths = df["sequence"].astype(str).str.len()
        assert (seq_lengths == 20).all(), "V2 sequences should all be 20 nt"

    def test_efficiency_range_v1(self):
        """Test V1 efficiency values are reasonable."""
        df = pd.read_csv(Path("data/processed/v1_cleaned.csv"))
        
        assert df["efficiency"].min() >= 0, "V1 efficiency should not be negative"
        assert df["efficiency"].max() <= 1, "V1 efficiency should not exceed 1"

    def test_efficiency_range_v2(self):
        """Test V2 efficiency values are reasonable."""
        df = pd.read_csv(Path("data/processed/v2_cleaned.csv"))
        
        assert df["efficiency"].min() >= 0, "V2 efficiency should not be negative"
        assert df["efficiency"].max() <= 1, "V2 efficiency should not exceed 1"

    def test_no_missing_values_v1_features(self):
        """Test V1 features dataset has no missing values."""
        df = pd.read_csv(Path("data/processed/v1_features.csv"))
        
        assert not df.isnull().any().any(), "V1 features should not have missing values"

    def test_no_missing_values_v2_features(self):
        """Test V2 features dataset has no missing values."""
        df = pd.read_csv(Path("data/processed/v2_features.csv"))
        
        assert not df.isnull().any().any(), "V2 features should not have missing values"

    def test_feature_values_numeric(self):
        """Test that all feature values are numeric."""
        df = pd.read_csv(Path("data/processed/v1_features.csv"))
        
        for col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} should be numeric"
