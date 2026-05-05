"""Unit tests for feature extraction."""
import pytest
from src.feature_extraction.extract_features_v1 import (
    extract_features_from_sequence,
    FEATURE_COLUMNS,
)


class TestFeatureExtraction:
    """Test feature extraction for 20-nt sequences."""

    def test_feature_extraction_valid_sequence(self):
        """Test feature extraction on a valid 20-nt sequence."""
        seq = "ATGCGTAGCTAAGCTAGCAC"
        features = extract_features_from_sequence(seq)

        assert isinstance(features, dict)
        assert "gc_content" in features
        assert "max_homopolymer" in features
        assert "has_poly_t4" in features

    def test_feature_extraction_all_features_present(self):
        """Test that all expected features are extracted."""
        seq = "ATGCGTAGCTAAGCTAGCAC"
        features = extract_features_from_sequence(seq)

        for col in FEATURE_COLUMNS:
            assert col in features, f"Missing feature: {col}"

    def test_feature_gc_content_range(self):
        """Test that GC content is between 0 and 1."""
        seq = "ATGCGTAGCTAAGCTAGCAC"
        features = extract_features_from_sequence(seq)

        assert 0 <= features["gc_content"] <= 1

    def test_feature_gc_content_all_g_c(self):
        """Test GC content is 1.0 for all-GC sequence."""
        seq = "GCGCGCGCGCGCGCGCGCGC"
        features = extract_features_from_sequence(seq)

        assert features["gc_content"] == 1.0

    def test_feature_gc_content_all_a_t(self):
        """Test GC content is 0.0 for all-AT sequence."""
        seq = "ATATATATATATATATATAT"
        features = extract_features_from_sequence(seq)

        assert features["gc_content"] == 0.0

    def test_feature_homopolymer_detection(self):
        """Test max homopolymer detection."""
        seq_short = "ATGCGTAGCTAAGCTAGCAC"
        features_short = extract_features_from_sequence(seq_short)
        assert features_short["max_homopolymer"] >= 1

        seq_long = "AAAAAAAAAATTTTTTTTTT"
        features_long = extract_features_from_sequence(seq_long)
        assert features_long["max_homopolymer"] == 10

    def test_feature_poly_t4_flag(self):
        """Test poly-T4 detection flag."""
        seq_with_poly_t = "ATTTGTAGCTAAGCTAGCAC"
        features_with = extract_features_from_sequence(seq_with_poly_t)
        assert features_with["has_poly_t4"] == 1

        seq_without_poly_t = "ATGCGTAGCTAAGCTAGCAC"
        features_without = extract_features_from_sequence(seq_without_poly_t)
        assert features_without["has_poly_t4"] == 0

    def test_feature_count_totals(self):
        """Test that nucleotide counts sum to 20."""
        seq = "ATGCGTAGCTAAGCTAGCAC"
        features = extract_features_from_sequence(seq)

        total_count = (
            features["a_count"]
            + features["t_count"]
            + features["g_count"]
            + features["c_count"]
        )
        assert total_count == 20

    def test_feature_position_flags_binary(self):
        """Test that position-specific flags are binary (0 or 1)."""
        seq = "ATGCGTAGCTAAGCTAGCAC"
        features = extract_features_from_sequence(seq)

        for i in range(1, 21):
            for nt in ["a", "t", "g", "c"]:
                key = f"position_{i}_{nt}"
                assert features[key] in [0, 1], f"{key} should be binary"

    def test_feature_position_sum_to_one(self):
        """Test that position-specific nucleotides sum to 1."""
        seq = "ATGCGTAGCTAAGCTAGCAC"
        features = extract_features_from_sequence(seq)

        for i in range(1, 21):
            pos_sum = (
                features[f"position_{i}_a"]
                + features[f"position_{i}_t"]
                + features[f"position_{i}_g"]
                + features[f"position_{i}_c"]
            )
            assert pos_sum == 1, f"Position {i} nucleotides should sum to 1"

    def test_feature_consistency_case_insensitive(self):
        """Test that features are consistent regardless of input case."""
        seq_lower = "atgcgtagctaagctagcac"
        seq_upper = "ATGCGTAGCTAAGCTAGCAC"

        features_lower = extract_features_from_sequence(seq_lower)
        features_upper = extract_features_from_sequence(seq_upper)

        assert features_lower["gc_content"] == features_upper["gc_content"]
        assert features_lower["max_homopolymer"] == features_upper["max_homopolymer"]
