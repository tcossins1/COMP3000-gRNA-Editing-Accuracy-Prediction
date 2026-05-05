"""Shared test fixtures and configuration."""
import pytest
import pandas as pd
from pathlib import Path
import tempfile


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_sequences():
    """Sample valid 20-nt gRNA sequences."""
    return [
        "ATGCGTAGCTAAGCTAGCAC",
        "GCTAGCTAGCTAGCTAGCTA",
        "TTTTTTTTTTTTTTTTTTTT",
        "AAAAAAAAAAAAAAAAAAA",
    ]


@pytest.fixture
def sample_features_df():
    """Sample feature dataset for testing."""
    return pd.DataFrame({
        "gc_content": [0.5, 0.4, 0.6],
        "gc_1_10": [0.4, 0.3, 0.5],
        "gc_11_20": [0.6, 0.5, 0.7],
        "has_poly_t4": [0, 1, 0],
        "max_homopolymer": [4, 3, 5],
        "gc_ratio_1_10_vs_11_20": [0.67, 0.6, 0.71],
        "a_count": [5, 4, 6],
        "t_count": [5, 6, 4],
        "g_count": [5, 5, 5],
        "c_count": [5, 5, 5],
        "position_1_a": [1, 0, 0],
        "position_1_t": [0, 1, 0],
        "position_1_g": [0, 0, 1],
        "position_1_c": [0, 0, 0],
        "position_20_a": [0, 1, 0],
        "position_20_t": [1, 0, 0],
        "position_20_g": [0, 0, 1],
        "position_20_c": [0, 0, 0],
        "dinuc_ag": [2, 1, 3],
        "dinuc_ct": [2, 2, 1],
        "dinuc_gc": [2, 2, 3],
        "dinuc_ta": [1, 2, 1],
        "efficiency": [0.5, 0.6, 0.7],
    })
