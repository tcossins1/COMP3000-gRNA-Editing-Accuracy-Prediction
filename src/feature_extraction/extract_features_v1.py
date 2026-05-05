"""Feature extraction utilities for ForeCas9."""

import argparse
from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = [
    "gc_content",
    "gc_1_10",
    "gc_11_20",
    "has_poly_t4",
    "max_homopolymer",
    "gc_ratio_1_10_vs_11_20",
    "a_count",
    "t_count",
    "g_count",
    "c_count",
    "position_1_a",
    "position_1_t",
    "position_1_g",
    "position_1_c",
    "position_20_a",
    "position_20_t",
    "position_20_g",
    "position_20_c",
    "dinuc_ag",
    "dinuc_ct",
    "dinuc_gc",
    "dinuc_ta",
]


def gc_content(seq: str) -> float:
    # fraction of G/C in sequence
    seq = seq.upper()
    return 0.0 if not seq else (seq.count("G") + seq.count("C")) / len(seq)


def gc_window(seq: str, start: int, end: int) -> float:
    # GC content for a sub-window of the sequence
    window = seq.upper()[start:end]
    return 0.0 if not window else (window.count("G") + window.count("C")) / len(window)


def has_poly_t4(seq: str) -> int:
    # 1 if sequence contains a TTTT run
    return 1 if "TTTT" in seq.upper() else 0


def max_homopolymer(seq: str) -> int:
    # longest repeat of the same base
    seq = seq.upper()
    if not seq:
        return 0

    max_run = 1
    current_run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 1
    return max_run


def extract_features_from_sequence(seq: str) -> dict:
    seq = seq.upper()

    features = {
        "gc_content": gc_content(seq),
        "gc_1_10": gc_window(seq, 0, 10),
        "gc_11_20": gc_window(seq, 10, 20),
        "has_poly_t4": has_poly_t4(seq),
        "max_homopolymer": max_homopolymer(seq),
    }

    # additional features
    features["gc_ratio_1_10_vs_11_20"] = features["gc_1_10"] / (features["gc_11_20"] + 1e-6)
    features["a_count"] = seq.count("A")
    features["t_count"] = seq.count("T")
    features["g_count"] = seq.count("G")
    features["c_count"] = seq.count("C")

    # 1-based position features
    features["position_1_a"] = 1 if seq[0] == "A" else 0
    features["position_1_t"] = 1 if seq[0] == "T" else 0
    features["position_1_g"] = 1 if seq[0] == "G" else 0
    features["position_1_c"] = 1 if seq[0] == "C" else 0
    features["position_20_a"] = 1 if seq[19] == "A" else 0
    features["position_20_t"] = 1 if seq[19] == "T" else 0
    features["position_20_g"] = 1 if seq[19] == "G" else 0
    features["position_20_c"] = 1 if seq[19] == "C" else 0

    # selected dinucleotide counts
    features["dinuc_ag"] = seq.count("AG")
    features["dinuc_ct"] = seq.count("CT")
    features["dinuc_gc"] = seq.count("GC")
    features["dinuc_ta"] = seq.count("TA")

    return features


def extract_features_from_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # add feature columns to the cleaned dataset
    df = df.copy()
    feature_df = df["sequence"].apply(extract_features_from_sequence).apply(pd.Series)
    return pd.concat([df, feature_df], axis=1)


def build_feature_dataset(input_path: str | Path = "data/processed/v1_cleaned.csv", output_path: str | Path = "data/processed/v1_features.csv") -> None:
    df = pd.read_csv(input_path)
    df_features = extract_features_from_dataset(df)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract sequence features from a cleaned dataset")
    parser.add_argument(
        "--input-path",
        default="data/processed/v1_cleaned.csv",
        help="Path to the cleaned dataset CSV file",
    )
    parser.add_argument(
        "--output-path",
        default="data/processed/v1_features.csv",
        help="Path to write the extracted feature CSV file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_feature_dataset(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
