import pandas as pd

def gc_content(seq: str) -> float:
    # Compute GC content of a nucleotide sequence
    seq = seq.upper()
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)

def gc_window(seq: str, start: int, end: int) -> float:
    """GC content in seq[start:end] (0-indexed, end exclusive)."""
    seq = seq.upper()
    window = seq[start:end]
    if not window:
        return 0.0
    return (window.count("G") + window.count("C")) / len(window)

def has_poly_t4(seq: str) -> int:
    """1 if the sequence contains 'TTTT', else 0."""
    return 1 if "TTTT" in seq.upper() else 0

def max_homopolymer(seq: str) -> int:
    """Length of the longest run of the same nucleotide (e.g., AAAA -> 4)."""
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
    # Extract model features from a single raw sequence
    # Used for processing UI inputs
    seq = seq.upper()
    return {
        "gc_content": gc_content(seq),
        "gc_1_10": gc_window(seq, 0, 10),
        "gc_11_20": gc_window(seq, 10, 20),
        "has_poly_t4": has_poly_t4(seq),
        "max_homopolymer": max_homopolymer(seq),
    }

def extract_features_from_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Apply feature extraction to preprocessed dataset
    # Used for training models
    df = df.copy()
    features = df["sequence"].apply(extract_features_from_sequence).apply(pd.Series)
    return pd.concat([df, features], axis=1)

def main():
    # For executing file solo - remove upon production
    df = pd.read_csv("data/processed/v1_cleaned.csv")
    df_features = extract_features_from_dataset(df)
    df_features.to_csv("data/processed/v1_features.csv", index=False)

if __name__ == "__main__":
    main()
