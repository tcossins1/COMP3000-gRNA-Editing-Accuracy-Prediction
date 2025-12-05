import pandas as pd

def gc_content(seq: str) -> float:
    # Compute GC content of a nucleotide sequence
    seq = seq.upper()
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)

def extract_features_from_sequence(seq: str) -> dict:
    # Extract model features from a single raw sequence
    # Used for processing UI inputs
    return {
        "gc_content": gc_content(seq)
    }

def extract_features_from_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Apply feature extraction to preprocessed dataset
    # Used for training models
    df = df.copy()
    df["gc_content"] = df["sequence"].apply(gc_content)
    return df

def main():
    # For executing file solo - remove upon production
    df = pd.read_csv("data/processed/v1_cleaned.csv")
    df_features = extract_features_from_dataset(df)
    df_features.to_csv("data/processed/v1_features.csv", index=False)

if __name__ == "__main__":
    main()
