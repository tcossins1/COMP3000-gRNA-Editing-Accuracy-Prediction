import pandas as pd

def gc_content(seq):
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / len(seq) if len(seq) > 0 else 0

def extract_features(df):
    df["gc_content"] = df["sequence"].apply(gc_content)
    return df

def main():
    df = pd.read_csv("data/processed/v1_cleaned.csv")
    df_features = extract_features(df)
    df_features.to_csv("data/processed/v1_features.csv", index=False)

if __name__ == "__main__":
    main()
