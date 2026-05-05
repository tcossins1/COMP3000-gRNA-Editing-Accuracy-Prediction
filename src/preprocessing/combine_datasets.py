import pandas as pd
from pathlib import Path

V1_CLEANED_PATH = Path("data/processed/v1_cleaned.csv")
V2_CLEANED_PATH = Path("data/processed/v2_cleaned.csv")
COMBINED_CLEANED_PATH = Path("data/processed/combined_cleaned.csv")


def load_cleaned_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {path}")
    return pd.read_csv(path)


def combine_v1_v2(v1_df: pd.DataFrame, v2_df: pd.DataFrame) -> pd.DataFrame:
    v1 = v1_df.copy()
    v2 = v2_df.copy()
    v1["source_dataset"] = "v1"
    v2["source_dataset"] = "v2"
    combined = pd.concat([v1, v2], ignore_index=True)
    combined["sequence"] = combined["sequence"].astype(str).str.strip().str.upper()
    combined = combined.dropna(subset=["sequence", "efficiency"])
    combined = combined[combined["sequence"].str.len() == 20]
    combined = combined.drop_duplicates(subset=["sequence", "efficiency"])
    return combined


def save_combined_dataset(df: pd.DataFrame, out_path: Path = COMBINED_CLEANED_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main() -> None:
    v1_df = load_cleaned_dataset(V1_CLEANED_PATH)
    v2_df = load_cleaned_dataset(V2_CLEANED_PATH)
    combined = combine_v1_v2(v1_df, v2_df)
    save_combined_dataset(combined)
    print(f"Combined dataset saved to: {COMBINED_CLEANED_PATH}")
    print(f"Total rows: {len(combined)}")
    print(f"Source counts:\n{combined['source_dataset'].value_counts().to_string()}\n")


if __name__ == "__main__":
    main()
