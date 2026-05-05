"""Preprocess Azimuth V1 data for model training"""

import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/v1_data.xlsx")
PROCESSED_PATH = Path("data/processed/v1_cleaned.csv")
SHEET_NAME = "Human"
TARGET_COLUMN = "TF1 CD13"


def load_raw_v1(sheet_name: str = SHEET_NAME) -> pd.DataFrame:
    # read the V1 Excel sheet for human data
    return pd.read_excel(RAW_PATH, sheet_name=sheet_name)


def clean_v1_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # drop columns that are not needed for training
    drop_columns = [
        "Target",
        "30mer",
        "Strand",
        "Transcript",
        "CutSite",
        "Annotation",
        "MOLM13 CD15",
        "MOLM13 CD33",
        "NB4 CD33",
        "TF1 CD33",
        "NB4 CD13",
    ]
    df = df.drop(columns=[col for col in drop_columns if col in df.columns])

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' not found in Azimuth V1 Human sheet.")

    # rename and normalize columns
    df = df.rename(columns={TARGET_COLUMN: "efficiency"})
    df.columns = df.columns.str.lower()

    df = df[["sequence", "efficiency"]]
    df = df.dropna(subset=["sequence", "efficiency"])
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    df = df[df["sequence"].str.len() == 20]
    return df


def save_processed(df: pd.DataFrame, out_path: Path = PROCESSED_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main() -> None:
    df = load_raw_v1()
    df_clean = clean_v1_dataset(df)
    save_processed(df_clean)


if __name__ == "__main__":
    main()
