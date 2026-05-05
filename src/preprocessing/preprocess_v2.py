import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/V2_data.xlsx")
PROCESSED_PATH = Path("data/processed/v2_cleaned.csv")
SHEET_NAME = "Results"
HEADER_ROW = 7
SEQUENCE_COLUMN = "Construct Barcode"
TARGET_COLUMN = "sgRNA Score"


def load_raw_v2(sheet_name: str = SHEET_NAME, header: int = HEADER_ROW) -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Azimuth V2 raw dataset not found: {RAW_PATH}")
    return pd.read_excel(RAW_PATH, sheet_name=sheet_name, header=header)


def clean_v2_dataset(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [col for col in [SEQUENCE_COLUMN, TARGET_COLUMN] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Expected columns not found in Azimuth V2 sheet: {missing_columns}")

    df = df[[SEQUENCE_COLUMN, TARGET_COLUMN]].copy()
    df = df.rename(columns={SEQUENCE_COLUMN: "sequence", TARGET_COLUMN: "efficiency"})
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    df["efficiency"] = pd.to_numeric(df["efficiency"], errors="coerce")
    df = df.dropna(subset=["sequence", "efficiency"])
    df = df[df["sequence"].str.len() == 20]
    df = df.drop_duplicates(subset=["sequence"])
    return df


def save_processed(df: pd.DataFrame, out_path: Path = PROCESSED_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main() -> None:
    df = load_raw_v2()
    df_clean = clean_v2_dataset(df)
    save_processed(df_clean)


if __name__ == "__main__":
    main()
