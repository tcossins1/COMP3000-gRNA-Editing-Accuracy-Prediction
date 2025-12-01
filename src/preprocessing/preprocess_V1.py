# Preprocessing the dataset 'V1_data.xlsx'

import pandas as pd
from pathlib import Path

def load_raw_v1():
    raw_path = Path("data/raw/V1_data.xlsx")
    df = pd.read_excel(raw_path)
    return df

def clean_v1_dataset(df):

    # Dropping unneeded columns
    df = df.drop(columns=[
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
        "NB4 CD13"
    ])

    # Currently using Tf1 CD13 efficiency column
    df = df.rename(columns={"TF1 CD13": "efficiency"})

    # Rename needed columns to lower case for consistency
    df.columns = df.columns.str.lower()

    # Ensure only appropriate columns selected
    df = df[["sequence", "efficiency"]]

    # Drop unsuitable rows
    df = df.dropna(subset=["sequence", "efficiency"])
    df = df[df["sequence"].str.len() == 20]  # ensure valid gRNA length

    return df

def save_processed(df):
    out_path = Path("data/processed/v1_cleaned.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

def main():
    df = load_raw_v1()
    df_clean = clean_v1_dataset(df)
    save_processed(df_clean)

if __name__ == "__main__":
    main()
