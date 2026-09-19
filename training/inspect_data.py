import pandas as pd
from pathlib import Path


DEMO_FILES = [
    "data/raw/nhanes/DEMO.XPT",
    "data/raw/nhanes/DEMO_B.XPT",
    "data/raw/nhanes/DEMO_C.XPT",
    "data/raw/nhanes/DEMO_D.XPT",
    "data/raw/nhanes/DEMO_E.XPT",
    "data/raw/nhanes/DEMO_F.XPT",
    "data/raw/nhanes/DEMO_G.XPT",
    "data/raw/nhanes/DEMO_H.XPT",
    "data/raw/nhanes/DEMO_I.XPT",
    "data/raw/nhanes/DEMO_J.XPT",
]


def load_demographics():
    dataframes = []

    for file_path in DEMO_FILES:
        print(f"Loading: {file_path}")

        df = pd.read_sas(file_path)

        # Keep track of which NHANES cycle this row came from
        df["NHANES_CYCLE"] = Path(file_path).stem

        dataframes.append(df)

        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")

    combined = pd.concat(
        dataframes,
        ignore_index=True,
        sort=False
    )

    return combined


def inspect_dataset(df):
    print("\n" + "=" * 70)
    print("COMBINED NHANES DEMOGRAPHICS DATASET")
    print("=" * 70)

    print(f"\nTotal rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nSEQN information:")
    print(f"Unique SEQN values: {df['SEQN'].nunique()}")
    print(f"Missing SEQN values: {df['SEQN'].isna().sum()}")

    print("\nRows per NHANES cycle:")
    print(df["NHANES_CYCLE"].value_counts().sort_index())

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    print(missing)


if __name__ == "__main__":
    demographics = load_demographics()
    inspect_dataset(demographics)