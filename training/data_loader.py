import numpy as np
import pandas as pd
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = REPOSITORY_ROOT / "data/raw/nhanes"
MORTALITY_DIR = REPOSITORY_ROOT / "data/raw/mortality"
PROCESSED_DIR = REPOSITORY_ROOT / "data/processed"
PROCESSED_PATH = PROCESSED_DIR / "demographics_mortality_clean.csv"


# Features identified in candidate_features.py / analyze_features.py as
# usable predictors. RIDRETH3 and DMDEDUC3 are intentionally excluded here:
# RIDRETH3 is ~60% missing because it wasn't collected before the 2007-2008
# cycle (RIDRETH1 covers every cycle instead), and DMDEDUC3 is ~93% missing
# because it only applies to participants aged 6-19.
CANDIDATE_FEATURES = [
    "RIDAGEYR",
    "RIAGENDR",
    "RIDRETH1",
    "DMDEDUC2",
    "DMDMARTL",
    "INDFMPIR",
    "DMDCITZN",
    "DMDYRSUS",
    "DMDHHSIZ",
]

# NHANES uses these numeric codes for "Refused" / "Don't know" answers on
# categorical fields. They are not real category values, so they need to be
# converted to NaN rather than treated as ordinary responses.
MISSING_CODES = {
    "DMDEDUC2": [7, 9],
    "DMDMARTL": [77, 99],
    "DMDCITZN": [7, 9],
    "DMDYRSUS": [77, 99],
}

# Drop any feature that is missing above this fraction of rows after the
# recoding above (this is what removes RIDRETH3 / DMDEDUC3 if callers pass
# the full merged dataset through clean_dataset instead of CANDIDATE_FEATURES).
MISSING_THRESHOLD = 0.5


DEMO_FILES = [
    "DEMO.xpt",
    "DEMO_B.xpt",
    "DEMO_C.xpt",
    "DEMO_D.xpt",
    "DEMO_E.xpt",
    "DEMO_F.xpt",
    "DEMO_G.xpt",
    "DEMO_H.xpt",
    "DEMO_I.xpt",
    "DEMO_J.xpt",
]


MORTALITY_FILES = [
    "NHANES_1999_2000_MORT_2019_PUBLIC.dat",
    "NHANES_2001_2002_MORT_2019_PUBLIC.dat",
    "NHANES_2003_2004_MORT_2019_PUBLIC.dat",
    "NHANES_2005_2006_MORT_2019_PUBLIC.dat",
    "NHANES_2007_2008_MORT_2019_PUBLIC.dat",
    "NHANES_2009_2010_MORT_2019_PUBLIC.dat",
    "NHANES_2011_2012_MORT_2019_PUBLIC.dat",
    "NHANES_2013_2014_MORT_2019_PUBLIC.dat",
    "NHANES_2015_2016_MORT_2019_PUBLIC.dat",
    "NHANES_2017_2018_MORT_2019_PUBLIC.dat",
]


def load_demographics():
    dataframes = []

    for filename in DEMO_FILES:
        file_path = DEMO_DIR / filename

        df = pd.read_sas(file_path)

        dataframes.append(df)

    return pd.concat(
        dataframes,
        ignore_index=True,
        sort=False
    )


def load_mortality():
    dataframes = []

    for filename in MORTALITY_FILES:
        file_path = MORTALITY_DIR / filename

        df = pd.read_fwf(
            file_path,
            colspecs=[
                (0, 14),   # SEQN
                (14, 15),  # ELIGSTAT
                (15, 16),  # MORTSTAT
            ],
            names=[
                "SEQN",
                "ELIGSTAT",
                "MORTSTAT",
            ],
            dtype=str,
        )

        dataframes.append(df)

    return pd.concat(
        dataframes,
        ignore_index=True
    )


def build_dataset():

    demographics = load_demographics()
    mortality = load_mortality()

    # Convert SEQN to the same numeric representation
    demographics["SEQN"] = pd.to_numeric(
        demographics["SEQN"],
        errors="coerce"
    )

    mortality["SEQN"] = pd.to_numeric(
        mortality["SEQN"],
        errors="coerce"
    )

    # Convert eligibility and mortality status
    mortality["ELIGSTAT"] = pd.to_numeric(
        mortality["ELIGSTAT"],
        errors="coerce"
    )

    mortality["MORTSTAT"] = pd.to_numeric(
        mortality["MORTSTAT"],
        errors="coerce"
    )

    # Keep only participants eligible for public-use mortality follow-up
    mortality = mortality[
        mortality["ELIGSTAT"] == 1
    ].copy()

    # Merge using SEQN
    merged = demographics.merge(
        mortality,
        on="SEQN",
        how="inner",
        validate="one_to_one"
    )

    return merged


def clean_dataset(df, features=CANDIDATE_FEATURES, missing_threshold=MISSING_THRESHOLD):
    """
    Clean the merged NHANES/mortality dataset using what
    candidate_features.py / analyze_features.py found:

    - pandas' SAS reader (read_sas) represents some SAS missing values as
      tiny denormalized floats (e.g. 5.397605e-79) instead of NaN. Those get
      converted to real NaN first so they aren't mistaken for valid values.
    - NHANES "Refused" (7/77) and "Don't know" (9/99) codes on categorical
      fields are converted to NaN.
    - Rows with no usable mortality outcome (MORTSTAT missing) are dropped.
    - Columns still missing above `missing_threshold` after recoding are
      dropped (this is what removes RIDRETH3/DMDEDUC3 if `features` isn't
      restricted to CANDIDATE_FEATURES).
    """

    df = df.copy()

    numeric_columns = df.select_dtypes(include="number").columns
    is_sentinel = (df[numeric_columns] != 0) & (df[numeric_columns].abs() < 1e-30)
    df[numeric_columns] = df[numeric_columns].mask(is_sentinel)

    for column, codes in MISSING_CODES.items():
        if column in df.columns:
            df[column] = df[column].replace(codes, np.nan)

    df = df.dropna(subset=["MORTSTAT"])

    keep_columns = [
        column for column in ["SEQN", "MORTSTAT"] + list(features)
        if column in df.columns
    ]
    df = df[keep_columns]

    missing_fraction = df.isna().mean()
    dropped = missing_fraction[missing_fraction > missing_threshold].index
    df = df.drop(columns=dropped)

    return df


def build_clean_dataset(features=CANDIDATE_FEATURES, missing_threshold=MISSING_THRESHOLD):
    return clean_dataset(
        build_dataset(),
        features=features,
        missing_threshold=missing_threshold,
    )


def save_processed_dataset(df, path=PROCESSED_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_processed_dataset(path=PROCESSED_PATH):
    return pd.read_csv(path)


if __name__ == "__main__":

    dataset = build_clean_dataset()
    saved_path = save_processed_dataset(dataset)

    print("=" * 70)
    print("FINAL INITIAL DATASET")
    print("=" * 70)

    print(f"Rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")

    print("\nMortality outcome:")
    print(dataset["MORTSTAT"].value_counts(dropna=False))

    print("\nMissing MORTSTAT:")
    print(dataset["MORTSTAT"].isna().sum())

    print("\nFirst 5 rows:")
    print(dataset.head())

    print(f"\nSaved processed dataset to: {saved_path}")